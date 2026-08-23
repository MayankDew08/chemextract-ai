"""Neo4j graph backend for Phase 4 storage.

This file contains Neo4jStore. It uses Neo4j MERGE operations for idempotent
persistence and maintains a NetworkX mirror for local query/export parity.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from src.schemas.chemical import ChemicalEntity
from src.schemas.paper import PaperMetadata
from src.schemas.recipe import ChemicalRecipe
from src.storage.base import BaseGraphStore, CoOccurrence, GraphJSON, RelationshipType, SolventFrequency
from src.storage.networkx_store import NetworkXStore

logger = logging.getLogger(__name__)


class Neo4jStore(BaseGraphStore):
    """Neo4jStore persists recipes to a Neo4j database with a NetworkX mirror."""

    @property
    def backend_name(self) -> str:
        """Return the backend identifier used by resolver and demos."""

        return "neo4j"

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Store connection settings without importing neo4j until initialize."""

        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "chemextract123")
        self._driver = None
        self._nx_mirror = NetworkXStore()
        self._recipes_cache: dict[str, ChemicalRecipe] = {}

    def initialize(self) -> None:
        """Connect to Neo4j and create uniqueness constraints idempotently."""

        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
        self._driver.verify_connectivity()
        self._nx_mirror.initialize()
        self._create_constraints()
        self._hydrate_recipe_cache()
        logger.info("Neo4jStore connected to %s", self._uri)

    def _create_constraints(self) -> None:
        """Create uniqueness constraints for stable graph node IDs."""

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ChemExtractChemical) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ChemExtractReaction) REQUIRE r.recipe_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:ChemExtractPaper) REQUIRE p.node_id IS UNIQUE",
        ]
        with self._driver.session() as session:
            for constraint in constraints:
                session.run(constraint)

    def _hydrate_recipe_cache(self) -> None:
        """Rebuild the query mirror from durable recipe payloads in Neo4j."""

        with self._driver.session() as session:
            records = list(
                session.run(
                    "MATCH (r:ChemExtractReaction {chemextract_managed: true}) "
                    "WHERE r.recipe_json IS NOT NULL "
                    "RETURN r.recipe_json AS recipe_json"
                )
            )
        loaded: dict[str, ChemicalRecipe] = {}
        for record in records:
            try:
                recipe = ChemicalRecipe.model_validate_json(record["recipe_json"])
            except Exception as exc:
                logger.warning("Skipping invalid persisted Neo4j recipe payload: %s", exc)
                continue
            self._nx_mirror.write_recipe_full(recipe)
            loaded[recipe.recipe_id] = recipe
        self._recipes_cache = loaded

    def write_recipe(self, recipe: ChemicalRecipe) -> str:
        """MERGE a reaction node and mirror it locally."""

        node_id = f"rxn::{recipe.recipe_id}"
        recipe_payload = recipe.model_dump(mode="json")
        validation_warnings = recipe_payload.get("validation_warnings", [])
        cypher = """
        MERGE (r:ChemExtract:ChemExtractReaction {recipe_id: $recipe_id})
        SET r.node_id = $node_id,
            r.chemextract_managed = true,
            r.title = $title,
            r.validation_status = $status,
            r.correction_count = $corrections,
            r.llm_model = $model,
            r.extracted_at = $extracted_at,
            r.source_url = $source_url,
            r.warning_count = $warning_count,
            r.validation_warnings = $validation_warnings,
            r.source_text_completeness = $source_text_completeness,
            r.source_acquisition_method = $source_acquisition_method,
            r.node_extraction_methods = $node_extraction_methods,
            r.recipe_json = $recipe_json,
            r.node_type = 'reaction'
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                {
                    "recipe_id": recipe.recipe_id,
                    "node_id": node_id,
                    "title": recipe.title or f"Reaction {recipe.recipe_id}",
                    "status": recipe.validation_status.value,
                    "corrections": recipe.correction_count(),
                    "model": recipe.llm_model,
                    "extracted_at": recipe.extracted_at.isoformat(),
                    "source_url": recipe.source_paper_url or "",
                    "warning_count": len(validation_warnings),
                    "validation_warnings": json.dumps(validation_warnings),
                    "source_text_completeness": recipe_payload.get("source_text_completeness"),
                    "source_acquisition_method": recipe_payload.get("source_acquisition_method"),
                    "node_extraction_methods": json.dumps(recipe_payload.get("node_extraction_methods", {})),
                    "recipe_json": recipe.model_dump_json(),
                },
            )
        self._nx_mirror.write_recipe(recipe)
        self._recipes_cache = {**self._recipes_cache, recipe.recipe_id: recipe}
        return node_id

    def write_chemical(self, entity: ChemicalEntity, recipe_id: str) -> str:
        """MERGE a chemical node and mirror it locally."""

        node_id = f"chem::{entity.name.lower().strip().replace(' ', '_')}"
        cypher = """
        MERGE (c:ChemExtract:ChemExtractChemical {node_id: $node_id})
        SET c.name = $name,
            c.chemextract_managed = true,
            c.formula = $formula,
            c.role = $role,
            c.node_type = 'chemical'
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                {"node_id": node_id, "name": entity.name, "formula": entity.formula or "", "role": entity.role.value},
            )
        self._nx_mirror.write_chemical(entity, recipe_id)
        return node_id

    def write_relationship(self, source_id: str, target_id: str, relationship_type: RelationshipType, properties: dict) -> None:
        """MERGE an enum-typed relationship and mirror it locally."""

        rel_name = relationship_type.value.upper()
        cypher = f"""
        MATCH (a:ChemExtract {{node_id: $source_id, chemextract_managed: true}})
        MATCH (b:ChemExtract {{node_id: $target_id, chemextract_managed: true}})
        MERGE (a)-[r:{rel_name}]->(b)
        SET r += $props
        """
        with self._driver.session() as session:
            session.run(cypher, {"source_id": source_id, "target_id": target_id, "props": properties})
        self._nx_mirror.write_relationship(source_id, target_id, relationship_type, properties)

    def write_paper(self, metadata: PaperMetadata) -> str:
        """MERGE a paper node and mirror it locally."""

        node_id = self._nx_mirror._paper_id(metadata)
        cypher = """
        MERGE (p:ChemExtract:ChemExtractPaper {node_id: $node_id})
        SET p.title = $title,
            p.chemextract_managed = true,
            p.doi = $doi,
            p.year = $year,
            p.source_url = $source_url,
            p.node_type = 'paper'
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                {
                    "node_id": node_id,
                    "title": metadata.title,
                    "doi": metadata.doi or "",
                    "year": metadata.year or 0,
                    "source_url": metadata.open_access_url or "",
                },
            )
        self._nx_mirror.write_paper(metadata)
        return node_id

    def query_solvents_for(self, compound_name: str) -> list[SolventFrequency]:
        """Delegate solvent query to the NetworkX mirror for parity."""

        return self._nx_mirror.query_solvents_for(compound_name)

    def query_recipes_for_chemical(self, chemical_name: str) -> list[ChemicalRecipe]:
        """Delegate recipe query to the NetworkX mirror."""

        return self._nx_mirror.query_recipes_for_chemical(chemical_name)

    def query_cooccurring_chemicals(self, chemical_name: str, limit: int = 10) -> list[CoOccurrence]:
        """Delegate co-occurrence query to the NetworkX mirror."""

        return self._nx_mirror.query_cooccurring_chemicals(chemical_name, limit)

    def export_graph_json(self) -> GraphJSON:
        """Delegate graph export to the NetworkX mirror."""

        return self._nx_mirror.export_graph_json()

    def get_recipe(self, recipe_id: str) -> Optional[ChemicalRecipe]:
        """Return cached recipe written in this process."""

        return self._recipes_cache.get(recipe_id)

    def get_all_recipes(self) -> list[ChemicalRecipe]:
        """Return cached recipes written in this process."""

        return list(self._recipes_cache.values())

    def node_count(self) -> int:
        """Return Neo4j node count."""

        with self._driver.session() as session:
            return session.run("MATCH (n:ChemExtract {chemextract_managed: true}) RETURN count(n) as count").single()["count"]

    def edge_count(self) -> int:
        """Return Neo4j relationship count."""

        with self._driver.session() as session:
            return session.run(
                "MATCH (:ChemExtract {chemextract_managed: true})-[r]->(:ChemExtract {chemextract_managed: true}) "
                "RETURN count(r) as count"
            ).single()["count"]

    def recipe_count(self) -> int:
        """Return Neo4j reaction count."""

        with self._driver.session() as session:
            return session.run(
                "MATCH (r:ChemExtractReaction {chemextract_managed: true}) RETURN count(r) as count"
            ).single()["count"]

    def close(self) -> None:
        """Close Neo4j connection and mirror state."""

        if self._driver:
            self._driver.close()
        self._nx_mirror.close()
        logger.info("Neo4jStore disconnected")

    def clear(self) -> None:
        """Delete all Neo4j records and reset the process-local query mirror."""

        if self._driver is None:
            raise RuntimeError("Neo4jStore is not initialized")
        with self._driver.session() as session:
            session.run("MATCH (n:ChemExtract {chemextract_managed: true}) DETACH DELETE n")
        self._nx_mirror.clear()
        self._recipes_cache = {}
