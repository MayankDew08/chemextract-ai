"""NetworkX graph backend for Phase 4 storage.

This file contains NetworkXStore. It provides the fastest local backend and is
also used as the query mirror for Obsidian and Neo4j so all backends share query
semantics.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import networkx as nx

from src.schemas.chemical import ChemicalEntity
from src.schemas.paper import PaperMetadata
from src.schemas.recipe import ChemicalRecipe
from src.storage.base import (
    BaseGraphStore,
    NODE_COLORS,
    CoOccurrence,
    GraphEdge,
    GraphJSON,
    GraphNode,
    NodeType,
    RelationshipType,
    SolventFrequency,
    get_chemical_color,
)

logger = logging.getLogger(__name__)


class NetworkXStore(BaseGraphStore):
    """NetworkXStore keeps recipes in an idempotent in-memory directed graph."""

    @property
    def backend_name(self) -> str:
        """Return the backend identifier used by the resolver and demos."""

        return "networkx"

    def __init__(self) -> None:
        """Delay graph allocation until initialize so lifecycle is explicit."""

        self._graph: Optional[nx.DiGraph] = None
        self._recipes: dict[str, ChemicalRecipe] = {}

    def initialize(self) -> None:
        """Create a fresh directed graph; safe to call repeatedly."""

        if self._graph is None:
            self._graph = nx.DiGraph()
        logger.info("NetworkXStore initialized with empty graph")

    def close(self) -> None:
        """Clear in-memory graph state for clean shutdown."""

        self._graph = None
        self._recipes = {}
        logger.info("NetworkXStore closed")

    def clear(self) -> None:
        """Remove all in-memory graph and recipe state while keeping the store open."""

        graph = self._require_graph()
        graph.clear()
        self._recipes = {}

    def _chemical_id(self, name: str) -> str:
        """Return the canonical chemical node ID used by every backend."""

        return f"chem::{name.lower().strip().replace(' ', '_')}"

    def _recipe_id_to_node_id(self, recipe_id: str) -> str:
        """Return the canonical reaction node ID used by every backend."""

        return f"rxn::{recipe_id}"

    def _paper_id(self, metadata: PaperMetadata) -> str:
        """Return the canonical paper node ID from DOI or title hash."""

        if metadata.doi:
            slug = metadata.doi.replace("/", "_").replace(".", "_")
            return f"paper::{slug}"
        title_hash = hashlib.md5(metadata.title.lower().encode(), usedforsecurity=False).hexdigest()[:12]
        return f"paper::{title_hash}"

    def write_recipe(self, recipe: ChemicalRecipe) -> str:
        """Create or update a reaction node and cache its full recipe."""

        graph = self._require_graph()
        node_id = self._recipe_id_to_node_id(recipe.recipe_id)
        recipe_payload = recipe.model_dump(mode="json")
        validation_warnings = recipe_payload.get("validation_warnings", [])
        graph.add_node(
            node_id,
            node_type=NodeType.REACTION.value,
            label=recipe.title or f"Reaction {recipe.recipe_id}",
            color=NODE_COLORS[NodeType.REACTION],
            size=8.0,
            recipe_id=recipe.recipe_id,
            validation_status=recipe.validation_status.value,
            correction_count=recipe.correction_count(),
            was_corrected=recipe.was_corrected(),
            llm_model=recipe.llm_model,
            extracted_at=recipe.extracted_at.isoformat(),
            source_url=recipe.source_paper_url or "",
            warning_count=len(validation_warnings),
            validation_warnings=validation_warnings,
            source_text_completeness=recipe_payload.get("source_text_completeness"),
            source_acquisition_method=recipe_payload.get("source_acquisition_method"),
            node_extraction_methods=recipe_payload.get("node_extraction_methods", {}),
        )
        self._recipes[recipe.recipe_id] = recipe
        return node_id

    def write_chemical(self, entity: ChemicalEntity, recipe_id: str) -> str:
        """Create or update a chemical node without duplicating chemical names."""

        graph = self._require_graph()
        node_id = self._chemical_id(entity.name)
        recipes = set(graph.nodes[node_id].get("recipe_ids", [])) if graph.has_node(node_id) else set()
        recipes.add(recipe_id)
        reaction_count = len(recipes)
        graph.add_node(
            node_id,
            node_type=NodeType.CHEMICAL.value,
            label=entity.name,
            color=get_chemical_color(entity.role.value),
            size=min(5.0 + reaction_count * 2, 50.0),
            role=entity.role.value,
            formula=entity.formula or "",
            reaction_count=reaction_count,
            recipe_ids=sorted(recipes),
            name=entity.name,
        )
        return node_id

    def write_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        properties: dict,
    ) -> None:
        """Create or update a typed relationship using enum-only relation types."""

        graph = self._require_graph()
        if not graph.has_node(source_id) or not graph.has_node(target_id):
            logger.warning("Relationship skipped because a node is missing: %s -> %s", source_id, target_id)
            return
        edge_key = (source_id, target_id)
        if graph.has_edge(*edge_key):
            edge_data = graph.edges[edge_key]
            if edge_data.get("relationship_type") == relationship_type.value:
                if relationship_type == RelationshipType.CO_OCCURS_WITH:
                    recipe_ids = set(edge_data.get("recipe_ids", []))
                    if properties.get("recipe_id"):
                        recipe_ids.add(properties["recipe_id"])
                    edge_data["recipe_ids"] = sorted(recipe_ids)
                    edge_data["weight"] = max(1, len(recipe_ids))
                    edge_data["width"] = edge_data["weight"] / 2.0
                edge_data.update({k: v for k, v in properties.items() if k != "recipe_id"})
                return
        edge_properties = dict(properties)
        recipe_id = edge_properties.pop("recipe_id", None)
        graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type.value,
            label=relationship_type.value,
            weight=1,
            width=1.0,
            recipe_ids=[recipe_id] if recipe_id else [],
            **edge_properties,
        )

    def write_paper(self, metadata: PaperMetadata) -> str:
        """Create or update a paper node from DOI or title hash."""

        graph = self._require_graph()
        node_id = self._paper_id(metadata)
        graph.add_node(
            node_id,
            node_type=NodeType.PAPER.value,
            label=f"{metadata.title[:40]}..." if len(metadata.title) > 40 else metadata.title,
            color=NODE_COLORS[NodeType.PAPER],
            size=6.0,
            title=metadata.title,
            doi=metadata.doi or "",
            year=metadata.year or 0,
            source_url=metadata.open_access_url or "",
        )
        return node_id

    def get_recipe(self, recipe_id: str) -> Optional[ChemicalRecipe]:
        """Return a stored recipe by recipe_id."""

        return self._recipes.get(recipe_id)

    def get_all_recipes(self) -> list[ChemicalRecipe]:
        """Return every recipe cached by this in-memory store."""

        return list(self._recipes.values())

    def query_solvents_for(self, compound_name: str) -> list[SolventFrequency]:
        """Find solvents participating in reactions that produce the compound."""

        graph = self._require_graph()
        product_node_id = self._chemical_id(compound_name)
        if not graph.has_node(product_node_id):
            return []
        reaction_ids = [
            source
            for source, target, data in graph.edges(data=True)
            if target == product_node_id and data.get("relationship_type") == RelationshipType.PRODUCES.value
        ]
        solvent_counts: dict[str, int] = {}
        solvent_examples: dict[str, list[str]] = {}
        for reaction_id in reaction_ids:
            for source, target, data in graph.edges(data=True):
                if target == reaction_id and data.get("relationship_type") == RelationshipType.IS_SOLVENT_IN.value:
                    solvent_name = graph.nodes[source].get("name", source)
                    solvent_counts[solvent_name] = solvent_counts.get(solvent_name, 0) + 1
                    solvent_examples.setdefault(solvent_name, [])
                    recipe_id = reaction_id.replace("rxn::", "")
                    if len(solvent_examples[solvent_name]) < 3:
                        solvent_examples[solvent_name].append(recipe_id)
        total = sum(solvent_counts.values()) or 1
        return sorted(
            [
                SolventFrequency(
                    solvent_name=name,
                    frequency=count,
                    percentage=round(count / total * 100, 1),
                    example_recipe_ids=solvent_examples.get(name, []),
                )
                for name, count in solvent_counts.items()
            ],
            key=lambda item: item.frequency,
            reverse=True,
        )

    def query_recipes_for_chemical(self, chemical_name: str) -> list[ChemicalRecipe]:
        """Return recipes where a chemical participates or is produced."""

        graph = self._require_graph()
        chemical_id = self._chemical_id(chemical_name)
        if not graph.has_node(chemical_id):
            return []
        recipe_ids = set()
        for _source, target, _data in graph.out_edges(chemical_id, data=True):
            if graph.nodes[target].get("node_type") == NodeType.REACTION.value:
                recipe_ids.add(graph.nodes[target].get("recipe_id"))
        for source, _target, _data in graph.in_edges(chemical_id, data=True):
            if graph.nodes[source].get("node_type") == NodeType.REACTION.value:
                recipe_ids.add(graph.nodes[source].get("recipe_id"))
        return [self._recipes[recipe_id] for recipe_id in sorted(recipe_ids) if recipe_id in self._recipes]

    def query_cooccurring_chemicals(self, chemical_name: str, limit: int = 10) -> list[CoOccurrence]:
        """Return chemicals sharing recipes with the query chemical."""

        graph = self._require_graph()
        chemical_id = self._chemical_id(chemical_name)
        if not graph.has_node(chemical_id):
            return []
        query_recipes = {recipe.recipe_id for recipe in self.query_recipes_for_chemical(chemical_name)}
        cooccurrences: dict[str, dict] = {}
        for node_id, data in graph.nodes(data=True):
            if node_id == chemical_id or data.get("node_type") != NodeType.CHEMICAL.value:
                continue
            other_recipes = {recipe.recipe_id for recipe in self.query_recipes_for_chemical(data.get("name", node_id))}
            shared = query_recipes & other_recipes
            if shared:
                cooccurrences[data.get("name", node_id)] = {
                    "count": len(shared),
                    "types": sorted(self._relationship_types_between_recipe_sets(node_id, shared)),
                }
        return sorted(
            [
                CoOccurrence(
                    chemical_name=name,
                    co_occurrence_count=data["count"],
                    relationship_types=data["types"],
                )
                for name, data in cooccurrences.items()
            ],
            key=lambda item: item.co_occurrence_count,
            reverse=True,
        )[:limit]

    def export_graph_json(self) -> GraphJSON:
        """Export nodes and edges in the shared GraphJSON schema."""

        graph = self._require_graph()
        nodes = [
            GraphNode(
                id=node_id,
                label=data.get("label", node_id),
                node_type=NodeType(data.get("node_type", "chemical")),
                color=data.get("color", "#6b7280"),
                size=data.get("size", 10.0),
                properties={k: _json_safe(v) for k, v in data.items() if k not in {"label", "node_type", "color", "size"}},
            )
            for node_id, data in graph.nodes(data=True)
        ]
        edges = [
            GraphEdge(
                id=f"edge_{index}",
                source=source,
                target=target,
                relationship_type=RelationshipType(data.get("relationship_type", RelationshipType.CO_OCCURS_WITH.value)),
                label=data.get("label", ""),
                width=data.get("width", 1.0),
                properties={k: _json_safe(v) for k, v in data.items() if k not in {"label", "relationship_type", "width"}},
            )
            for index, (source, target, data) in enumerate(graph.edges(data=True))
        ]
        return GraphJSON(
            nodes=nodes,
            edges=edges,
            metadata={"backend": self.backend_name, "node_count": len(nodes), "edge_count": len(edges), "recipe_count": self.recipe_count()},
        )

    def node_count(self) -> int:
        """Return the number of graph nodes."""

        return self._graph.number_of_nodes() if self._graph else 0

    def edge_count(self) -> int:
        """Return the number of graph edges."""

        return self._graph.number_of_edges() if self._graph else 0

    def recipe_count(self) -> int:
        """Return the number of stored recipes."""

        return len(self._recipes)

    def _relationship_types_between_recipe_sets(self, node_id: str, recipe_ids: set[str]) -> set[str]:
        """Collect relationship types connecting a chemical to shared recipes."""

        graph = self._require_graph()
        types = set()
        reaction_nodes = {self._recipe_id_to_node_id(recipe_id) for recipe_id in recipe_ids}
        for source, target, data in graph.edges(data=True):
            if (source == node_id and target in reaction_nodes) or (target == node_id and source in reaction_nodes):
                types.add(data.get("relationship_type", ""))
        return types

    def _require_graph(self) -> nx.DiGraph:
        """Return initialized graph or raise a clear lifecycle error."""

        if self._graph is None:
            raise RuntimeError("NetworkXStore is not initialized")
        return self._graph


def _json_safe(value: object) -> object:
    """Convert sets and unsupported objects into JSON-friendly values."""

    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    return str(value)
