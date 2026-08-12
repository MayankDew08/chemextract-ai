"""Abstract graph storage contract and graph export schemas.

This file contains BaseGraphStore plus the shared node, edge, query-result, and
visualization schemas. Every storage backend implements this exact interface so
pipeline code can switch backends through configuration without backend imports.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.chemical import ChemicalEntity
from src.schemas.paper import PaperMetadata
from src.schemas.recipe import ChemicalRecipe

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """RelationshipType constrains graph edges to known chemistry relations."""

    IS_REACTANT_IN = "IS_REACTANT_IN"
    IS_SOLVENT_IN = "IS_SOLVENT_IN"
    IS_CATALYST_IN = "IS_CATALYST_IN"
    IS_ADDITIVE_IN = "IS_ADDITIVE_IN"
    PRODUCES = "PRODUCES"
    SOURCED_FROM = "SOURCED_FROM"
    HAS_CONDITIONS = "HAS_CONDITIONS"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"


class NodeType(str, Enum):
    """NodeType keeps visualization and query grouping consistent."""

    CHEMICAL = "chemical"
    REACTION = "reaction"
    PAPER = "paper"
    CONDITION = "condition"


class SolventFrequency(BaseModel):
    """SolventFrequency answers solvent usage questions for a target product."""

    solvent_name: str
    frequency: int
    percentage: float
    example_recipe_ids: list[str] = Field(default_factory=list)


class CoOccurrence(BaseModel):
    """CoOccurrence summarizes chemicals appearing near a query chemical."""

    chemical_name: str
    co_occurrence_count: int
    relationship_types: list[str]


class GraphNode(BaseModel):
    """GraphNode is the vis.js-compatible node abstraction."""

    id: str
    label: str
    node_type: NodeType
    color: str
    size: float = Field(default=10.0)
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """GraphEdge is the vis.js-compatible edge abstraction."""

    id: str
    source: str
    target: str
    relationship_type: RelationshipType
    label: str
    width: float = Field(default=1.0)
    properties: dict = Field(default_factory=dict)


class GraphJSON(BaseModel):
    """GraphJSON exports the full graph in a format Phase 5 can feed to vis.js."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict = Field(default_factory=dict)

    def to_visjs_dict(self) -> dict:
        """Convert graph data into the exact shape expected by vis.js Network."""

        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "color": node.color,
                    "size": node.size,
                    "title": f"{node.node_type.value}: {node.label}",
                    "group": node.node_type.value,
                    **node.properties,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "from": edge.source,
                    "to": edge.target,
                    "label": edge.label,
                    "width": edge.width,
                    **edge.properties,
                }
                for edge in self.edges
            ],
            "metadata": self.metadata,
        }


NODE_COLORS = {
    NodeType.CHEMICAL: {
        "REACTANT": "#ef4444",
        "SOLVENT": "#3b82f6",
        "CATALYST": "#22c55e",
        "ADDITIVE": "#a855f7",
        "PRODUCT": "#f59e0b",
        "UNKNOWN": "#6b7280",
    },
    NodeType.REACTION: "#06b6d4",
    NodeType.PAPER: "#64748b",
    NodeType.CONDITION: "#84cc16",
}


def get_chemical_color(role: str) -> str:
    """Return the visualization color for a chemical role."""

    return NODE_COLORS[NodeType.CHEMICAL].get(role, "#6b7280")


class BaseGraphStore(ABC):
    """Abstract interface implemented by every graph storage backend."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the stable backend identifier."""

    @abstractmethod
    def initialize(self) -> None:
        """Set up directories, in-memory graphs, or database connections idempotently."""

    @abstractmethod
    def close(self) -> None:
        """Flush or close backend resources without raising during shutdown."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every recipe, node, and relationship owned by this backend."""

    @abstractmethod
    def write_recipe(self, recipe: ChemicalRecipe) -> str:
        """Persist one recipe node and return its graph node ID."""

    @abstractmethod
    def write_chemical(self, entity: ChemicalEntity, recipe_id: str) -> str:
        """Persist one chemical node and return its graph node ID."""

    @abstractmethod
    def write_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        properties: dict,
    ) -> None:
        """Create or update one typed relationship between existing nodes."""

    @abstractmethod
    def write_paper(self, metadata: PaperMetadata) -> str:
        """Persist one source paper node and return its graph node ID."""

    @abstractmethod
    def get_recipe(self, recipe_id: str) -> Optional[ChemicalRecipe]:
        """Fetch one recipe by recipe_id or None if absent."""

    @abstractmethod
    def get_all_recipes(self) -> list[ChemicalRecipe]:
        """Return all stored recipes for batch operations and demos."""

    @abstractmethod
    def query_solvents_for(self, compound_name: str) -> list[SolventFrequency]:
        """Return solvent frequencies for reactions producing a compound."""

    @abstractmethod
    def query_recipes_for_chemical(self, chemical_name: str) -> list[ChemicalRecipe]:
        """Return recipes containing a chemical in any role."""

    @abstractmethod
    def query_cooccurring_chemicals(self, chemical_name: str, limit: int = 10) -> list[CoOccurrence]:
        """Return chemicals frequently appearing in the same reactions."""

    @abstractmethod
    def export_graph_json(self) -> GraphJSON:
        """Export the graph for vis.js visualization."""

    @abstractmethod
    def node_count(self) -> int:
        """Return total nodes in the backend graph."""

    @abstractmethod
    def edge_count(self) -> int:
        """Return total relationships in the backend graph."""

    @abstractmethod
    def recipe_count(self) -> int:
        """Return count of stored reaction recipes."""

    def write_recipe_full(self, recipe: ChemicalRecipe) -> str:
        """Write a recipe, paper, chemicals, role edges, and co-occurrence edges."""

        recipe_node_id = self.write_recipe(recipe)
        if recipe.source_paper_doi or recipe.source_paper_title or recipe.source_paper_url:
            paper_meta = PaperMetadata(
                title=recipe.source_paper_title or "Unknown",
                doi=recipe.source_paper_doi,
                source_db="extracted",
                open_access=bool(recipe.source_paper_url),
                open_access_url=recipe.source_paper_url,
            )
            paper_id = self.write_paper(paper_meta)
            self.write_relationship(recipe_node_id, paper_id, RelationshipType.SOURCED_FROM, {})

        chemical_ids = []
        for entity in recipe.entities:
            chemical_id = self.write_chemical(entity, recipe.recipe_id)
            chemical_ids.append((entity, chemical_id))
            rel_type = {
                "REACTANT": RelationshipType.IS_REACTANT_IN,
                "SOLVENT": RelationshipType.IS_SOLVENT_IN,
                "CATALYST": RelationshipType.IS_CATALYST_IN,
                "ADDITIVE": RelationshipType.IS_ADDITIVE_IN,
                "PRODUCT": RelationshipType.PRODUCES,
            }.get(entity.role.value, RelationshipType.IS_REACTANT_IN)
            if entity.role.value == "PRODUCT":
                self.write_relationship(recipe_node_id, chemical_id, RelationshipType.PRODUCES, {})
            else:
                self.write_relationship(chemical_id, recipe_node_id, rel_type, {})

        non_product_ids = [(entity, node_id) for entity, node_id in chemical_ids if entity.role.value != "PRODUCT"]
        for index, (_entity_1, id_1) in enumerate(non_product_ids):
            for _entity_2, id_2 in non_product_ids[index + 1 :]:
                if id_1 != id_2:
                    self.write_relationship(id_1, id_2, RelationshipType.CO_OCCURS_WITH, {"recipe_id": recipe.recipe_id})
        return recipe_node_id
