"""Obsidian vault graph backend for Phase 4 storage.

This file contains ObsidianVaultStore. It persists human-readable markdown notes
for durability and delegates graph queries to an internal NetworkXStore so query
semantics stay identical to the in-memory backend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.schemas.chemical import ChemicalEntity
from src.schemas.paper import PaperMetadata
from src.schemas.recipe import ChemicalRecipe
from src.storage.base import BaseGraphStore, CoOccurrence, GraphJSON, RelationshipType, SolventFrequency
from src.storage.markdown_writer import (
    append_recipe_link_to_chemical,
    generate_chemical_markdown,
    generate_paper_markdown,
    generate_recipe_markdown,
    parse_recipe_from_markdown,
    sanitize_filename,
)
from src.storage.networkx_store import NetworkXStore

logger = logging.getLogger(__name__)


class ObsidianVaultStore(BaseGraphStore):
    """ObsidianVaultStore writes markdown while querying through NetworkX."""

    @property
    def backend_name(self) -> str:
        """Return the backend identifier used by the resolver and demos."""

        return "obsidian"

    def __init__(self, vault_path: Optional[str] = None) -> None:
        """Store vault path and create the NetworkX query layer."""

        self._vault_path = Path(vault_path or os.getenv("OBSIDIAN_VAULT_PATH", "./data/obsidian_vault")).expanduser()
        self._nx_store = NetworkXStore()

    def initialize(self) -> None:
        """Create vault folders and rebuild the in-memory graph from reactions."""

        (self._vault_path / "chemicals").mkdir(parents=True, exist_ok=True)
        (self._vault_path / "reactions").mkdir(parents=True, exist_ok=True)
        (self._vault_path / "papers").mkdir(parents=True, exist_ok=True)
        self._nx_store.initialize()
        self._rebuild_from_vault()
        logger.info("ObsidianVaultStore initialized at %s", self._vault_path)

    def _rebuild_from_vault(self) -> None:
        """Rebuild NetworkX state from existing reaction markdown files."""

        reaction_files = list((self._vault_path / "reactions").glob("*.md"))
        if not reaction_files:
            logger.info("No existing vault files found. Starting fresh.")
            return
        loaded = 0
        failed = 0
        for md_file in reaction_files:
            try:
                recipe = parse_recipe_from_markdown(md_file)
                if recipe:
                    self._nx_store.write_recipe_full(recipe)
                    loaded += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.warning("Failed to load %s: %s", md_file.name, exc)
                failed += 1
        logger.info("Vault rebuild complete: %s loaded, %s failed.", loaded, failed)

    def write_recipe(self, recipe: ChemicalRecipe) -> str:
        """Write a recipe to NetworkX and reactions markdown idempotently."""

        node_id = self._nx_store.write_recipe(recipe)
        safe_recipe_id = sanitize_filename(recipe.recipe_id)
        if safe_recipe_id != recipe.recipe_id:
            raise ValueError(f"Unsafe recipe ID for Obsidian filename: {recipe.recipe_id!r}")
        md_path = self._vault_path / "reactions" / f"{safe_recipe_id}.md"
        md_path.write_text(generate_recipe_markdown(recipe), encoding="utf-8")
        return node_id

    def write_chemical(self, entity: ChemicalEntity, recipe_id: str) -> str:
        """Write a chemical to NetworkX and create/update its note."""

        node_id = self._nx_store.write_chemical(entity, recipe_id)
        safe_name = sanitize_filename(entity.name)
        md_path = self._vault_path / "chemicals" / f"{safe_name}.md"
        if md_path.exists():
            append_recipe_link_to_chemical(md_path, recipe_id, entity.role.value)
        else:
            md_path.write_text(generate_chemical_markdown(entity, recipe_id), encoding="utf-8")
        return node_id

    def write_relationship(self, source_id: str, target_id: str, relationship_type: RelationshipType, properties: dict) -> None:
        """Delegate relationship writes to NetworkX because WikiLinks live in notes."""

        self._nx_store.write_relationship(source_id, target_id, relationship_type, properties)

    def write_paper(self, metadata: PaperMetadata) -> str:
        """Write a paper node and markdown provenance note."""

        node_id = self._nx_store.write_paper(metadata)
        safe_name = sanitize_filename(metadata.doi or metadata.title[:30])
        md_path = self._vault_path / "papers" / f"{safe_name}.md"
        if not md_path.exists():
            md_path.write_text(generate_paper_markdown(metadata), encoding="utf-8")
        return node_id

    def get_recipe(self, recipe_id: str) -> Optional[ChemicalRecipe]:
        """Delegate recipe lookup to the NetworkX query layer."""

        return self._nx_store.get_recipe(recipe_id)

    def get_all_recipes(self) -> list[ChemicalRecipe]:
        """Delegate all recipe retrieval to the NetworkX query layer."""

        return self._nx_store.get_all_recipes()

    def query_solvents_for(self, compound_name: str) -> list[SolventFrequency]:
        """Delegate solvent queries to NetworkX."""

        return self._nx_store.query_solvents_for(compound_name)

    def query_recipes_for_chemical(self, chemical_name: str) -> list[ChemicalRecipe]:
        """Delegate chemical recipe queries to NetworkX."""

        return self._nx_store.query_recipes_for_chemical(chemical_name)

    def query_cooccurring_chemicals(self, chemical_name: str, limit: int = 10) -> list[CoOccurrence]:
        """Delegate co-occurrence queries to NetworkX."""

        return self._nx_store.query_cooccurring_chemicals(chemical_name, limit)

    def export_graph_json(self) -> GraphJSON:
        """Delegate graph export to NetworkX."""

        return self._nx_store.export_graph_json()

    def node_count(self) -> int:
        """Return NetworkX node count."""

        return self._nx_store.node_count()

    def edge_count(self) -> int:
        """Return NetworkX edge count."""

        return self._nx_store.edge_count()

    def recipe_count(self) -> int:
        """Return NetworkX recipe count."""

        return self._nx_store.recipe_count()

    def close(self) -> None:
        """Close the NetworkX query layer while leaving markdown persisted."""

        self._nx_store.close()
        logger.info("ObsidianVaultStore closed. Vault at %s", self._vault_path)

    def clear(self) -> None:
        """Remove all Markdown notes in the vault and reset its query mirror."""

        resolved = self._vault_path.resolve()
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
            raise ValueError(f"Refusing to clear unsafe vault path: {resolved}")
        required_folders = {"chemicals", "reactions", "papers"}
        if resolved.exists() and not all((resolved / folder).is_dir() for folder in required_folders):
            raise ValueError(f"Refusing to clear path without a ChemExtract vault layout: {resolved}")
        if not resolved.exists():
            self.initialize()
            return
        for markdown_path in resolved.rglob("*.md"):
            if markdown_path.is_file():
                markdown_path.unlink()
        self._nx_store.clear()
