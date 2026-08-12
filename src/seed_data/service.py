"""Reset and seed orchestration for the local chemistry demonstration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path

from pydantic import BaseModel

from src.observability.metrics_store import MetricsStore
from src.seed_data.recipes import load_seed_recipes
from src.storage.base import BaseGraphStore
from src.storage.neo4j_store import Neo4jStore
from src.storage.obsidian_store import ObsidianVaultStore

logger = logging.getLogger(__name__)


class SeedSummary(BaseModel):
    """Counts verified immediately after the reset and seed operation."""

    recipes_seeded: int
    graph_node_count: int
    graph_edge_count: int
    vault_reaction_count: int
    vault_chemical_count: int
    vault_paper_count: int
    total_tokens_used: int
    total_cost_usd: float
    avg_latency_seconds: float
    neo4j_status: str
    neo4j_node_count: int | None = None
    neo4j_edge_count: int | None = None


async def reset_and_seed_demo_data(
    *,
    active_store: BaseGraphStore,
    metrics_store: MetricsStore,
    vault_path: str | Path,
    neo4j_settings: Mapping[str, str] | None = None,
) -> SeedSummary:
    """Replace local persisted data with the deterministic literature corpus."""

    recipes = load_seed_recipes()
    resolved_vault = _validate_vault_path(vault_path)
    logger.info("Resetting metrics and active %s graph", active_store.backend_name)
    await metrics_store.clear()
    active_store.clear()
    for recipe in recipes:
        active_store.write_recipe_full(recipe)
        await metrics_store.record_recipe(recipe)

    logger.info("Resetting Obsidian Markdown at %s", resolved_vault)
    vault_store = ObsidianVaultStore(str(resolved_vault))
    vault_store.initialize()
    try:
        vault_store.clear()
        for recipe in recipes:
            vault_store.write_recipe_full(recipe)
    finally:
        vault_store.close()

    vault_verifier = ObsidianVaultStore(str(resolved_vault))
    vault_verifier.initialize()
    try:
        if vault_verifier.recipe_count() != len(recipes):
            raise RuntimeError("Seed verification failed: Obsidian did not rebuild all recipes")
        for expected in recipes:
            if vault_verifier.get_recipe(expected.recipe_id) != expected:
                raise RuntimeError(f"Seed verification failed: Obsidian recipe mismatch for {expected.recipe_id}")
    finally:
        vault_verifier.close()

    neo4j_status = "skipped"
    neo4j_node_count = None
    neo4j_edge_count = None
    if neo4j_settings is not None:
        logger.info("Attempting optional Neo4j reset and seed")
        neo4j_status, neo4j_node_count, neo4j_edge_count = _reset_neo4j(recipes, neo4j_settings)

    logger.info("Verifying seeded graph, metrics, and vault")
    metrics = await metrics_store.compute_metrics(active_store)
    reaction_count = len(list((resolved_vault / "reactions").glob("*.md")))
    if active_store.recipe_count() != len(recipes) or metrics.total_recipes != len(recipes) or reaction_count != len(recipes):
        raise RuntimeError("Seed verification failed: graph, metrics, and vault recipe counts must all equal 10")
    return SeedSummary(
        recipes_seeded=len(recipes),
        graph_node_count=active_store.node_count(),
        graph_edge_count=active_store.edge_count(),
        vault_reaction_count=reaction_count,
        vault_chemical_count=len(list((resolved_vault / "chemicals").glob("*.md"))),
        vault_paper_count=len(list((resolved_vault / "papers").glob("*.md"))),
        total_tokens_used=metrics.total_tokens_used,
        total_cost_usd=metrics.total_cost_usd,
        avg_latency_seconds=metrics.avg_latency_seconds,
        neo4j_status=neo4j_status,
        neo4j_node_count=neo4j_node_count,
        neo4j_edge_count=neo4j_edge_count,
    )


def _validate_vault_path(vault_path: str | Path) -> Path:
    """Resolve a vault target and reject broad destructive paths."""

    resolved = Path(vault_path).expanduser().resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"Refusing to reset unsafe vault path: {resolved}")
    required_folders = {"chemicals", "reactions", "papers"}
    if resolved.exists() and not all((resolved / folder).is_dir() for folder in required_folders):
        raise ValueError(f"Refusing to reset path without a ChemExtract vault layout: {resolved}")
    return resolved


def _reset_neo4j(
    recipes: list,
    settings: Mapping[str, str],
) -> tuple[str, int | None, int | None]:
    """Reset and seed Neo4j when reachable, returning unavailable otherwise."""

    store = Neo4jStore(
        uri=settings.get("uri"),
        username=settings.get("username"),
        password=settings.get("password"),
    )
    if settings.get("confirm_reset") != "chemextract-managed-nodes":
        raise ValueError("Neo4j reset requires confirm_reset='chemextract-managed-nodes'")
    try:
        store.initialize()
    except Exception:
        store.close()
        return "unavailable", None, None
    try:
        store.clear()
        for recipe in recipes:
            store.write_recipe_full(recipe)
        return "seeded", store.node_count(), store.edge_count()
    finally:
        store.close()
