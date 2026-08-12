"""Behavioral coverage for the deterministic real-chemistry demo seed.

The seed suite exercises only public boundaries: the recipe loader, the reset
service, and the existing FastAPI read APIs.  Neo4j remains optional so these
tests never require a live external database.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.observability.metrics_store import MetricsStore
from src.schemas.chemical import ChemicalRole
from src.seed_data.recipes import load_seed_recipes
from src.seed_data.service import reset_and_seed_demo_data
from src.storage.networkx_store import NetworkXStore
from src.storage.obsidian_store import ObsidianVaultStore


def _summary_value(summary: object, field: str) -> object:
    """Read a documented summary field from either a model or mapping."""

    if isinstance(summary, Mapping):
        return summary[field]
    return getattr(summary, field)


def _product_formula(recipe: object) -> str:
    """Return the single product formula used to classify a seed recipe."""

    products = [entity for entity in recipe.entities if entity.role == ChemicalRole.PRODUCT]
    assert len(products) == 1, f"{recipe.recipe_id} must have exactly one product"
    return products[0].formula or products[0].name


def test_recipe_loader_returns_ten_source_backed_realistic_recipes() -> None:
    """The deterministic corpus contains the promised chemistry and run data."""

    recipes = load_seed_recipes()

    assert len(recipes) == 10
    assert len({recipe.recipe_id for recipe in recipes}) == 10
    assert Counter(_product_formula(recipe) for recipe in recipes) == {
        "ZnO": 5,
        "Fe3O4": 3,
        "TiO2": 2,
    }

    for recipe in recipes:
        assert recipe.source_paper_title
        assert recipe.source_paper_doi and recipe.source_paper_doi.startswith("10.")
        assert recipe.source_paper_url and recipe.source_paper_url.startswith("https://")
        assert "example.org" not in recipe.source_paper_url
        assert recipe.conditions is not None
        assert recipe.conditions.temperature_celsius is not None
        assert recipe.conditions.duration_hours is not None
        assert recipe.conditions.technique
        assert any(
            entity.quantity is not None
            for entity in recipe.entities
            if entity.role != ChemicalRole.PRODUCT
        )
        assert 100 <= recipe.total_tokens_used <= 50_000
        assert 0 < recipe.estimated_cost_usd < 1
        assert 0 < recipe.total_latency_seconds < 600
        assert recipe.node_latencies
        assert recipe.node_tokens

    corrected = [recipe for recipe in recipes if recipe.was_corrected()]
    assert corrected, "the demo corpus must include realistic self-corrections"
    assert all(recipe.validation_status.value == "CORRECTED" for recipe in corrected)
    assert all(recipe.correction_count() == len(recipe.correction_history) for recipe in recipes)


@pytest.mark.asyncio
async def test_reset_and_seed_replaces_metrics_graph_and_vault(tmp_path: Path) -> None:
    """One reset replaces stale local data and reports the complete seeded state."""

    graph_store = NetworkXStore()
    graph_store.initialize()
    metrics_store = MetricsStore(db_path=str(tmp_path / "metrics.db"))
    await metrics_store.initialize()
    vault_path = tmp_path / "vault"
    stale_reaction = vault_path / "reactions" / "stale-recipe.md"
    stale_root_note = vault_path / "stale-root-note.md"
    stale_reaction.parent.mkdir(parents=True)
    (vault_path / "chemicals").mkdir()
    (vault_path / "papers").mkdir()
    stale_reaction.write_text("stale reaction", encoding="utf-8")
    stale_root_note.write_text("stale note", encoding="utf-8")

    stale_recipe = load_seed_recipes()[0].model_copy(
        deep=True,
        update={"recipe_id": "stale_recipe", "source_chunk_id": "stale_recipe"},
    )
    graph_store.write_recipe_full(stale_recipe)
    await metrics_store.record_recipe(stale_recipe)

    try:
        summary = await reset_and_seed_demo_data(
            active_store=graph_store,
            metrics_store=metrics_store,
            vault_path=vault_path,
            neo4j_settings=None,
        )
        metrics = await metrics_store.compute_metrics(graph_store)
        graph = graph_store.export_graph_json()

        assert graph_store.get_recipe("stale_recipe") is None
        assert graph_store.recipe_count() == 10
        assert metrics.total_recipes == 10
        assert metrics.total_tokens_used > 0
        assert metrics.total_cost_usd > 0
        assert metrics.avg_latency_seconds > 0
        assert graph.nodes
        assert graph.edges
        assert graph.metadata["recipe_count"] == 10

        assert not stale_reaction.exists()
        assert not stale_root_note.exists()
        assert len(list((vault_path / "reactions").glob("*.md"))) == 10
        assert len(list((vault_path / "papers").glob("*.md"))) == 10
        assert len(list((vault_path / "chemicals").glob("*.md"))) >= 3

        reopened_vault = ObsidianVaultStore(str(vault_path))
        reopened_vault.initialize()
        try:
            original = next(recipe for recipe in load_seed_recipes() if recipe.was_corrected())
            rebuilt = reopened_vault.get_recipe(original.recipe_id)
            assert rebuilt == original
        finally:
            reopened_vault.close()

        assert _summary_value(summary, "recipes_seeded") == 10
        assert _summary_value(summary, "graph_node_count") == graph_store.node_count()
        assert _summary_value(summary, "graph_edge_count") == graph_store.edge_count()
        assert _summary_value(summary, "vault_reaction_count") == 10
        assert _summary_value(summary, "neo4j_status") == "skipped"

        repeated = await reset_and_seed_demo_data(
            active_store=graph_store,
            metrics_store=metrics_store,
            vault_path=vault_path,
            neo4j_settings=None,
        )
        repeated_metrics = await metrics_store.compute_metrics(graph_store)
        assert _summary_value(repeated, "recipes_seeded") == 10
        assert graph_store.recipe_count() == 10
        assert repeated_metrics.total_recipes == 10
        assert len(list((vault_path / "reactions").glob("*.md"))) == 10
    finally:
        await metrics_store.close()
        graph_store.close()


@pytest.mark.asyncio
async def test_seeded_stores_are_visible_through_existing_api_endpoints(tmp_path: Path) -> None:
    """Dashboard APIs expose the same ten recipes, metrics, nodes, and edges."""

    graph_store = NetworkXStore()
    graph_store.initialize()
    metrics_store = MetricsStore(db_path=str(tmp_path / "metrics.db"))
    await metrics_store.initialize()

    try:
        await reset_and_seed_demo_data(
            active_store=graph_store,
            metrics_store=metrics_store,
            vault_path=tmp_path / "vault",
            neo4j_settings=None,
        )
        app.state.graph_store = graph_store
        app.state.metrics_store = metrics_store

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            metrics_response = await client.get("/api/metrics")
            recipes_response = await client.get("/api/recipes")
            graph_response = await client.get("/api/graph/data")
            stats_response = await client.get("/api/graph/stats")

        assert metrics_response.status_code == 200
        assert metrics_response.json()["total_recipes"] == 10
        assert metrics_response.json()["total_tokens_used"] > 0
        assert metrics_response.json()["total_cost_usd"] > 0
        assert metrics_response.json()["avg_latency_seconds"] > 0

        assert recipes_response.status_code == 200
        recipes = recipes_response.json()
        assert len(recipes) == 10
        assert Counter(product for recipe in recipes for product in recipe["product_names"]) == {
            "ZnO": 5,
            "Fe3O4": 3,
            "TiO2": 2,
        }

        assert graph_response.status_code == 200
        assert graph_response.json()["nodes"]
        assert graph_response.json()["edges"]
        assert stats_response.status_code == 200
        assert stats_response.json()["recipe_count"] == 10
        assert stats_response.json()["node_count"] > 0
        assert stats_response.json()["edge_count"] > 0
    finally:
        app.state.graph_store = None
        app.state.metrics_store = None
        await metrics_store.close()
        graph_store.close()
