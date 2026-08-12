"""Read-only verification for the persisted literature-backed demo seed."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.observability.metrics_store import MetricsStore
from src.schemas.chemical import ChemicalRole
from src.seed_data.recipes import load_seed_recipes
from src.storage.obsidian_store import ObsidianVaultStore


async def _verify() -> dict:
    """Reopen durable stores and verify the dashboard-facing read contracts."""

    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./data/obsidian_vault")).expanduser().resolve()
    metrics_path = os.getenv("METRICS_DB_PATH", "./data/metrics.db")
    graph_store = ObsidianVaultStore(str(vault_path))
    graph_store.initialize()
    metrics_store = MetricsStore(metrics_path)
    await metrics_store.initialize()
    try:
        expected = {recipe.recipe_id: recipe for recipe in load_seed_recipes()}
        actual = {recipe.recipe_id: recipe for recipe in graph_store.get_all_recipes()}
        if actual != expected:
            raise RuntimeError("Persisted Obsidian recipes do not match the seed corpus")

        app.state.graph_store = graph_store
        app.state.metrics_store = metrics_store
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://seed-verification") as client:
            metrics_response = await client.get("/api/metrics")
            recipes_response = await client.get("/api/recipes")
            graph_response = await client.get("/api/graph/data")
            dashboard_response = await client.get("/dashboard")
        for response in (metrics_response, recipes_response, graph_response, dashboard_response):
            response.raise_for_status()

        metrics = metrics_response.json()
        recipes = recipes_response.json()
        graph = graph_response.json()
        product_counts = Counter(
            entity.formula or entity.name
            for recipe in actual.values()
            for entity in recipe.entities
            if entity.role == ChemicalRole.PRODUCT
        )
        verified = (
            metrics["total_recipes"] == 10
            and len(recipes) == 10
            and product_counts == {"ZnO": 5, "Fe3O4": 3, "TiO2": 2}
            and bool(graph["nodes"])
            and bool(graph["edges"])
            and "ChemExtract AI" in dashboard_response.text
        )
        if not verified:
            raise RuntimeError("Persisted dashboard verification failed")
        return {
            "verified": True,
            "recipes": len(recipes),
            "products": dict(product_counts),
            "metrics": {
                "tokens": metrics["total_tokens_used"],
                "cost_usd": metrics["total_cost_usd"],
                "avg_latency_seconds": metrics["avg_latency_seconds"],
                "success_rate_percent": metrics["success_rate_percent"],
            },
            "graph": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])},
            "vault": {
                "reactions": len(list((vault_path / "reactions").glob("*.md"))),
                "chemicals": len(list((vault_path / "chemicals").glob("*.md"))),
                "papers": len(list((vault_path / "papers").glob("*.md"))),
            },
        }
    finally:
        await metrics_store.close()
        graph_store.close()


def main() -> int:
    """Load environment configuration and print verification JSON."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    print(json.dumps(asyncio.run(_verify()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
