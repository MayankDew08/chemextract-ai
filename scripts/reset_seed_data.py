"""CLI for replacing ChemExtract's demo stores with real chemistry data."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.observability.metrics_store import MetricsStore
from src.seed_data.service import reset_and_seed_demo_data
from src.storage.networkx_store import NetworkXStore


def _parser() -> argparse.ArgumentParser:
    """Describe the explicit destructive targets at the command boundary."""

    parser = argparse.ArgumentParser(description="Clear and seed ChemExtract with ten literature-backed recipes.")
    parser.add_argument("--metrics-db", default="./data/metrics.db")
    parser.add_argument("--vault-path", default=None)
    parser.add_argument("--neo4j-if-running", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion of metrics, graph records, and vault Markdown.")
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Create the stores, run the reset, and print machine-readable verification."""

    if not args.yes:
        raise SystemExit("Refusing destructive reset without --yes")
    vault_path = args.vault_path or os.getenv("OBSIDIAN_VAULT_PATH", "./data/obsidian_vault")
    graph_store = NetworkXStore()
    graph_store.initialize()
    metrics_store = MetricsStore(db_path=args.metrics_db)
    await metrics_store.initialize()
    neo4j_settings = None
    if args.neo4j_if_running:
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            raise SystemExit("NEO4J_PASSWORD is required with --neo4j-if-running")
        neo4j_settings = {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": neo4j_password,
            "confirm_reset": "chemextract-managed-nodes",
        }
    try:
        summary = await reset_and_seed_demo_data(
            active_store=graph_store,
            metrics_store=metrics_store,
            vault_path=vault_path,
            neo4j_settings=neo4j_settings,
        )
        dashboard_api = await _verify_dashboard_api(graph_store, metrics_store)
        print(json.dumps({**summary.model_dump(), "dashboard_api": dashboard_api}, indent=2, sort_keys=True))
    finally:
        await metrics_store.close()
        graph_store.close()
    return 0


async def _verify_dashboard_api(graph_store: NetworkXStore, metrics_store: MetricsStore) -> dict[str, int | bool]:
    """Verify the existing dashboard APIs against the freshly seeded stores."""

    app.state.graph_store = graph_store
    app.state.metrics_store = metrics_store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://seed-verification") as client:
        metrics_response = await client.get("/api/metrics")
        recipes_response = await client.get("/api/recipes")
        graph_response = await client.get("/api/graph/data")
        dashboard_response = await client.get("/dashboard")
    metrics_response.raise_for_status()
    recipes_response.raise_for_status()
    graph_response.raise_for_status()
    dashboard_response.raise_for_status()
    metrics = metrics_response.json()
    recipes = recipes_response.json()
    graph = graph_response.json()
    verified = (
        metrics["total_recipes"] == 10
        and len(recipes) == 10
        and bool(graph["nodes"])
        and bool(graph["edges"])
        and "ChemExtract AI" in dashboard_response.text
    )
    if not verified:
        raise RuntimeError("Dashboard API verification failed")
    return {
        "verified": True,
        "recipes": len(recipes),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
    }


def main() -> int:
    """Load project settings and execute the confirmed reset."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
