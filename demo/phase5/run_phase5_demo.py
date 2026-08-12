"""Standalone Phase 5 FastAPI dashboard demonstration.

This demo preloads temporary in-memory stores, calls every public API endpoint
through ASGITransport, verifies HTML shells load, and prints a terminal report.
It lives under demo/, while pytest coverage lives under tests/.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

load_dotenv(PROJECT_ROOT / ".env")

from src.api.main import app
from src.observability.metrics_store import MetricsStore
from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus
from src.storage.networkx_store import NetworkXStore


def print_header(text: str, width: int = 72) -> None:
    """Print a high-visibility demo section heading."""

    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width)


def print_check(passed: bool, message: str) -> None:
    """Print one completion checklist item."""

    print(f"{'✅' if passed else '❌'} {message}")


def recipes() -> list[ChemicalRecipe]:
    """Create the five required Phase 4 recipes for Phase 5 API testing."""

    return [
        _recipe("rxn_zno_1", "ZnO", "zinc acetate", "methanol", "KOH", "reflux", 65, 2),
        _recipe("rxn_zno_2", "ZnO", "zinc nitrate", "ethanol", "NaOH", "hydrothermal", 120, 12),
        _recipe("rxn_zno_3", "ZnO", "zinc chloride", "water", "ammonia", "stirring", 25, 4),
        _recipe("rxn_tio2_1", "TiO2", "titanium isopropoxide", "ethanol", "HCl", "sol-gel", 80, 3),
        _recipe("rxn_tio2_2", "TiO2", "TiCl4", "water", "HNO3", "hydrothermal", 150, 8),
    ]


def _recipe(
    recipe_id: str,
    product: str,
    reactant: str,
    solvent: str,
    catalyst: str,
    technique: str,
    temperature: float,
    duration: float,
) -> ChemicalRecipe:
    """Build one validated recipe with dashboard-friendly metrics."""

    return ChemicalRecipe(
        recipe_id=recipe_id,
        source_chunk_id=recipe_id,
        source_paper_title=f"Paper for {recipe_id}",
        source_paper_doi=f"10.0000/{recipe_id}",
        title=f"{technique.title()} synthesis of {product}",
        validation_status=ValidationStatus.PASSED,
        entities=[
            ChemicalEntity(name=reactant, role=ChemicalRole.REACTANT, quantity=Quantity(value=1.0, unit="g")),
            ChemicalEntity(name=solvent, role=ChemicalRole.SOLVENT, quantity=Quantity(value=100.0, unit="mL")),
            ChemicalEntity(name=catalyst, role=ChemicalRole.CATALYST, quantity=Quantity(value=1.0, unit="g")),
            ChemicalEntity(name=product, role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=temperature, duration_hours=duration, technique=technique),
        total_tokens_used=1048,
        estimated_cost_usd=0.0008,
        total_latency_seconds=2.9,
        node_latencies={"entity_node": 1.1, "quantity_node": 0.9, "condition_node": 0.8},
        node_tokens={"entity_node": 420, "quantity_node": 360, "condition_node": 268},
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
    )


async def prepare_app_state(db_path: Path) -> tuple[NetworkXStore, MetricsStore]:
    """Preload temporary stores so the API can be tested without external calls."""

    store = NetworkXStore()
    store.initialize()
    metrics = MetricsStore(db_path=str(db_path))
    await metrics.initialize()
    for recipe in recipes():
        store.write_recipe_full(recipe)
        await metrics.record_recipe(recipe)
    app.state.graph_store = store
    app.state.metrics_store = metrics
    return store, metrics


async def run_demo() -> None:
    """Execute every Phase 5 demo check and print the final report."""

    print_header("ChemExtract AI Phase 5 Demo")
    checks: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        store, metrics = await prepare_app_state(Path(tmpdir) / "metrics.db")
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                health = await client.get("/health")
                checks.append((health.status_code == 200 and health.json().get("status") == "ok", "Health endpoint responds"))

                metrics_response = await client.get("/api/metrics")
                metrics_json = metrics_response.json()
                checks.append((metrics_response.status_code == 200 and metrics_json.get("total_recipes") == 5, "Metrics endpoint returns valid PipelineMetrics"))
                print(f"Metrics: recipes={metrics_json.get('total_recipes')} success={metrics_json.get('success_rate_percent')}% tokens={metrics_json.get('total_tokens_used')}")

                errors = await client.get("/api/metrics/errors")
                checks.append((errors.status_code == 200 and isinstance(errors.json(), list), "Error frequency endpoint works"))

                recipe_list = await client.get("/api/recipes")
                checks.append((recipe_list.status_code == 200 and len(recipe_list.json()) == 5, "Recipes list returns all 5 recipes"))
                print(f"First recipe summary: {recipe_list.json()[0]}")

                recipe = await client.get("/api/recipes/rxn_zno_1")
                checks.append((recipe.status_code == 200 and recipe.json().get("recipe_id") == "rxn_zno_1" and "entities" in recipe.json() and "conditions" in recipe.json(), "Single recipe endpoint works"))

                missing = await client.get("/api/recipes/nonexistent")
                checks.append((missing.status_code == 404, "404 returned for missing recipe"))

                graph_response = await client.get("/api/graph/data")
                graph_json = graph_response.json()
                graph_has_data = graph_response.status_code == 200 and graph_json.get("nodes") and graph_json.get("edges")
                graph_visjs = graph_has_data and all("id" in node and "label" in node for node in graph_json["nodes"]) and all("from" in edge and "to" in edge for edge in graph_json["edges"])
                checks.append((bool(graph_has_data), "Graph data has nodes and edges"))
                checks.append((bool(graph_visjs), "Graph data valid for vis.js (has id, label, from, to)"))
                print(f"Graph: {len(graph_json.get('nodes', []))} nodes, {len(graph_json.get('edges', []))} edges")

                solvents = await client.get("/api/graph/query/solvents/ZnO")
                checks.append((solvents.status_code == 200 and len(solvents.json()) == 3, "Solvent query returns correct results"))
                print(f"Solvents for ZnO: {solvents.json()}")

                cooccurrence = await client.get("/api/graph/query/cooccurrence/ZnO")
                checks.append((cooccurrence.status_code == 200 and len(cooccurrence.json()) > 0, "Co-occurrence query works"))

                dashboard = await client.get("/dashboard")
                recipes_page = await client.get("/recipes")
                graph_page = await client.get("/graph")
                html_ok = dashboard.status_code == 200 and recipes_page.status_code == 200 and graph_page.status_code == 200
                checks.append((html_ok, "All three HTML pages load without errors"))
                checks.append(("vis-network" in graph_page.text, "vis.js script referenced in graph page"))

                root = await client.get("/", follow_redirects=False)
                print(f"Root response status: {root.status_code}")
        finally:
            await metrics.close()
            store.close()

    print_header("Completion Checklist")
    for passed, message in checks:
        print_check(passed, message)
    if all(passed for passed, _message in checks):
        print("\n✅ PHASE 5 COMPLETE — CHEMEXTRACT AI IS READY FOR DEMO")
    else:
        print("\n❌ PHASE 5 INCOMPLETE")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_demo())
