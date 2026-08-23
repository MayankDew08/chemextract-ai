"""Pytest coverage for the Phase 5 FastAPI boundary.

These tests exercise the public API routes against an in-memory graph store and
temporary metrics database so dashboard behavior is verified without live LLM or
Obsidian side effects.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.observability.metrics_store import MetricsStore
from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus
from src.storage.networkx_store import NetworkXStore
from src.validator.error_types import ErrorSeverity, ErrorType, ResponsibleAgent, ValidationError


def test_recipe_detail_exposes_original_paper_link() -> None:
    """Recipe details must give users a safe link to the stored source paper URL."""

    template = Path("src/api/templates/recipe_detail.html").read_text(encoding="utf-8")
    script = Path("src/api/static/js/recipe_detail.js").read_text(encoding="utf-8")

    assert 'id="paper-url"' in template
    assert "recipe.source_paper_url" in script
    assert "Open original paper" in script
    assert "https:" in script and "http:" in script


def test_recipe_detail_renders_warning_and_routing_audit_text_safely() -> None:
    """Validation content should use textContent and truthful routing terminology."""

    script = Path("src/api/static/js/recipe_detail.js").read_text(encoding="utf-8")

    assert "validation_warnings" in script
    assert "agent_routed_to" in script
    assert "routed to" in script
    assert "fixed by" not in script
    assert "validationSection" in script
    assert "validationSection.innerHTML" not in script
    assert ".textContent" in script


def make_recipes() -> list[ChemicalRecipe]:
    """Create stable recipes that exercise graph and metrics endpoints."""

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
    """Build one validated recipe with realistic observability fields."""

    return ChemicalRecipe(
        recipe_id=recipe_id,
        source_chunk_id=recipe_id,
        source_paper_title=f"Paper for {recipe_id}",
        source_paper_doi=f"10.0000/{recipe_id}",
        source_paper_url=f"https://example.org/papers/{recipe_id}",
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


async def _prepare_app(tmp_path: Path) -> tuple[NetworkXStore, MetricsStore]:
    """Install temporary app state for endpoint testing."""

    store = NetworkXStore()
    store.initialize()
    metrics = MetricsStore(db_path=str(tmp_path / "metrics.db"))
    await metrics.initialize()
    for recipe in make_recipes():
        store.write_recipe_full(recipe)
        await metrics.record_recipe(recipe)
    app.state.graph_store = store
    app.state.metrics_store = metrics
    return store, metrics


@pytest.mark.asyncio
async def test_phase5_api_endpoints_expose_metrics_recipes_and_graph(tmp_path: Path) -> None:
    """The FastAPI API should expose all dashboard data from injected stores."""

    store, metrics = await _prepare_app(tmp_path)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            metrics_response = await client.get("/api/metrics")
            assert metrics_response.status_code == 200
            assert metrics_response.json()["total_recipes"] == 5

            recipes_response = await client.get("/api/recipes")
            assert recipes_response.status_code == 200
            assert len(recipes_response.json()) == 5
            assert all(recipe["warning_count"] == 0 for recipe in recipes_response.json())

            recipe_response = await client.get("/api/recipes/rxn_zno_1")
            assert recipe_response.status_code == 200
            assert recipe_response.json()["recipe_id"] == "rxn_zno_1"
            assert recipe_response.json()["source_paper_url"] == "https://example.org/papers/rxn_zno_1"

            graph_response = await client.get("/api/graph/data")
            assert graph_response.status_code == 200
            graph = graph_response.json()
            assert graph["nodes"]
            assert graph["edges"]
            assert all("id" in node and "label" in node for node in graph["nodes"])
            assert all("from" in edge and "to" in edge for edge in graph["edges"])

            solvents_response = await client.get("/api/graph/query/solvents/ZnO")
            assert solvents_response.status_code == 200
            assert len(solvents_response.json()) == 3

            dashboard_response = await client.get("/dashboard")
            assert dashboard_response.status_code == 200
            assert "ChemExtract AI" in dashboard_response.text
    finally:
        await metrics.close()
        store.close()


@pytest.mark.asyncio
async def test_recipe_api_exposes_warning_details_and_provenance(tmp_path: Path) -> None:
    """List and detail endpoints should expose warning count, details, and provenance."""

    store = NetworkXStore()
    store.initialize()
    recipe = ChemicalRecipe.model_validate(
        make_recipes()[0].model_dump()
        | {
            "validation_warnings": [
                ValidationError(
                    error_type=ErrorType.NO_CONDITIONS,
                    severity=ErrorSeverity.WARNING,
                    responsible_agent=ResponsibleAgent.CONDITION_AGENT,
                    field_path="conditions",
                    message="No conditions found.",
                    suggested_fix="Inspect source text.",
                )
            ],
            "source_text_completeness": "partial",
            "source_acquisition_method": "open_access_pdf",
            "node_extraction_methods": {"condition_agent": "fallback_llm"},
        }
    )
    store.write_recipe_full(recipe)
    app.state.graph_store = store
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            summaries = (await client.get("/api/recipes")).json()
            detail = (await client.get(f"/api/recipes/{recipe.recipe_id}")).json()

        assert summaries[0]["warning_count"] == 1
        assert detail["validation_warnings"][0]["error_type"] == "NO_CONDITIONS"
        assert detail["source_text_completeness"] == "partial"
        assert detail["source_acquisition_method"] == "open_access_pdf"
        assert detail["node_extraction_methods"] == {"condition_agent": "fallback_llm"}
    finally:
        store.close()


@pytest.mark.asyncio
async def test_metrics_store_additively_migrates_and_records_warning_types(tmp_path: Path) -> None:
    """An existing Phase 5 database should gain warning_types without losing compatibility."""

    db_path = tmp_path / "legacy-metrics.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE recipe_runs (
            recipe_id TEXT PRIMARY KEY,
            validation_status TEXT NOT NULL,
            correction_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            latency_seconds REAL DEFAULT 0.0,
            llm_model TEXT DEFAULT '',
            error_types TEXT DEFAULT '[]',
            node_latencies TEXT DEFAULT '{}',
            node_tokens TEXT DEFAULT '{}',
            extracted_at TEXT
        )
        """
    )
    connection.commit()
    connection.close()

    warning = ValidationError(
        error_type=ErrorType.NO_CONDITIONS,
        severity=ErrorSeverity.WARNING,
        responsible_agent=ResponsibleAgent.CONDITION_AGENT,
        field_path="conditions",
        message="No conditions found.",
        suggested_fix="Inspect source text.",
    )
    recipe = ChemicalRecipe.model_validate(make_recipes()[0].model_dump() | {"validation_warnings": [warning]})
    metrics = MetricsStore(db_path=str(db_path))
    await metrics.initialize()
    await metrics.record_recipe(recipe)
    await metrics.close()

    connection = sqlite3.connect(db_path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(recipe_runs)")}
    warning_types = connection.execute(
        "SELECT warning_types FROM recipe_runs WHERE recipe_id = ?", (recipe.recipe_id,)
    ).fetchone()[0]
    connection.close()

    assert "warning_types" in columns
    assert warning_types == '["NO_CONDITIONS"]'
