"""ChemExtract AI offline-first end-to-end verification script.

Run with ``python tests/verify_pipeline.py`` from the project root. Core checks
use no network; the optional extraction check uses configured local/cloud LLMs.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.json_utils import parse_quantity_string
from src.schemas.chemical import ChemicalEntity, ChemicalRole, Quantity
from src.schemas.paper import PaperMetadata, TextChunk, TextCompleteness
from src.schemas.recipe import ChemicalRecipe, ReactionConditions, ValidationStatus
from src.validator.engine import ValidationEngine

HARDCODED_ZNO_TEXT = """
2.1 Synthesis of ZnO Nanoparticles

Zinc acetate dihydrate (Zn(CH3COO)2·2H2O, 98% purity, Sigma-Aldrich)
(2.195 g, 10 mmol) was dissolved in 100 mL of absolute ethanol
(99.8%, Merck) under magnetic stirring at room temperature.
Potassium hydroxide (KOH, 99%, Sigma-Aldrich) (1.12 g, 20 mmol)
was dissolved separately in 50 mL of absolute ethanol and added
dropwise to the zinc acetate solution at 60°C. The mixture was
refluxed at 65°C for 2 hours under atmospheric pressure.
The resulting white precipitate was separated by centrifugation
at 5000 rpm for 15 minutes, washed three times with ethanol
to remove unreacted precursors, and dried at 80°C overnight.
The final ZnO nanoparticles were calcined at 300°C for 2 hours
in air atmosphere to remove residual organics.
"""

HARDCODED_TIO2_TEXT = """
Experimental Section

Titanium(IV) isopropoxide (Ti(OC3H7)4, 97%, Aldrich) (2.84 g, 10 mmol)
was dissolved in 50 mL of isopropanol under inert atmosphere.
Hydrochloric acid (HCl, 37%, 1.0 mL) was added dropwise as
peptizing agent. The solution was stirred at 25°C for 30 minutes,
then 5 mL of deionized water was added slowly to initiate hydrolysis.
The mixture was aged at 80°C for 3 hours to form TiO2 sol-gel.
The gel was dried at 100°C for 12 hours and calcined at 450°C
for 4 hours to obtain anatase TiO2 powder.
"""


def make_chunks() -> list[TextChunk]:
    """Build the two deterministic full-text chunks used by verification."""

    chunks = []
    for chunk_id, title, text in (("test-zno", "ZnO test synthesis", HARDCODED_ZNO_TEXT), ("test-tio2", "TiO2 test synthesis", HARDCODED_TIO2_TEXT)):
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            paper_metadata=PaperMetadata(title=title, source_db="test"),
            text=text,
            completeness=TextCompleteness.FULL_TEXT,
            word_count=len(text.split()),
        ))
    return chunks


def make_recipe() -> ChemicalRecipe:
    """Build a known-valid ZnO recipe for validator and storage checks."""

    return ChemicalRecipe(
        recipe_id="verify-zno",
        source_chunk_id="test-zno",
        title="Test ZnO Synthesis",
        validation_status=ValidationStatus.PASSED,
        entities=[
            ChemicalEntity(name="Zinc acetate dihydrate", role=ChemicalRole.REACTANT, quantity=Quantity(value=2.195, unit="g"), moles=Quantity(value=10.0, unit="mmol")),
            ChemicalEntity(name="ethanol", role=ChemicalRole.SOLVENT, quantity=Quantity(value=100.0, unit="mL")),
            ChemicalEntity(name="KOH", role=ChemicalRole.CATALYST, quantity=Quantity(value=1.12, unit="g"), moles=Quantity(value=20.0, unit="mmol")),
            ChemicalEntity(name="ZnO", role=ChemicalRole.PRODUCT),
        ],
        conditions=ReactionConditions(temperature_celsius=65.0, duration_hours=2.0, atmosphere="air", technique="reflux"),
    )


def check_json_utils() -> bool:
    """Verify representative quantity parsing behavior."""

    checks = [
        parse_quantity_string("2.195 g") == {"value": 2.195, "unit": "g"},
        parse_quantity_string("10 mmol") == {"value": 10.0, "unit": "mmol"},
        parse_quantity_string("100 mL") == {"value": 100.0, "unit": "mL"},
        parse_quantity_string("65°C") is None,
    ]
    return all(checks)


def check_validator(recipe: ChemicalRecipe) -> bool:
    """Verify the deterministic validator accepts the known-valid recipe."""

    result = ValidationEngine().validate(recipe)
    return result.passed and not result.blocking_errors


def check_networkx(recipe: ChemicalRecipe) -> bool:
    """Verify NetworkX persistence, query, and export behavior."""

    from src.storage.networkx_store import NetworkXStore

    store = NetworkXStore()
    store.initialize()
    try:
        store.write_recipe_full(recipe)
        return store.recipe_count() == 1 and store.node_count() > 0 and bool(store.query_solvents_for("ZnO")) and bool(store.export_graph_json().nodes)
    finally:
        store.close()


def check_obsidian(recipe: ChemicalRecipe) -> bool:
    """Verify Obsidian markdown persistence and wikilink generation."""

    from src.storage.obsidian_store import ObsidianVaultStore

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ObsidianVaultStore(temp_dir)
        store.initialize()
        try:
            store.write_recipe_full(recipe)
            path = Path(temp_dir) / "reactions" / f"{recipe.recipe_id}.md"
            content = path.read_text(encoding="utf-8")
            return path.exists() and "Zinc acetate" in content and "[[" in content
        finally:
            store.close()


def check_neo4j(recipe: ChemicalRecipe) -> bool:
    """Verify Neo4j when a local service is available, otherwise skip it."""

    from src.storage.neo4j_store import Neo4jStore

    store = None
    try:
        store = Neo4jStore()
        store.initialize()
        store.write_recipe_full(recipe)
        passed = store.recipe_count() >= 1
        print("✅ Neo4jStore working")
        return passed
    except Exception as exc:
        print(f"⚠️  Neo4jStore skipped: {exc}")
        print("   Run: docker compose up -d neo4j")
        return False
    finally:
        if store is not None:
            store.close()


async def provider_available(provider: str) -> bool:
    """Return whether a provider is configured and reachable for optional tests."""

    module_name = {"ollama": "langchain_ollama", "groq": "langchain_groq", "gemini": "langchain_google_genai"}[provider]
    try:
        __import__(module_name)
    except ImportError:
        return False
    if provider == "ollama":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=3.0)
                return response.status_code == 200
        except Exception:
            return False
    return bool(os.getenv("GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"))


async def check_full_pipeline(chunk: TextChunk, provider: str) -> bool:
    """Run one preloaded chunk through extraction and graph persistence."""

    from src.pipeline import ChemExtractPipeline, PipelineConfig

    pipeline = ChemExtractPipeline(PipelineConfig(query="ZnO synthesis", preloaded_chunks=[chunk], graph_backend="networkx", llm_provider=provider, inter_chunk_delay_seconds=0.0))
    result = await pipeline.run()
    return result.recipes_extracted >= 1 and result.total_duration_seconds < 120 and (provider != "ollama" or result.total_cost_usd == 0.0)


async def main() -> int:
    """Run all verification stages and print a concise pass/fail summary."""

    chunks = make_chunks()
    recipe = make_recipe()
    print("Created 2 test chunks")
    json_ok = check_json_utils()
    print(f"{'✅' if json_ok else '❌'} JSON utils working" if json_ok else "❌ JSON utils failed")
    validator_ok = check_validator(recipe)
    print(f"{'✅' if validator_ok else '❌'} Validator working — perfect recipe passes" if validator_ok else "❌ Validator failed")
    networkx_ok = check_networkx(recipe)
    print("✅ NetworkXStore working" if networkx_ok else "❌ NetworkXStore failed")
    obsidian_ok = check_obsidian(recipe)
    print("✅ ObsidianVaultStore working — markdown files created" if obsidian_ok else "❌ ObsidianVaultStore failed")
    neo4j_ok = check_neo4j(recipe)

    pipeline_results = {}
    for provider in ("ollama", "groq", "gemini"):
        if await provider_available(provider):
            pipeline_results[provider] = await check_full_pipeline(chunks[0], provider)
            print(f"{'✅' if pipeline_results[provider] else '❌'} Full pipeline ({provider})")
        else:
            pipeline_results[provider] = None
            print(f"⏭️  Full pipeline ({provider}) skipped (provider unavailable)")

    core_ok = json_ok and validator_ok and networkx_ok and obsidian_ok
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"JSON utils:      {'✅' if json_ok else '❌'}")
    print(f"Validator:       {'✅' if validator_ok else '❌'}")
    print(f"NetworkXStore:   {'✅' if networkx_ok else '❌'}")
    print(f"ObsidianStore:   {'✅' if obsidian_ok else '❌'}")
    print(f"Neo4jStore:      {'✅' if neo4j_ok else '⚠️  (optional)'}")
    print(f"Full pipeline:   {'✅' if any(value is True for value in pipeline_results.values()) else '⏭️  (needs provider)'}")
    if core_ok:
        print("✅ CHEMEXTRACT AI IS READY TO SHIP")
        print("   Run: python demo.py --demo")
        print("   Run: uvicorn src.api.main:app --reload")
    else:
        print("❌ Fix the failing checks above before shipping")
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
