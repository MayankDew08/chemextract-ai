"""Standalone Phase 2 demonstration runner.

This script runs one representative TextChunk through the LangGraph extraction
pipeline and prints the resulting ChemicalRecipe. It intentionally lives under
demo, not tests, because pytest files are the automated verification surface.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.pipeline import ExtractionPipeline
from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness


def print_header(text: str, width: int = 70) -> None:
    """Print a centered heading for the terminal report."""

    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width)


def print_section(text: str) -> None:
    """Print a section heading for each demo phase."""

    print("\n" + "─" * 70)
    print(text)
    print("─" * 70)


def print_check(passed: bool, message: str) -> None:
    """Print a consistent pass/fail checklist line."""

    print(f"{'✅' if passed else '❌'} {message}")


def build_demo_chunk() -> TextChunk:
    """Create a realistic Phase 1 TextChunk for deterministic Phase 2 testing."""

    text = (
        "Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in "
        "100 mL of methanol. The solution was refluxed at 65°C for "
        "2 hours with 1.12 g of KOH (20 mmol) as base catalyst. "
        "The ZnO precipitate was filtered and dried at 120°C."
    )
    metadata = PaperMetadata(
        title="Demo synthesis of zinc oxide nanoparticles",
        doi="10.0000/chemextract-demo",
        source_db="demo",
        abstract=text,
        open_access=True,
    )
    return TextChunk(
        chunk_id="phase2-demo-zno",
        paper_metadata=metadata,
        text=text,
        section_name="experimental",
        completeness=TextCompleteness.FULL_TEXT,
        acquisition_method=AcquisitionMethod.USER_UPLOAD,
        chemical_density_score=1.0,
        word_count=len(text.split()),
    )


async def main() -> None:
    """Run the Phase 2 extraction pipeline and print validation checks."""

    print_header("ChemExtract AI Phase 2 Demo")
    chunk = build_demo_chunk()

    print_section("STEP 1: Input TextChunk")
    print(chunk.text)

    print_section("STEP 2: LangGraph Extraction")
    pipeline = ExtractionPipeline()
    recipe = await pipeline.extract(chunk)
    print(f"Recipe ID: {recipe.recipe_id}")
    print(f"Provider: {recipe.llm_provider}")
    print(f"Model: {recipe.llm_model}")
    print(f"Tokens: {recipe.total_tokens_used}")
    print(f"Latency: {recipe.total_latency_seconds:.2f}s")
    print(f"Estimated cost: ${recipe.estimated_cost_usd:.6f}")

    print_section("STEP 3: Extracted Entities")
    for entity in recipe.entities:
        quantity = str(entity.quantity) if entity.quantity else "None"
        moles = str(entity.moles) if entity.moles else "None"
        print(f"- {entity.name:<28} role={entity.role.value:<8} quantity={quantity:<12} moles={moles}")

    print_section("STEP 4: Extracted Conditions")
    conditions = recipe.conditions
    if conditions:
        print(f"Temperature: {conditions.temperature_celsius}")
        print(f"Duration: {conditions.duration_hours}")
        print(f"Pressure: {conditions.pressure_atm}")
        print(f"Atmosphere: {conditions.atmosphere}")
        print(f"Technique: {conditions.technique}")
        print(f"Yield: {conditions.yield_percent}")

    print_section("STEP 5: Pass/Fail Checklist")
    names = {entity.name.lower() for entity in recipe.entities}
    roles = {entity.role.value for entity in recipe.entities}
    checks = [
        (len(recipe.entities) >= 4, "At least 4 entities extracted"),
        ({"REACTANT", "SOLVENT", "CATALYST", "PRODUCT"}.issubset(roles), "Core chemical roles extracted"),
        (any("zinc acetate" in name for name in names), "Zinc acetate reactant found"),
        (any("methanol" in name for name in names), "Methanol solvent found"),
        (any("koh" in name for name in names), "KOH catalyst found"),
        (conditions is not None and conditions.temperature_celsius == 65.0, "Temperature normalized to Celsius"),
        (conditions is not None and conditions.duration_hours == 2.0, "Duration normalized to hours"),
        (conditions is not None and conditions.technique == "reflux", "Technique extracted"),
        (recipe.total_tokens_used > 0, "Observability tokens recorded"),
    ]
    for passed, message in checks:
        print_check(passed, message)

    failed = [message for passed, message in checks if not passed]
    if failed:
        print("\n❌ PHASE 2 DEMO INCOMPLETE")
        sys.exit(1)
    print("\n✅ PHASE 2 COMPLETE — LANGGRAPH EXTRACTION PASSED")


if __name__ == "__main__":
    asyncio.run(main())
