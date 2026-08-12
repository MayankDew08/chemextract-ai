#!/usr/bin/env python3
"""ChemExtract AI — One-Command Demo.

Usage:
    python demo.py --query "synthesis of zinc oxide nanoparticles"
    python demo.py --query "LiFePO4 cathode preparation" --max-papers 5
    python demo.py --query "TiO2 nanoparticles" --backend obsidian
    python demo.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from src.schemas.paper import AcquisitionMethod, PaperMetadata, TextChunk, TextCompleteness


def _sample_chunk(chunk_id: str, title: str, text: str) -> TextChunk:
    """Build a real TextChunk object for demo mode."""

    return TextChunk(
        chunk_id=chunk_id,
        paper_metadata=PaperMetadata(title=title, doi=f"10.0000/{chunk_id}", source_db="demo"),
        text=text,
        section_name="Experimental",
        completeness=TextCompleteness.FULL_TEXT,
        acquisition_method=AcquisitionMethod.USER_UPLOAD,
        chemical_density_score=0.9,
        word_count=len(text.split()),
    )


SAMPLE_CHUNKS = [
    _sample_chunk(
        "demo_zno_reflux",
        "Hydrothermal Synthesis of ZnO Nanoparticles",
        (
            "Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in 100 mL of methanol. "
            "The solution was refluxed at 65°C for 2 hours. KOH (1.12 g, 20 mmol) was added "
            "as base catalyst. The resulting ZnO precipitate was filtered."
        ),
    ),
    _sample_chunk(
        "demo_zno_water",
        "Aqueous Synthesis of Zinc Oxide",
        (
            "Zinc acetate (1.10 g, 5 mmol) was dissolved in 80 mL of water. Potassium hydroxide "
            "(0.56 g, 10 mmol) was added slowly while stirring. The mixture was heated at 90°C "
            "for 3 hours and ZnO nanoparticles were collected by filtration."
        ),
    ),
    _sample_chunk(
        "demo_generic",
        "Generic Synthesis Paper",
        (
            "The compound was prepared using standard laboratory methods. Some chemicals were "
            "mixed and heated. The resulting material was isolated after routine processing."
        ),
    ),
]


async def main() -> None:
    """Parse CLI arguments, run the pipeline, and print demo output."""

    parser = argparse.ArgumentParser(
        description="ChemExtract AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", type=str, help="Chemistry query to process")
    parser.add_argument("--max-papers", "-n", type=int, default=8, help="Maximum papers to fetch")
    parser.add_argument("--backend", "-b", choices=["networkx", "obsidian", "neo4j"], default=None)
    parser.add_argument("--demo", action="store_true", help="Run with preloaded sample data. No API keys needed.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if not args.query and not args.demo:
        parser.print_help()
        print("\nError: Provide --query or --demo")
        sys.exit(1)

    from src.pipeline import ChemExtractPipeline, PipelineConfig

    if args.demo:
        print("\n⚗️  Running in DEMO MODE (preloaded data, no API needed)\n")
        config = PipelineConfig(
            query="synthesis of zinc oxide nanoparticles",
            preloaded_chunks=SAMPLE_CHUNKS,
            graph_backend=args.backend or "networkx",
            max_retries=1,
            verbose=True,
        )
    else:
        config = PipelineConfig(
            query=args.query,
            max_papers=args.max_papers,
            graph_backend=args.backend,
            verbose=True,
        )

    print("⚗️  ChemExtract AI")
    print(f"   Query: {config.query}")
    print(f"   Backend: {config.graph_backend or os.getenv('GRAPH_BACKEND', 'networkx')}")
    print(f"   LLM: {os.getenv('LLM_PROVIDER', 'groq')}")

    pipeline = ChemExtractPipeline(config)
    async for event in pipeline.stream():
        print(event.to_terminal_line())

    result = pipeline._result
    _print_recipe_table(result)
    _print_stats(result)
    _write_output(args.output, result)
    print()


def _print_recipe_table(result) -> None:
    """Print a compact table of extracted recipes."""

    if not result.recipes:
        return
    print(f"\n{'─' * 70}")
    print("  EXTRACTED RECIPES")
    print(f"{'─' * 70}")
    print(f"  {'ID':<12} {'STATUS':<12} {'PRODUCT':<20} {'TEMP':<8} {'TIME':<8} {'CORR'}")
    print(f"  {'─' * 10} {'─' * 10} {'─' * 18} {'─' * 6} {'─' * 6} {'─' * 4}")
    for recipe in result.recipes:
        products = [entity.name for entity in recipe.entities if entity.role.value == "PRODUCT"]
        product = products[0][:18] if products else "unknown"
        temp = (
            f"{recipe.conditions.temperature_celsius:.0f}°C"
            if recipe.conditions and recipe.conditions.temperature_celsius
            else "N/A"
        )
        duration = (
            f"{recipe.conditions.duration_hours:.1f}h"
            if recipe.conditions and recipe.conditions.duration_hours
            else "N/A"
        )
        status_icon = {"PASSED": "✅ PASS", "CORRECTED": "⚠️  CORR", "FAILED": "❌ FAIL"}.get(
            recipe.validation_status.value,
            "?",
        )
        print(
            f"  {recipe.recipe_id:<12} {status_icon:<12} "
            f"{product:<20} {temp:<8} {duration:<8} {recipe.correction_count()}"
        )
    print(f"{'─' * 70}")


def _print_stats(result) -> None:
    """Print final pipeline statistics."""

    print("\n  PIPELINE STATISTICS")
    print(f"  {'─' * 40}")
    print(f"  Papers fetched      : {result.papers_fetched}")
    print(f"  Chunks processed    : {result.chunks_produced}")
    print(f"  Recipes extracted   : {result.recipes_extracted}")
    print(f"  Success rate        : {result.success_rate():.1f}%")
    print(f"  Self-corrected      : {result.self_corrected}")
    print(f"  Failed              : {result.failed}")
    print(f"  Graph nodes added   : {result.nodes_added}")
    print(f"  Graph edges added   : {result.edges_added}")
    print(f"  Total tokens used   : {result.total_tokens_used:,}")
    print(f"  Estimated cost      : ${result.total_cost_usd:.6f}")
    print(f"  Total duration      : {result.total_duration_seconds:.1f}s")
    print(f"  LLM used            : {result.llm_model}")


def _write_output(output_path: str | None, result) -> None:
    """Write a JSON summary when requested."""

    if not output_path:
        return
    output_data = result.model_dump(mode="json", exclude={"recipes"})
    output_data["recipe_summaries"] = [
        {
            "recipe_id": recipe.recipe_id,
            "validation_status": recipe.validation_status.value,
            "entities": len(recipe.entities),
            "corrections": recipe.correction_count(),
        }
        for recipe in result.recipes
    ]
    Path(output_path).write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
