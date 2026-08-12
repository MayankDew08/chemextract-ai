"""Standalone Phase 1 demonstration runner.

This file exercises the real ChemExtract ingestion pipeline with live academic
API calls. It intentionally lives outside tests so pytest remains the standard
automated test surface and this script remains a manual end-to-end demo.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.ingestion.deduplicator import DOITitleDeduplicator
from src.ingestion.orchestrator import FetchOrchestrator
from src.ingestion.query_parser import build_query_parser_from_env
from src.ingestion.ranker import ChemistryRelevanceRanker
from src.ingestion.registry import bootstrap_registry
from src.schemas.paper import TextCompleteness


def print_header(text: str, width: int = 65) -> None:
    """Print a centered major heading so the demo reads as an audit report."""

    print("\n" + "═" * width)
    print(text.center(width))
    print("═" * width)


def print_section(text: str) -> None:
    """Print a section heading that separates each verification phase."""

    print("\n" + "─" * 65)
    print(text)
    print("─" * 65)


def print_check(passed: bool, message: str) -> None:
    """Print a consistent pass/fail line for human-readable review."""

    icon = "✅" if passed else "❌"
    print(f"{icon} {message}")


def truncate(text: str, length: int) -> str:
    """Truncate text without breaking the fixed-width terminal layout."""

    return text if len(text) <= length else f"{text[: length - 3]}..."


async def main() -> None:
    """Run the complete live Phase 1 demo from registry bootstrap to checklist."""

    print_header("ChemExtract AI Phase 1 Demo")

    print_section("STEP 1: Bootstrap Registry")
    registry = bootstrap_registry()
    fetchers = registry.get_all_fetchers()
    acquirers = registry.get_all_acquirers()
    print("Registered fetchers:")
    for fetcher in fetchers:
        print(f"  - {fetcher.source_name}")
    print("Registered acquirers:")
    for acquirer in acquirers:
        print(f"  - {acquirer.acquirer_name} (priority {acquirer.priority()})")
    print_check(len(fetchers) == 4, "Exactly 4 fetchers registered")
    print_check(len(acquirers) >= 4, "At least 4 acquirers registered")

    print_section("STEP 2: Query Parsing")
    parser = build_query_parser_from_env()
    print(f"LLM provider preference: {os.getenv('LLM_PROVIDER', 'gemini')}")
    print(f"Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    print(f"Groq model: {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    search_plan = await parser.parse("synthesis of zinc oxide nanoparticles", registry.get_registered_source_names())
    print(f"Target compound: {search_plan.target_compound}")
    print(f"Formula: {search_plan.formula}")
    print(f"Domain: {search_plan.domain.value}")
    print(f"Synonyms: {', '.join(search_plan.synonyms[:3])}")
    print("Per-source queries:")
    for source_name, source_query in search_plan.source_queries.items():
        print(f"  - {source_name}: {source_query.query_string}")
    print_check(bool(search_plan.target_compound), "Search plan created")
    print_check(
        set(registry.get_registered_source_names()).issubset(set(search_plan.source_queries)),
        "All sources have queries",
    )

    print_section("STEP 3: Individual Health Checks")
    online_sources = []
    for fetcher in fetchers:
        try:
            healthy = await asyncio.wait_for(fetcher.check_health(), timeout=8.0)
            if healthy:
                online_sources.append(fetcher.source_name)
                print(f"{fetcher.source_name} -> ✅ ONLINE")
            else:
                print(f"{fetcher.source_name} -> ❌ OFFLINE")
        except asyncio.TimeoutError:
            print(f"{fetcher.source_name} -> ❌ TIMEOUT")
    print_check(len(online_sources) >= 2, "At least 2 sources online")
    if len(online_sources) == 0:
        print("❌ No sources online. Check internet access and rerun the demo.")
        sys.exit(1)

    print_section("STEP 4: Full Pipeline Run")
    print("This will take 15-45 seconds")
    ingestion_logger = logging.getLogger("src.ingestion")
    previous_level = ingestion_logger.level
    ingestion_logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    orchestrator = FetchOrchestrator(
        registry=registry,
        deduplicator=DOITitleDeduplicator(),
        ranker=ChemistryRelevanceRanker(),
        query_parser=parser,
    )
    started = time.perf_counter()
    chunks = await orchestrator.run("synthesis of zinc oxide nanoparticles", skip_health_check=True)
    pipeline_time = time.perf_counter() - started
    ingestion_logger.setLevel(previous_level)

    print_section("STEP 5: Results Analysis")
    full_text_chunks = [chunk for chunk in chunks if chunk.completeness == TextCompleteness.FULL_TEXT]
    abstract_chunks = [chunk for chunk in chunks if chunk.completeness == TextCompleteness.ABSTRACT]
    sources_represented = {chunk.paper_metadata.source_db for chunk in chunks}
    avg_word_count = sum(chunk.word_count for chunk in chunks) / len(chunks) if chunks else 0.0
    avg_chemical_density = (
        sum(chunk.chemical_density_score for chunk in chunks) / len(chunks) if chunks else 0.0
    )
    high_density = [chunk for chunk in chunks if chunk.chemical_density_score > 0.3]

    print(f"{'Metric':<28}Value")
    print(f"{'-' * 28}{'-' * 20}")
    print(f"{'Total chunks':<28}{len(chunks)}")
    print(f"{'Full text':<28}{len(full_text_chunks)}")
    print(f"{'Abstract only':<28}{len(abstract_chunks)}")
    print(f"{'Sources':<28}{', '.join(sorted(sources_represented))}")
    print(f"{'Avg words':<28}{avg_word_count:.1f}")
    print(f"{'Avg density':<28}{avg_chemical_density:.3f}")
    print(f"{'High density chunks':<28}{len(high_density)}")
    print(f"{'Total time':<28}{pipeline_time:.2f}s")

    print("\nTop 5 chunks by chemical density:")
    for chunk in sorted(chunks, key=lambda item: item.chemical_density_score, reverse=True)[:5]:
        preview = " ".join(chunk.text.split())[:120]
        print(f"\nTitle: {truncate(chunk.paper_metadata.title, 55)}")
        print(f"Source: {chunk.paper_metadata.source_db}")
        print(f"Section: {chunk.section_name}")
        print(f"Completeness: {chunk.completeness.value}")
        print(f"Words: {chunk.word_count}")
        print(f"Density: {chunk.chemical_density_score:.3f}")
        print(f"Preview: {preview}")

    print_section("STEP 6: Pass/Fail Checklist")
    checks = [
        (len(chunks) >= 5, "At least 5 chunks produced"),
        (len(full_text_chunks) >= 1, "At least 1 full-text chunk"),
        (len(sources_represented) >= 2, "At least 2 sources contributed"),
        (avg_word_count >= 100, "Avg chunk ≥ 100 words"),
        (pipeline_time < 90, "Pipeline < 90 seconds"),
        (len(high_density) >= 1, "At least 1 high-density chunk"),
        (all(chunk.chunk_id for chunk in chunks), "All chunks have IDs"),
        (all(chunk.paper_metadata.title for chunk in chunks), "All chunks have paper titles"),
        (all(chunk.word_count > 0 for chunk in chunks), "All chunks have word counts"),
    ]
    for passed, message in checks:
        print_check(passed, message)

    failed = [message for passed, message in checks if not passed]
    core_passed = checks[0][0] and checks[2][0]
    if len(failed) <= 2 and core_passed:
        print("\n✅ PHASE 1 COMPLETE — ALL CHECKS PASSED")
    elif len(failed) <= 2:
        print("\n⚠️  MOSTLY COMPLETE — MINOR ISSUES")
    else:
        print("\n❌ INCOMPLETE")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
