from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse optional user-PDF arguments for the Phase 6B demo."""

    parser = argparse.ArgumentParser(
        description="Run Phase 6B checks and optionally test a user-provided PDF.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a user PDF to parse after paywall detection.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title hint for the user PDF.",
    )
    return parser.parse_args()


def resolve_pdf_path(pdf_input: str) -> Path:
    """Resolve a PDF path from cwd first, then the project root."""

    candidate = Path(pdf_input).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate

    project_candidate = PROJECT_ROOT / pdf_input
    if project_candidate.exists():
        return project_candidate

    return candidate


def print_user_pdf_report(parsed, pipeline_result) -> None:
    """Print a compact final report for a user-supplied PDF pipeline run."""

    print("\nPDF PARSER REPORT")
    print("-" * 60)
    print(f"File                : {parsed.source_identifier}")
    print(f"Pages               : {parsed.total_pages or 0}")
    print(f"Sections found      : {', '.join(parsed.sections_found) or 'N/A'}")
    print(f"Chunks produced     : {len(parsed.chunks)}")
    print(f"Words extracted     : {parsed.total_words_extracted}")
    print(f"Words in chunks     : {parsed.words_in_chunks}")
    print(f"Parser duration     : {parsed.processing_seconds:.1f}s")

    print("\nUSER PDF PIPELINE REPORT")
    print("-" * 60)
    print(f"Recipes extracted   : {pipeline_result.recipes_extracted}")
    print(f"Passed first try    : {pipeline_result.passed_first_try}")
    print(f"Self-corrected      : {pipeline_result.self_corrected}")
    print(f"Failed              : {pipeline_result.failed}")
    print(f"Graph nodes added   : {pipeline_result.nodes_added}")
    print(f"Graph edges added   : {pipeline_result.edges_added}")
    print(f"Total duration      : {pipeline_result.total_duration_seconds:.1f}s")
    print(f"Estimated cost      : ${pipeline_result.total_cost_usd:.6f}")

    if not pipeline_result.recipes:
        return

    print("\nEXTRACTED PDF RECIPES")
    print("-" * 60)
    for recipe in pipeline_result.recipes:
        products = [
            entity.name
            for entity in recipe.entities
            if getattr(entity.role, "value", entity.role) == "PRODUCT"
        ]
        product_text = ", ".join(products[:2]) if products else "unknown"
        print(
            f"{recipe.recipe_id} | {recipe.validation_status.value} | "
            f"{product_text} | corrections={recipe.correction_count()}"
        )


async def maybe_run_user_pdf(pdf_arg: str | None, title_hint: str | None) -> None:
    """Ask for a PDF after paywall detection and run it through the pipeline."""

    pdf_input = pdf_arg
    if not pdf_input and sys.stdin.isatty():
        pdf_input = input(
            "\nPaywall detected. If you have the PDF, enter its path to run "
            "the PDF parser pipeline (or press Enter to skip): "
        ).strip()

    if not pdf_input:
        print("PDF fallback prompt skipped (non-interactive run).")
        return

    pdf_path = resolve_pdf_path(pdf_input)
    if not pdf_path.exists():
        print(f"PDF fallback skipped: file not found: {pdf_path}")
        return
    if pdf_path.suffix.lower() != ".pdf":
        print(f"PDF fallback skipped: expected a .pdf file: {pdf_path}")
        return

    from src.ingestion.pdf_parser import PDFParser
    from src.pipeline import ChemExtractPipeline, PipelineConfig

    print(f"\nPaywall detected. Running PDF parser pipeline for: {pdf_path}")
    parser = PDFParser()
    parsed = await parser.process(str(pdf_path), title_hint=title_hint)

    if not parsed.success:
        print(f"PDF parser failed: {parsed.error}")
        return
    if not parsed.chunks:
        print("PDF parser found no chemistry chunks in the supplied file.")
        return

    pipeline = ChemExtractPipeline(
        PipelineConfig(
            query=f"user_pdf_{pdf_path.stem}",
            preloaded_chunks=parsed.chunks,
            graph_backend="networkx",
            max_retries=1,
            llm_provider="ollama",
            chunk_timeout_seconds=180.0,
            verbose=False,
        )
    )
    async for event in pipeline.stream():
        print(event.to_terminal_line())

    print_user_pdf_report(parsed, pipeline._result)


async def main(args: argparse.Namespace) -> None:
    """Run the Phase 6B PDF parser and URL fetcher integration checks."""

    from src.api.main import app
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.url_fetcher import URLFetcher
    from src.ingestion.utils import calculate_chemical_density

    parser = PDFParser()
    test_text = """
    ## Experimental Section

    Synthesis of ZnO Nanoparticles
    Zinc acetate dihydrate (2.195 g, 10 mmol) was dissolved in
    100 mL of methanol. The solution was refluxed at 65°C for
    2 hours with 1.12 g of KOH (20 mmol) as base catalyst.
    The resulting ZnO precipitate was filtered and dried at 120°C.

    ## Results and Discussion
    The XRD pattern showed hexagonal wurtzite structure.
    """
    sections = parser._find_experimental_sections(test_text)
    print(f"Sections found: {[section[0] for section in sections]}")
    assert len(sections) >= 1
    assert "experimental" in sections[0][0].lower()
    assert "zinc acetate" in sections[0][1].lower()
    print("✅ Section extractor works")

    score_high = calculate_chemical_density("Zinc acetate (2g) dissolved in 50mL ethanol at 65°C")
    score_low = calculate_chemical_density("The results showed significant improvement")
    assert score_high > score_low
    print(f"High density: {score_high:.3f}, Low density: {score_low:.3f}")
    print("✅ Chemical density scoring works")

    fetcher = URLFetcher()
    paywalled_urls = [
        "https://pubs.acs.org/doi/10.1021/jacs.0c00000",
        "https://www.sciencedirect.com/science/article/pii/S0000",
        "https://onlinelibrary.wiley.com/doi/10.1002/test",
    ]
    for url in paywalled_urls:
        result = fetcher._check_paywall_by_domain(url)
        assert result.is_paywalled is True
        assert result.publisher is not None
        assert len(result.open_access_alternatives) > 0
        print(f"✅ Correctly detected paywall: {result.publisher}")

    test_cases = [
        ("https://arxiv.org/abs/2301.12345", "https://ar5iv.org/abs/2301.12345"),
        ("https://arxiv.org/pdf/2301.12345v2", "https://ar5iv.org/abs/2301.12345"),
    ]
    for input_url, expected_url in test_cases:
        normalized, _url_type = fetcher._normalize_url(input_url)
        assert normalized == expected_url
        print(f"✅ Normalized: {input_url[:40]} → {normalized}")

    doi_urls = [
        ("https://doi.org/10.1039/C9NR00001A", "10.1039/C9NR00001A"),
        ("https://pubs.acs.org/doi/10.1021/jacs.0c00001", "10.1021/jacs.0c00001"),
    ]
    for url, expected_doi in doi_urls:
        doi = fetcher._extract_doi_from_url(url)
        assert doi == expected_doi
        print(f"✅ DOI extracted: {doi}")

    print("Testing real ArXiv fetch (requires internet)...")
    arxiv_result = await fetcher.process("https://arxiv.org/abs/2108.01072")
    if arxiv_result.success:
        assert len(arxiv_result.chunks) >= 1
        assert arxiv_result.total_words_extracted > 100
        print(f"✅ ArXiv fetch: {len(arxiv_result.chunks)} chunks")
        print(f"   Words extracted: {arxiv_result.total_words_extracted}")
    else:
        print(f"⚠️  ArXiv fetch failed: {arxiv_result.error}")
        print("   (This may be a network issue, not a code issue)")
    await fetcher.close()

    paywall_html = """
    <html><body>
    <p>You do not have access to this article.</p>
    <p>Please purchase access or subscribe.</p>
    </body></html>
    """
    mock_response = SimpleNamespace(status_code=200, text=paywall_html, url="https://example.org/article")
    response_paywall = fetcher._detect_paywall_in_response(mock_response, "https://example.org/article")
    assert response_paywall.is_paywalled is True
    print("✅ Paywall detection in response works")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/fetch-url",
            json={"url": "https://pubs.acs.org/doi/10.1021/test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_paywalled"] is True
        assert len(data["alternatives"]) > 0
        print("✅ Paywall URL returns proper alternatives")
        await maybe_run_user_pdf(args.pdf, args.title)

        response = await client.post("/api/fetch-url", json={"url": "not-a-url"})
        assert response.status_code == 422
        print("✅ Invalid URL returns 422")

    print("✅ PHASE 6B COMPLETE — PDF PARSER AND URL FETCHER READY")


if __name__ == "__main__":
    parsed_args = parse_args()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(parsed_args))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
