"""Regression coverage for structured HTML methods extraction."""

from __future__ import annotations

import httpx
import pytest

from src.ingestion.url_fetcher import URLFetcher


def test_url_fetcher_preserves_html_sections_and_isolates_synthesis() -> None:
    """A methods heading must produce a focused chunk instead of the whole article."""

    introduction = " ".join(["Considerable attention has been paid to the synthesis of nano-size materials."] * 80)
    characterization = " ".join(["The sample was characterized by diffraction spectroscopy."] * 40)
    thermal_analysis = (
        "Thermal behavior of prepared CuO samples was studied in air at 40 mL per minute. "
        "The samples were heated from room temperature to 800 °C while measurements were recorded "
        "continuously by the instrument to characterize mass loss and phase stability of each material."
    )
    photocatalysis_test = (
        "In a typical procedure, 75 mL dye solution was mixed with 10 mg of prepared CuO catalyst. "
        "The suspension was stirred for 30 minutes before irradiation to measure photocatalytic activity, "
        "and aliquots were collected periodically for absorption analysis during the degradation experiment."
    )
    html = f"""
    <html><head><title>CuO synthesis paper</title></head><body>
      <article>
        <h2>1. Introduction</h2><p>{introduction}</p>
        <h2>2. Experimental</h2>
        <h3>2.1. Materials</h3><p>Copper acetate and sodium hydroxide were analytical grade.</p>
        <h3>2.2. Synthesis of CuO NPs</h3>
        <p>In a typical procedure, 5 mmol copper acetate and 20 mmol sodium hydroxide were placed in a flask with 100 mL distilled water. The mixture was refluxed at 100 °C for 1 hour, cooled, centrifuged, washed, and dried at 60 °C for 3 hours to produce CuO nanoparticles.</p>
        <h3>2.3. Characterization</h3>
        <p>{thermal_analysis}</p><p>{photocatalysis_test}</p><p>{characterization}</p>
      </article>
    </body></html>
    """

    fetcher = URLFetcher()
    article_text, _title = fetcher._extract_article_text(html, "pmc.ncbi.nlm.nih.gov")
    sections = fetcher._find_experimental_sections(article_text)

    assert sections
    assert len(sections) == 1
    assert "5 mmol copper acetate" in sections[0][1]
    synthesis_text = next(text for _name, text in sections if "5 mmol copper acetate" in text)
    assert introduction not in synthesis_text
    assert characterization not in synthesis_text
    assert len(synthesis_text.split()) < 250


@pytest.mark.asyncio
async def test_url_fetcher_attaches_exact_requested_url_to_every_chunk() -> None:
    """A user-submitted paper URL must remain attached to acquired research text."""

    source_url = "https://example.org/papers/cuo-synthesis?view=full"
    procedure = " ".join(
        [
            "Copper acetate 5 mmol was dissolved in 100 mL water, mixed with sodium hydroxide, "
            "heated at 100 °C for 1 hour, centrifuged, washed, and dried to produce CuO nanoparticles."
        ]
        * 5
    )
    html = f"<html><body><article><h2>Synthesis of CuO</h2><p>{procedure}</p></article></body></html>"

    def respond(request: httpx.Request) -> httpx.Response:
        """Return a deterministic open paper page."""

        return httpx.Response(200, text=html, request=request)

    fetcher = URLFetcher()
    fetcher._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        result = await fetcher.process(source_url)
    finally:
        await fetcher.close()

    assert result.success
    assert result.chunks
    assert all(chunk.paper_metadata.open_access_url == source_url for chunk in result.chunks)
