from __future__ import annotations

from pathlib import Path

from src.ingestion.pdf_parser import PDFParser


def test_pdf_parser_splits_oversized_sections_with_overlap() -> None:
    """Oversized sections stay below the model budget and retain boundary context."""

    parser = PDFParser()
    words = [f"word{index}" for index in range(3200)]
    chunks = parser._split_if_too_long("Experimental", " ".join(words))

    assert parser.MAX_CHUNK_WORDS == 1500
    assert len(chunks) == 3
    assert all(len(text.split()) <= 1500 for _, text in chunks)
    assert chunks[0][1].split()[-150:] == chunks[1][1].split()[:150]
    assert chunks[1][0] == "Experimental_part_2"


def test_pdf_parser_detects_numbered_synthesis_subsections() -> None:
    """Numbered subsection headers should become independently extractable chunks."""

    text = "\n".join(
        [
            "2.2 Synthesis of ZnO",
            "Zinc acetate was dissolved in methanol and stirred at 65°C for 2 hours. " * 12,
            "2.3 Preparation of TiO2",
            "Titanium isopropoxide was mixed with ethanol and calcined at 400°C. " * 12,
        ]
    )
    sections = PDFParser()._find_experimental_sections(text)
    names = [name.lower() for name, _ in sections]

    assert any("2.2 synthesis" in name for name in names)
    assert any("2.3 preparation" in name for name in names)


def test_upload_pipeline_allows_ten_preloaded_chunks() -> None:
    """The preloaded upload route must not retain the old three-chunk limit."""

    source = Path("src/api/routes/upload.py").read_text(encoding="utf-8")

    assert "max_chunks_per_paper=10" in source
