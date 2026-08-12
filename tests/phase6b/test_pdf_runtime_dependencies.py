from __future__ import annotations

import importlib.util


def test_pdf_parser_runtime_dependencies_are_available() -> None:
    """The uv-managed environment must include both PDF parser backends."""

    assert importlib.util.find_spec("pymupdf4llm") is not None
    assert importlib.util.find_spec("pdfplumber") is not None
