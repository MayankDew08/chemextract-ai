from __future__ import annotations

from pathlib import Path


def test_paywall_notice_offers_pdf_parser_fallback() -> None:
    """A paywalled URL should guide users into the PDF parser pipeline."""

    dashboard_html = Path("src/api/templates/dashboard.html").read_text(encoding="utf-8")
    dashboard_js = Path("src/api/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "paywallPdfFile" in dashboard_html
    assert "uploadPaywallPDF()" in dashboard_html
    assert "ChemExtract will run the PDF parser pipeline" in dashboard_html
    assert "async function uploadPaywallPDF()" in dashboard_js
    assert "fetch('/api/upload-pdf'" in dashboard_js
    assert "connectProgressSocket(data.job_id, progressContainer, paywallPdfBtn)" in dashboard_js
