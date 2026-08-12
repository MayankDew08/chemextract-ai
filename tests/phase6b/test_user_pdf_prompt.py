from __future__ import annotations

from pathlib import Path


def test_phase6b_demo_supports_user_pdf_after_paywall() -> None:
    """The Phase 6B demo should offer a user-PDF fallback after paywall detection."""

    project_root = Path(__file__).resolve().parents[2]
    script = (project_root / "demo/phase6b/run_phase6b_demo.py").read_text()

    assert "--pdf" in script
    assert "Paywall detected" in script
    assert "PDF parser pipeline" in script
    assert "PDF PARSER REPORT" in script
    assert "USER PDF PIPELINE REPORT" in script
    assert "PDF fallback prompt skipped (non-interactive run)." in script
    assert 'llm_provider="ollama"' in script
