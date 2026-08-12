from __future__ import annotations

from pathlib import Path


def test_user_input_pipeline_respects_runtime_llm_provider() -> None:
    """PDF and URL jobs should resolve the provider selected in session settings."""

    source = Path("src/api/routes/upload.py").read_text(encoding="utf-8")

    assert 'llm_provider="ollama"' not in source
