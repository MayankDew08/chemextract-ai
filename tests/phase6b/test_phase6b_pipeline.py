from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase6b_demo_script_completes_end_to_end() -> None:
    """The Phase 6B demo script should validate PDF and URL input support."""

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "demo/phase6b/run_phase6b_demo.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "✅ Section extractor works" in result.stdout
    assert "✅ Chemical density scoring works" in result.stdout
    assert "✅ Paywall URL returns proper alternatives" in result.stdout
    assert "✅ PHASE 6B COMPLETE — PDF PARSER AND URL FETCHER READY" in result.stdout
