from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase6a_demo_script_completes_end_to_end() -> None:
    """The Phase 6A demo script should complete through the integrated pipeline."""

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "demo/phase6a/run_phase6a_demo.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline initialized" in result.stdout
    assert "[starting]" in result.stdout
    assert "[fetched]" in result.stdout
    assert "[extracting]" in result.stdout
    assert "[complete]" in result.stdout
    assert "✅ PHASE 6A COMPLETE — END-TO-END PIPELINE CONNECTED" in result.stdout
