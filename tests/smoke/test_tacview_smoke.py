"""
Smoke tests for Tacview scenario generation.

Verifies that:
- run_tacview_scenario completes without error (no checkpoint, no Ray)
- The output ACMI file is created and non-empty
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_tacview_imports():
    """Tacview generate module can be imported."""
    from air_to_air_rl.tacview.generate import DefaultModel, run_tacview_scenario


def test_tacview_run_no_checkpoint(tmp_path):
    """run_tacview_scenario generates a valid ACMI file using the default random model."""
    from air_to_air_rl.tacview.generate import run_tacview_scenario

    acmi_path = str(tmp_path / "smoke_test.acmi")
    run_tacview_scenario(
        frames=10,
        acmi_path=acmi_path,
        checkpoint_path=None,
        seed=42,
    )

    out = Path(acmi_path)
    assert out.exists(), f"ACMI file was not created at {acmi_path}"
    assert out.stat().st_size > 0, "ACMI file is empty"


def test_tacview_acmi_content(tmp_path):
    """Generated ACMI file contains expected Tacview header markers."""
    from air_to_air_rl.tacview.generate import run_tacview_scenario

    acmi_path = str(tmp_path / "content_test.acmi")
    run_tacview_scenario(frames=5, acmi_path=acmi_path, seed=0)

    content = Path(acmi_path).read_text(encoding="utf-8", errors="replace")
    assert "FileType=text/acmi" in content or "TACVIEW" in content.upper() or "#" in content, (
        "ACMI file does not look like a valid Tacview file"
    )
