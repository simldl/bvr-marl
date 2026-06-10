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
    from bvr_marl_core.tacview.generate import DefaultModel, run_tacview_scenario


def test_tacview_infers_checkpoint_train_config(tmp_path):
    """Tacview finds campaign train configs saved next to checkpoint directories."""
    from bvr_marl_core.tacview.generate import infer_train_config_for_checkpoint

    model_dir = tmp_path / "campaign_alpha" / "blue_policy"
    checkpoint_dir = model_dir / "PPO_Env_00000_0_2026-03-20_14-31-51" / "checkpoint_000003"
    checkpoint_dir.mkdir(parents=True)
    train_config = model_dir / "autotune_train_config.yaml"
    train_config.write_text("training_mode: bvr\n", encoding="utf-8")

    assert infer_train_config_for_checkpoint(checkpoint_dir) == str(train_config)


def test_tacview_unique_filename_uses_output_root(tmp_path, monkeypatch):
    """Auto-generated Tacview filenames use the shared output-root helper."""
    output_root = tmp_path / "acmi"
    monkeypatch.setenv("BVR_TACVIEW_OUTPUT_DIR", str(output_root))

    from bvr_marl_core.tacview.generate import generate_unique_filename

    acmi_path = Path(generate_unique_filename())

    assert acmi_path.parent == output_root
    assert acmi_path.name.startswith("scenario_")
    assert acmi_path.suffix == ".acmi"


def test_tacview_run_no_checkpoint(tmp_path):
    """run_tacview_scenario generates a valid ACMI file using the default random model."""
    from bvr_marl_core.tacview.generate import run_tacview_scenario

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
    """Generated ACMI file contains frame and unit data, not just the header."""
    from bvr_marl_core.tacview.generate import run_tacview_scenario

    acmi_path = str(tmp_path / "content_test.acmi")
    run_tacview_scenario(frames=5, acmi_path=acmi_path, seed=0)

    content = Path(acmi_path).read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    assert "FileType=text/acmi/tacview" in lines
    assert any(line.startswith("#") for line in lines), "ACMI file contains no frame records"
    assert any(",T=" in line for line in lines), "ACMI file contains no unit transform records"
