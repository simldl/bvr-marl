"""
Smoke tests for the GUI application.

Verifies that:
- All GUI component modules can be imported (no stale paths, no broken imports)
- The services.training module can be imported and its functions are callable
- The GUI launcher entry point exists
"""

import pytest

pytestmark = pytest.mark.smoke


def test_gui_component_imports():
    """All GUI components can be imported without error."""
    from bvr_marl_core.gui.components import config_validator, output_paths
    from bvr_marl_core.gui.components.analysis_interface import analysis_interface
    from bvr_marl_core.gui.components.checkpoint_picker import (
        checkpoint_picker,
        find_campaign_checkpoints,
        find_train_config_for_checkpoint,
        list_checkpoints,
        list_model_names,
        list_runs,
    )
    from bvr_marl_core.gui.components.config_builder import config_builder
    from bvr_marl_core.gui.components.tacview_generator import tacview_generator
    from bvr_marl_core.gui.components.training_dashboard import training_dashboard
    from bvr_marl_core.gui.components.visualization_config_builder import (
        visualization_config_builder,
    )
    from bvr_marl_core.gui.components.visualization_panel import visualization_panel


def test_gui_services_training_import():
    """services.training can be imported and exposes expected functions."""
    from bvr_marl_core.services.training import (
        build_simple_training_cmd,
        build_training_cmd,
        launch_background_process,
        list_training_configs,
    )

    assert callable(list_training_configs)
    assert callable(build_training_cmd)
    assert callable(build_simple_training_cmd)
    assert callable(launch_background_process)


def test_gui_services_visualization_import():
    """services.visualization can be imported and exposes expected symbols."""
    from bvr_marl_core.services.visualization import (
        VISUALIZATION_MODES,
        build_visualization_cmd,
        find_checkpoint_files,
    )

    assert callable(build_visualization_cmd)
    assert callable(find_checkpoint_files)
    assert "standard" in VISUALIZATION_MODES.values()


def test_gui_services_visualization_mode_routing():
    """Each visualization mode invokes its own distinct module."""
    from bvr_marl_core.services.visualization import build_visualization_cmd

    expected = {
        "standard": "live_view",
        "behavior-tree": "bvr-view-bt",
        "commands": "live_view_commands",
    }
    for mode, module_suffix in expected.items():
        cmd = build_visualization_cmd(mode=mode)
        joined = " ".join(cmd)
        assert module_suffix in joined, (
            f"Mode '{mode}' should invoke '{module_suffix}', got: {joined}"
        )


def test_gui_services_tacview_import():
    """services.tacview can be imported and build_tacview_cmd works."""
    from bvr_marl_core.services.tacview import build_tacview_cmd

    cmd = build_tacview_cmd(controller="rl", frames=10)
    assert isinstance(cmd, list)
    assert "bvr_marl_core.tacview.generate" in " ".join(cmd)


def test_gui_tacview_output_path_matches_generator_default():
    """Tacview output information points at the generator's default ACMI folder."""
    from bvr_marl_core.gui.components.output_paths import get_output_paths
    from bvr_marl_core.utils.paths import tacview_output_root

    assert get_output_paths()["tacview"]["tacview_files"] == tacview_output_root()


def test_gui_services_processes_import():
    """services.processes can be imported and ProcessMonitor is instantiable."""
    from bvr_marl_core.services.processes import ProcessMonitor, ProcessRecord

    monitor = ProcessMonitor("_test_smoke_processes.json")
    assert callable(monitor.load)
    assert callable(monitor.update)


def test_gui_services_configs_import():
    """services.configs can be imported and list functions return lists."""
    from bvr_marl_core.services.configs import (
        list_training_configs,
        list_visualization_configs,
    )

    assert isinstance(list_training_configs(), list)
    assert isinstance(list_visualization_configs(), list)


def test_gui_services_training_list_configs():
    """list_training_configs() returns a list (possibly empty if no configs installed)."""
    from bvr_marl_core.services.training import list_training_configs

    configs = list_training_configs()
    assert isinstance(configs, list)


def test_gui_services_training_build_cmd():
    """build_training_cmd() returns a non-empty list starting with the Python executable."""
    import sys

    from bvr_marl_core.services.training import build_training_cmd

    cmd = build_training_cmd()
    assert isinstance(cmd, list)
    assert len(cmd) >= 2
    # Command must invoke python or the train module
    joined = " ".join(cmd)
    assert "train" in joined.lower() or "python" in joined.lower()


def test_gui_launcher_entry_point_exists():
    """The gui_launcher module exists and exposes a main() function."""
    from bvr_marl_core import gui_launcher

    assert callable(getattr(gui_launcher, "main", None)), "gui_launcher.main() must be callable"


def test_gui_app_module_exists():
    """The gui.app module is importable (Streamlit import guard not an issue at import time)."""
    # We only check the module is importable; we don't call main() as it requires
    # a live Streamlit session.
    import importlib

    spec = importlib.util.find_spec("bvr_marl_core.gui.app")
    assert spec is not None, "bvr_marl_core.gui.app module could not be found"


def test_model_run_widget_keys_allow_duplicate_names(tmp_path):
    """Model rows with the same display name still get unique Streamlit keys."""
    from bvr_marl_core.gui.components.analysis_interface import _model_run_widget_key

    first_path = tmp_path / "models_a" / "gui_training_model"
    second_path = tmp_path / "models_b" / "gui_training_model"
    first_path.mkdir(parents=True)
    second_path.mkdir(parents=True)

    first_key = _model_run_widget_key(
        "eval_viz", {"name": "gui_training_model", "path": str(first_path)}
    )
    second_key = _model_run_widget_key(
        "eval_viz", {"name": "gui_training_model", "path": str(second_path)}
    )

    assert first_key.startswith("eval_viz_")
    assert second_key.startswith("eval_viz_")
    assert first_key != second_key


def test_checkpoint_picker_helpers(tmp_path):
    """list_model_names / list_runs / list_checkpoints work on a synthetic models/ tree."""
    from bvr_marl_core.gui.components.checkpoint_picker import (
        list_checkpoints,
        list_model_names,
        list_runs,
    )

    # Build a minimal models/ tree
    run_dir = (
        tmp_path / "my_model" / "PPO_SimplifiedMultiAgentEnv_abc12_00000_0_2026-03-20_14-31-51"
    )
    for i in range(3):
        (run_dir / f"checkpoint_{i:06d}").mkdir(parents=True)

    models = list_model_names(tmp_path)
    assert models == ["my_model"]

    runs = list_runs("my_model", tmp_path)
    assert len(runs) == 1
    label, path = runs[0]
    assert "2026-03-20" in label
    assert "14:31:51" in label

    checkpoints = list_checkpoints(path)
    assert len(checkpoints) == 3
    # Sorted ascending; last is latest
    assert checkpoints[-1][0] == "checkpoint #2"


def test_find_campaign_checkpoints(tmp_path):
    """Campaign checkpoint discovery returns the latest checkpoint for each campaign model."""
    from bvr_marl_core.gui.components.checkpoint_picker import find_campaign_checkpoints

    model_dir = tmp_path / "campaign_alpha" / "blue_policy"
    run_dir = model_dir / "PPO_Env_abc12_00000_0_2026-03-20_14-31-51"
    for i in (1, 3, 2):
        (run_dir / f"checkpoint_{i:06d}").mkdir(parents=True)
    (model_dir / "run_manifest.json").write_text('{"campaign_id": "campaign_alpha"}\n')

    standalone_model = tmp_path / "standalone_policy"
    standalone_run = standalone_model / "PPO_Env_abc12_00000_0_2026-03-20_14-31-51"
    (standalone_run / "checkpoint_000009").mkdir(parents=True)
    (standalone_model / "run_manifest.json").write_text("{}\n")

    checkpoints = find_campaign_checkpoints(tmp_path)

    assert len(checkpoints) == 1
    _mtime, label, checkpoint_path = checkpoints[0]
    assert label == "campaign_alpha / blue_policy"
    assert checkpoint_path.endswith("checkpoint_000003")


def test_find_train_config_for_checkpoint(tmp_path):
    """find_train_config_for_checkpoint locates train_config.yaml next to the model."""
    from bvr_marl_core.gui.components.checkpoint_picker import find_train_config_for_checkpoint

    # Build a minimal models/ tree with a train_config.yaml
    run_dir = tmp_path / "my_model" / "PPO_Env_abc12_00000_0_2026-03-20_14-31-51"
    ckpt_dir = run_dir / "checkpoint_000005"
    ckpt_dir.mkdir(parents=True)
    train_cfg = tmp_path / "my_model" / "train_config.yaml"
    train_cfg.write_text("training_mode: simplified\n")

    result = find_train_config_for_checkpoint(ckpt_dir)
    assert result is not None
    assert result == str(train_cfg)

    # Without train_config.yaml but with a "Simplified" run dir name, returns None
    # (no configs/training/ in tmp_path, so heuristic finds nothing)
    train_cfg.unlink()
    assert find_train_config_for_checkpoint(ckpt_dir) is None

    autotune_cfg = tmp_path / "my_model" / "autotune_train_config.yaml"
    autotune_cfg.write_text("training_mode: bvr\n")
    assert find_train_config_for_checkpoint(ckpt_dir) == str(autotune_cfg)
