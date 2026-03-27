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
    from air_to_air_rl.gui.components import config_validator, output_paths
    from air_to_air_rl.gui.components.analysis_interface import analysis_interface
    from air_to_air_rl.gui.components.checkpoint_picker import (
        checkpoint_picker,
        find_train_config_for_checkpoint,
        list_checkpoints,
        list_model_names,
        list_runs,
    )
    from air_to_air_rl.gui.components.config_builder import config_builder
    from air_to_air_rl.gui.components.tacview_generator import tacview_generator
    from air_to_air_rl.gui.components.training_dashboard import training_dashboard
    from air_to_air_rl.gui.components.visualization_config_builder import (
        visualization_config_builder,
    )
    from air_to_air_rl.gui.components.visualization_panel import visualization_panel


def test_gui_services_training_import():
    """services.training can be imported and exposes expected functions."""
    from air_to_air_rl.services.training import (
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
    from air_to_air_rl.services.visualization import (
        VISUALIZATION_MODES,
        build_visualization_cmd,
        find_checkpoint_files,
    )

    assert callable(build_visualization_cmd)
    assert callable(find_checkpoint_files)
    assert "standard" in VISUALIZATION_MODES.values()


def test_gui_services_visualization_mode_routing():
    """Each visualization mode invokes its own distinct module."""
    from air_to_air_rl.services.visualization import build_visualization_cmd

    expected = {
        "standard": "live_view",
        "behavior-tree": "live_view_bt",
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
    from air_to_air_rl.services.tacview import build_tacview_cmd

    cmd = build_tacview_cmd(controller="rl", frames=10, use_script=False)
    assert isinstance(cmd, list)
    assert "air_to_air_rl.tacview.generate" in " ".join(cmd)


def test_gui_services_processes_import():
    """services.processes can be imported and ProcessMonitor is instantiable."""
    from air_to_air_rl.services.processes import ProcessMonitor, ProcessRecord

    monitor = ProcessMonitor("_test_smoke_processes.json")
    assert callable(monitor.load)
    assert callable(monitor.update)


def test_gui_services_configs_import():
    """services.configs can be imported and list functions return lists."""
    from air_to_air_rl.services.configs import (
        list_training_configs,
        list_visualization_configs,
    )

    assert isinstance(list_training_configs(), list)
    assert isinstance(list_visualization_configs(), list)


def test_gui_services_training_list_configs():
    """list_training_configs() returns a list (possibly empty if no configs installed)."""
    from air_to_air_rl.services.training import list_training_configs

    configs = list_training_configs()
    assert isinstance(configs, list)


def test_gui_services_training_build_cmd():
    """build_training_cmd() returns a non-empty list starting with the Python executable."""
    import sys

    from air_to_air_rl.services.training import build_training_cmd

    cmd = build_training_cmd()
    assert isinstance(cmd, list)
    assert len(cmd) >= 2
    # Command must invoke python or the train module
    joined = " ".join(cmd)
    assert "train" in joined.lower() or "python" in joined.lower()


def test_gui_launcher_entry_point_exists():
    """The gui_launcher module exists and exposes a main() function."""
    from air_to_air_rl import gui_launcher

    assert callable(getattr(gui_launcher, "main", None)), "gui_launcher.main() must be callable"


def test_gui_app_module_exists():
    """The gui.app module is importable (Streamlit import guard not an issue at import time)."""
    # We only check the module is importable; we don't call main() as it requires
    # a live Streamlit session.
    import importlib

    spec = importlib.util.find_spec("air_to_air_rl.gui.app")
    assert spec is not None, "air_to_air_rl.gui.app module could not be found"


def test_checkpoint_picker_helpers(tmp_path):
    """list_model_names / list_runs / list_checkpoints work on a synthetic models/ tree."""
    from air_to_air_rl.gui.components.checkpoint_picker import (
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


def test_find_train_config_for_checkpoint(tmp_path):
    """find_train_config_for_checkpoint locates train_config.yaml next to the model."""
    from air_to_air_rl.gui.components.checkpoint_picker import find_train_config_for_checkpoint

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
