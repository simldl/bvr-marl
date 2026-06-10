"""
Smoke and functional tests for all bvr-marl-behavior entry-point executables.

Covers:
  - bvr-train-simple  (training/train_simple.py:main)
  - bvr-view and variants  (visualization/live_view*.py:main)
  - bvr-tacview  (tacview/generate.py:main)
  - bvr-export-plots  (analysis/export_plots.py:main)
  - bvr-gui  (gui_launcher.py:main)  <- already in test_gui_smoke.py; checked for completeness
  - Checkpoint loading helpers  (DefaultModel, TrainedModelWrapper)

Tests that spin up a real simulator use a small frame count so they finish
quickly while still verifying the end-to-end plumbing works.
"""

import importlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_entry_point(module_dotted: str) -> tuple[bool, str]:
    """Check whether a module exports a callable main().

    Returns:
        (ok, message)  ok is True on success; message is empty on success
        and describes the actual failure otherwise.  The three distinct
        failure modes are:
          - import failed:   the module (or one of its dependencies) raised
                             ImportError / ModuleNotFoundError at import time.
          - missing main():  the module imported successfully but does not
                             define a ``main`` attribute at all.
          - non-callable:    a ``main`` attribute exists but is not callable
                             (e.g. it was accidentally assigned a non-function
                             value).
    """
    try:
        mod = importlib.import_module(module_dotted)
    except ImportError as exc:
        return False, f"import failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"import raised {type(exc).__name__}: {exc}"

    main_attr = getattr(mod, "main", None)
    if main_attr is None:
        return False, "main() attribute not found on module"
    if not callable(main_attr):
        return False, (
            f"main attribute exists but is not callable (type={type(main_attr).__name__!r})"
        )
    return True, ""


def _has_main(module_dotted: str) -> bool:
    """Return True if the named module exports a callable main()."""
    ok, _ = _check_entry_point(module_dotted)
    return ok


# ---------------------------------------------------------------------------
# Entry-point existence
# ---------------------------------------------------------------------------


class TestEntryPointsExist:
    """Every console_scripts entry point must expose a callable main()."""

    def test_train_simple_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.training.train_simple")
        assert ok, f"bvr_marl_core.training.train_simple  {msg}"

    def test_live_view_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.visualization.live_view")
        assert ok, f"bvr_marl_core.visualization.live_view  {msg}"

    def test_live_view_commands_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.visualization.live_view_commands")
        assert ok, f"bvr_marl_core.visualization.live_view_commands  {msg}"

    def test_tacview_generate_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.tacview.generate")
        assert ok, f"bvr_marl_core.tacview.generate  {msg}"

    def test_export_plots_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.analysis.export_plots")
        assert ok, f"bvr_marl_core.analysis.export_plots  {msg}"

    def test_gui_launcher_main_callable(self):
        ok, msg = _check_entry_point("bvr_marl_core.gui_launcher")
        assert ok, f"bvr_marl_core.gui_launcher  {msg}"


# ---------------------------------------------------------------------------
# DefaultModel
# ---------------------------------------------------------------------------


class TestDefaultModel:
    """DefaultModel must return a valid 10-dim numpy action for any observation."""

    def test_compute_action_returns_numpy_array(self):
        from bvr_marl_core.visualization.model_wrapper.model_wrapper import DefaultModel

        mock_env = Mock()
        model = DefaultModel(mock_env)
        obs = np.zeros(10, dtype=np.float32)
        action = model.compute_single_action(obs, "A1")

        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        assert action.shape == (10,), f"Expected shape (10,), got {action.shape}"
        assert action.dtype == np.float32, f"Expected float32, got {action.dtype}"

    def test_compute_action_agent_id_optional(self):
        from bvr_marl_core.visualization.model_wrapper.model_wrapper import DefaultModel

        mock_env = Mock()
        model = DefaultModel(mock_env)
        obs = np.zeros(10, dtype=np.float32)
        # Should not raise even without agent_id
        action = model.compute_single_action(obs, agent_id=None)
        assert action is not None

    def test_tacview_default_model_compute_action(self):
        """DefaultModel inside tacview/generate.py also returns 10-dim actions."""
        from bvr_marl_core.tacview.generate import DefaultModel as TacviewDefaultModel

        mock_env = Mock()
        model = TacviewDefaultModel(mock_env)
        obs = np.zeros(10, dtype=np.float32)
        action = model.compute_single_action(obs, "A1")

        assert isinstance(action, np.ndarray)
        assert action.shape == (10,)


# ---------------------------------------------------------------------------
# TrainedModelWrapper  checkpoint loading (mocked)
# ---------------------------------------------------------------------------


class TestTrainedModelWrapperLoading:
    """TrainedModelWrapper must load from a properly structured checkpoint directory."""

    def _build_fake_checkpoint(
        self, tmp_path: Path, policy_names: tuple[str, ...] = ("agent_policy",)
    ) -> Path:
        """Build the minimal directory tree that TrainedModelWrapper expects."""
        rl_module_dir = tmp_path / "checkpoint" / "learner_group" / "learner" / "rl_module"
        rl_module_dir.mkdir(parents=True)
        for policy_name in policy_names:
            (rl_module_dir / policy_name).mkdir()
        return tmp_path / "checkpoint"

    def test_loads_from_checkpoint_with_mocked_ray(self, tmp_path):
        """TrainedModelWrapper initializes without error when RLModule.from_checkpoint is mocked."""
        try:
            from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
                TrainedModelWrapper,
            )
        except ImportError as e:
            pytest.skip(f"ray/rllib not available: {e}")

        checkpoint_path = self._build_fake_checkpoint(tmp_path)
        mock_env = Mock()

        # Build a mock MultiRLModule that has the keys() interface
        mock_rl_module = Mock()
        mock_rl_module.keys.return_value = ["agent_policy", "opponent_policy"]

        with (
            patch("ray.init"),
            patch("ray.rllib.core.rl_module.RLModule.from_checkpoint", return_value=mock_rl_module),
        ):
            wrapper = TrainedModelWrapper(str(checkpoint_path), mock_env)

        assert wrapper.rl_module is mock_rl_module

    def test_compute_action_with_mocked_module(self, tmp_path):
        """compute_single_action returns a clipped numpy array via the mocked RLModule."""
        try:
            from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
                TrainedModelWrapper,
            )
        except ImportError as e:
            pytest.skip(f"ray/rllib not available: {e}")

        import torch
        from ray.rllib.core.columns import Columns

        checkpoint_path = self._build_fake_checkpoint(tmp_path)
        mock_env = Mock()

        # Fake action output from forward_inference
        fake_actions = torch.zeros(1, 10)  # (batch=1, action_dim=10)
        mock_policy_module = Mock()
        mock_policy_module.forward_inference.return_value = {Columns.ACTIONS: fake_actions}

        mock_rl_module = Mock()
        mock_rl_module.keys.return_value = ["agent_policy", "opponent_policy"]
        mock_rl_module.__getitem__ = Mock(return_value=mock_policy_module)

        with (
            patch("ray.init"),
            patch("ray.rllib.core.rl_module.RLModule.from_checkpoint", return_value=mock_rl_module),
        ):
            wrapper = TrainedModelWrapper(str(checkpoint_path), mock_env)

        obs = np.zeros(10, dtype=np.float32)
        action = wrapper.compute_single_action(obs, "A1")

        assert isinstance(action, np.ndarray)
        assert action.shape == (10,)
        # Actions must be clipped to [0, 1]
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_compute_action_with_current_team_separate_module_names(self, tmp_path):
        """compute_single_action maps A/B agents to attacker/defender policy modules."""
        try:
            from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
                TrainedModelWrapper,
            )
        except ImportError as e:
            pytest.skip(f"ray/rllib not available: {e}")

        import torch
        from ray.rllib.core.columns import Columns

        checkpoint_path = self._build_fake_checkpoint(
            tmp_path,
            policy_names=("attacker_policy", "defender_policy"),
        )
        mock_env = Mock()

        attacker_module = Mock()
        attacker_module.forward_inference.return_value = {Columns.ACTIONS: torch.full((1, 10), 0.2)}
        defender_module = Mock()
        defender_module.forward_inference.return_value = {Columns.ACTIONS: torch.full((1, 10), 0.8)}
        modules = {
            "attacker_policy": attacker_module,
            "defender_policy": defender_module,
        }

        mock_rl_module = MagicMock()
        mock_rl_module.keys.return_value = list(modules)
        mock_rl_module.__getitem__.side_effect = modules.__getitem__

        with (
            patch("ray.init"),
            patch("ray.rllib.core.rl_module.RLModule.from_checkpoint", return_value=mock_rl_module),
        ):
            wrapper = TrainedModelWrapper(str(checkpoint_path), mock_env)

        obs = np.zeros(10, dtype=np.float32)

        assert np.allclose(wrapper.compute_single_action(obs, "A0"), 0.2)
        assert np.allclose(wrapper.compute_single_action(obs, "B0"), 0.8)


# ---------------------------------------------------------------------------
# Tacview scenario  actual functional runs
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTacviewScenarioRuns:
    """Run a real tacview scenario with a small frame count to verify end-to-end plumbing."""

    def test_rl_scenario_runs_with_default_model(self, tmp_path):
        """run_tacview_scenario executes 5 steps using DefaultModel (random actions)."""
        from bvr_marl_core.tacview.generate import run_tacview_scenario

        acmi_path = str(tmp_path / "test_scenario.acmi")
        # No checkpoint  DefaultModel used; 5 frames is enough to verify the loop runs
        run_tacview_scenario(frames=5, acmi_path=acmi_path, progress_interval=10)

        # ACMI file must have been written
        assert Path(acmi_path).exists(), "ACMI output file was not created"
        assert Path(acmi_path).stat().st_size > 0, "ACMI output file is empty"

    def test_rl_scenario_with_seed_is_deterministic(self, tmp_path):
        """The same seed produces the same ACMI file (tacview frame count matches)."""
        from bvr_marl_core.tacview.generate import run_tacview_scenario

        path_a = str(tmp_path / "seed_a.acmi")
        path_b = str(tmp_path / "seed_b.acmi")

        run_tacview_scenario(frames=10, acmi_path=path_a, seed=42, progress_interval=20)
        run_tacview_scenario(frames=10, acmi_path=path_b, seed=42, progress_interval=20)

        # Both runs should produce files of the same size (deterministic with fixed seed)
        size_a = Path(path_a).stat().st_size
        size_b = Path(path_b).stat().st_size
        assert size_a == size_b, (
            f"Tacview output size differs between identical seeds: {size_a} vs {size_b}"
        )


# ---------------------------------------------------------------------------
# Visualization live_view  functional run with mocked display
# ---------------------------------------------------------------------------


class TestLiveViewRun:
    """run_live_view must complete without error when display calls are mocked."""

    def test_run_live_view_random_model_no_display(self, tmp_path):
        """run_live_view with use_random=True runs through the simulation loop."""
        try:
            import bvr_marl_core.visualization.reward_logging.reward_logger as _rl_mod
            import bvr_marl_core.visualization.scenplotter.video_generation as _vg_mod
            from bvr_marl_core.visualization.live_view import run_live_view
        except ImportError as e:
            pytest.skip(f"visualization deps not available: {e}")

        minimal_viz_cfg = {
            "env_type": "simplified",
            "visualization": {
                "frames": 3,
                "interval": 100,
                "save_video": False,
                "save_rewards": False,
                "symbol_mode": "nato",
                "dpi": 100,
                "symbol_scale": 1.0,
            },
        }
        minimal_train_cfg = {
            "training_mode": "simplified",
            "env": {
                "num_agents_per_team": 1,
                "map_size": 300,
                "max_steps": 10,
                "simulation_config": {},
                "weapon_config": {},
            },
        }

        with (
            patch(
                "bvr_marl_core.visualization.live_view.load_viz_config",
                return_value=minimal_viz_cfg,
            ),
            patch(
                "bvr_marl_core.visualization.live_view.load_train_config",
                return_value=minimal_train_cfg,
            ),
            patch.object(_vg_mod, "live_simulation") as mock_sim,
            patch.object(_rl_mod, "save_reward_logs"),
        ):
            run_live_view(use_random=True, save_rewards=False)

        # live_simulation must have been called once with a model and env
        mock_sim.assert_called_once()
        call_kwargs = mock_sim.call_args.kwargs
        assert "trained_model" in call_kwargs
        assert "env" in call_kwargs
        assert call_kwargs["frames"] == 3

    def test_run_live_view_uses_default_model_without_checkpoint(self, tmp_path):
        """Without a checkpoint path, run_live_view uses DefaultModel."""
        try:
            import bvr_marl_core.visualization.reward_logging.reward_logger as _rl_mod
            import bvr_marl_core.visualization.scenplotter.video_generation as _vg_mod
            from bvr_marl_core.visualization.live_view import run_live_view
            from bvr_marl_core.visualization.model_wrapper.model_wrapper import DefaultModel
        except ImportError as e:
            pytest.skip(f"visualization deps not available: {e}")

        with (
            patch(
                "bvr_marl_core.visualization.live_view.load_viz_config",
                return_value={"env_type": "simplified", "visualization": {"frames": 2}},
            ),
            patch(
                "bvr_marl_core.visualization.live_view.load_train_config",
                return_value={
                    "training_mode": "simplified",
                    "env": {"max_steps": 5, "simulation_config": {}, "weapon_config": {}},
                },
            ),
            patch.object(_vg_mod, "live_simulation") as mock_sim,
            patch.object(_rl_mod, "save_reward_logs"),
        ):
            run_live_view(checkpoint_path=None, use_random=False, save_rewards=False)

        call_kwargs = mock_sim.call_args.kwargs
        assert isinstance(call_kwargs["trained_model"], DefaultModel), (
            "Expected DefaultModel when no checkpoint is provided"
        )


# ---------------------------------------------------------------------------
# export_plots  argument parsing smoke
# ---------------------------------------------------------------------------


class TestExportPlots:
    def test_main_callable_and_module_importable(self):
        """analysis.export_plots is importable and main() is callable."""
        from bvr_marl_core.analysis.export_plots import main

        assert callable(main)

    def test_export_plots_requires_logdir_arg(self):
        """main() raises SystemExit if --logdir is not provided."""
        import sys

        from bvr_marl_core.analysis.export_plots import main

        with patch.object(sys, "argv", ["bvr-export-plots"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        # argparse exits with code 2 on missing required args
        assert exc_info.value.code == 2
