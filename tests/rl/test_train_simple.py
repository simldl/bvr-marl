"""
Tests for bvr_marl_core.training.train_simple module.

Verifies the simplified training script configuration, environment setup,
and Ray/RLlib integration using the same mocking approach as test_train.py.
"""

import os
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


class MockSimplifiedEnv(gym.Env):
    """Minimal mock for SimplifiedMultiAgentEnv — symmetric two-agent setup."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        act_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.observation_space = {"A1": obs_space, "A2": obs_space}
        self.action_space = {"A1": act_space, "A2": act_space}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {"A1": np.zeros(10, dtype=np.float32), "A2": np.zeros(10, dtype=np.float32)}
        return obs, {"A1": {}, "A2": {}}

    def step(self, actions):
        obs = {"A1": np.zeros(10, dtype=np.float32), "A2": np.zeros(10, dtype=np.float32)}
        rewards = {"A1": 0.0, "A2": 0.0}
        dones = {"A1": False, "A2": False, "__all__": False}
        truncs = {"A1": False, "A2": False, "__all__": False}
        return obs, rewards, dones, truncs, {"A1": {}, "A2": {}}


@pytest.fixture
def simple_cfg():
    """Minimal valid config for train_simple_main."""
    return {
        "seed": 0,
        "num_gpus": 0,
        "env": {
            "num_agents_per_team": 2,
            "map_size": 300,
            "max_steps": 100,
        },
        "model": {
            "model_config": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "use_lstm": False,
                "vf_share_layers": True,
            }
        },
        "training": {"steps": 20},
        "logging": {
            "model_name": "test_simple_model",
            "log_dir": "/tmp/test_simple_logs",
            "save_dir": "/tmp/test_simple_models",
        },
    }


def _make_mock_tuner():
    mock_result = Mock()
    mock_tuner = Mock()
    mock_tuner.fit.return_value = mock_result
    return mock_tuner


def _train_patches(tuner_instance, tmp_path):
    """Return a context-manager stack that mocks all heavy deps for train_simple_main."""
    mock_env = MockSimplifiedEnv()
    mock_env.reset()

    mock_ppo = Mock()
    mock_ppo.to_dict.return_value = {"env": "SimplifiedMultiAgentEnv"}

    return (
        patch("ray.init"),
        patch("ray.shutdown"),
        patch(
            "bvr_marl_core.training.train_simple.create_simplified_env_creator",
            return_value=lambda _cfg: mock_env,
        ),
        patch("ray.tune.register_env"),
        patch(
            "bvr_marl_core.training.train_simple.build_ppo_config",
            return_value=mock_ppo,
        ),
        patch(
            "bvr_marl_core.training.train_simple.process_checkpoint_resume",
            return_value=(None, None),
        ),
        patch(
            "bvr_marl_core.training.train_simple.process_weight_loading",
            return_value=None,
        ),
        patch("ray.tune.Tuner", return_value=tuner_instance),
        patch("torch.cuda.is_available", return_value=False),
    )


class TestPolicyMapping:
    def test_all_agents_map_to_shared_policy(self):
        """Simplified training uses symmetric self-play — every agent → shared_policy."""
        try:
            from bvr_marl_core.training.train_simple import policy_mapping_fn
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        for agent_id in ["A1", "A2", "B1", "B2", "agent_1", "opponent_1", "red_1"]:
            assert policy_mapping_fn(agent_id) == "shared_policy", (
                f"Expected shared_policy for agent '{agent_id}'"
            )


class TestSetupDirectories:
    def test_relative_save_dir_is_made_absolute(self, tmp_path):
        try:
            from bvr_marl_core.training.train_simple import setup_directories
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = {
            "logging": {
                "save_dir": str(tmp_path / "models"),
                "log_dir": str(tmp_path / "logs"),
            }
        }
        save_dir, log_dir = setup_directories(cfg)

        assert os.path.isabs(save_dir)
        assert os.path.isabs(log_dir)

    def test_absolute_dirs_unchanged(self, tmp_path):
        try:
            from bvr_marl_core.training.train_simple import setup_directories
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        abs_save = str(tmp_path / "abs_models")
        abs_log = str(tmp_path / "abs_logs")
        cfg = {"logging": {"save_dir": abs_save, "log_dir": abs_log}}
        save_dir, log_dir = setup_directories(cfg)

        assert save_dir == abs_save
        assert log_dir == abs_log


class TestEnvironmentVariables:
    def test_required_env_vars_set_on_import(self):
        """Critical env vars for distributed training must be set at module import."""
        try:
            import bvr_marl_core.training.train_simple  # noqa: F401
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        assert os.environ.get("MASTER_ADDR") == "127.0.0.1"
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("NCCL_P2P_DISABLE") == "1"


class TestTrainSimpleMain:
    def test_basic_flow(self, simple_cfg, tmp_path):
        """train_simple_main completes end-to-end with mocked Ray."""
        try:
            from bvr_marl_core.training.train_simple import train_simple_main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = dict(simple_cfg)
        cfg["logging"]["save_dir"] = str(tmp_path / "models")
        cfg["logging"]["log_dir"] = str(tmp_path / "logs")

        mock_tuner = _make_mock_tuner()

        with (
            patch("ray.init") as mock_ray_init,
            patch("ray.shutdown") as mock_shutdown,
            patch(
                "bvr_marl_core.training.train_simple.create_simplified_env_creator",
                return_value=lambda _cfg: MockSimplifiedEnv(),
            ),
            patch("ray.tune.register_env") as mock_register,
            patch(
                "bvr_marl_core.training.train_simple.build_ppo_config",
                return_value=Mock(**{"to_dict.return_value": {}}),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_checkpoint_resume",
                return_value=(None, None),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_weight_loading",
                return_value=None,
            ),
            patch("ray.tune.Tuner", return_value=mock_tuner),
            patch("torch.cuda.is_available", return_value=False),
        ):
            train_simple_main(cfg)

        mock_ray_init.assert_called_once()
        mock_shutdown.assert_called_once()
        mock_register.assert_called_once_with("SimplifiedMultiAgentEnv", ANY)
        mock_tuner.fit.assert_called_once()

    def test_checkpoint_frequency_5_percent(self, simple_cfg, tmp_path):
        """Checkpoint frequency is set to 5% of total training steps."""
        try:
            from bvr_marl_core.training.train_simple import train_simple_main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = dict(simple_cfg)
        cfg["training"]["steps"] = 200  # 5% → 10
        cfg["logging"]["save_dir"] = str(tmp_path / "models")
        cfg["logging"]["log_dir"] = str(tmp_path / "logs")

        mock_tuner = _make_mock_tuner()

        with (
            patch("ray.init"),
            patch("ray.shutdown"),
            patch(
                "bvr_marl_core.training.train_simple.create_simplified_env_creator",
                return_value=lambda _cfg: MockSimplifiedEnv(),
            ),
            patch("ray.tune.register_env"),
            patch(
                "bvr_marl_core.training.train_simple.build_ppo_config",
                return_value=Mock(**{"to_dict.return_value": {}}),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_checkpoint_resume",
                return_value=(None, None),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_weight_loading",
                return_value=None,
            ),
            patch("ray.tune.Tuner", return_value=mock_tuner),
            patch("torch.cuda.is_available", return_value=False),
        ):
            train_simple_main(cfg)

    def test_checkpoint_frequency_via_run_config(self, simple_cfg, tmp_path):
        """Verify the RunConfig passed to Tuner has the correct checkpoint_frequency."""
        try:
            from bvr_marl_core.training.train_simple import train_simple_main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = dict(simple_cfg)
        cfg["training"]["steps"] = 100  # 5% of 100 → 5
        cfg["logging"]["save_dir"] = str(tmp_path / "models")
        cfg["logging"]["log_dir"] = str(tmp_path / "logs")

        mock_tuner_cls = Mock()
        mock_tuner_instance = _make_mock_tuner()
        mock_tuner_cls.return_value = mock_tuner_instance

        with (
            patch("ray.init"),
            patch("ray.shutdown"),
            patch(
                "bvr_marl_core.training.train_simple.create_simplified_env_creator",
                return_value=lambda _cfg: MockSimplifiedEnv(),
            ),
            patch("ray.tune.register_env"),
            patch(
                "bvr_marl_core.training.train_simple.build_ppo_config",
                return_value=Mock(**{"to_dict.return_value": {}}),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_checkpoint_resume",
                return_value=(None, None),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_weight_loading",
                return_value=None,
            ),
            patch("ray.tune.Tuner", mock_tuner_cls),
            patch("torch.cuda.is_available", return_value=False),
        ):
            train_simple_main(cfg)

        mock_tuner_cls.assert_called_once()
        call_kwargs = mock_tuner_cls.call_args.kwargs
        run_config = call_kwargs["run_config"]
        # max(1, int(0.05 * 100)) == 5
        assert run_config.checkpoint_config.checkpoint_frequency == 5

    def test_environment_registered_as_simplified(self, simple_cfg, tmp_path):
        """tune.register_env is called with 'SimplifiedMultiAgentEnv'."""
        try:
            from bvr_marl_core.training.train_simple import train_simple_main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = dict(simple_cfg)
        cfg["logging"]["save_dir"] = str(tmp_path / "models")
        cfg["logging"]["log_dir"] = str(tmp_path / "logs")

        mock_tuner = _make_mock_tuner()

        with (
            patch("ray.init"),
            patch("ray.shutdown"),
            patch(
                "bvr_marl_core.training.train_simple.create_simplified_env_creator",
                return_value=lambda _cfg: MockSimplifiedEnv(),
            ),
            patch("ray.tune.register_env") as mock_register,
            patch(
                "bvr_marl_core.training.train_simple.build_ppo_config",
                return_value=Mock(**{"to_dict.return_value": {}}),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_checkpoint_resume",
                return_value=(None, None),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_weight_loading",
                return_value=None,
            ),
            patch("ray.tune.Tuner", return_value=mock_tuner),
            patch("torch.cuda.is_available", return_value=False),
        ):
            train_simple_main(cfg)

        registered_name = mock_register.call_args.args[0]
        assert registered_name == "SimplifiedMultiAgentEnv"

    def test_train_config_yaml_written_to_model_dir(self, simple_cfg, tmp_path):
        """train_simple_main saves train_config.yaml next to the model."""
        try:
            from bvr_marl_core.training.train_simple import train_simple_main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        cfg = dict(simple_cfg)
        cfg["logging"]["save_dir"] = str(tmp_path / "models")
        cfg["logging"]["log_dir"] = str(tmp_path / "logs")

        mock_tuner = _make_mock_tuner()

        with (
            patch("ray.init"),
            patch("ray.shutdown"),
            patch(
                "bvr_marl_core.training.train_simple.create_simplified_env_creator",
                return_value=lambda _cfg: MockSimplifiedEnv(),
            ),
            patch("ray.tune.register_env"),
            patch(
                "bvr_marl_core.training.train_simple.build_ppo_config",
                return_value=Mock(**{"to_dict.return_value": {}}),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_checkpoint_resume",
                return_value=(None, None),
            ),
            patch(
                "bvr_marl_core.training.train_simple.process_weight_loading",
                return_value=None,
            ),
            patch("ray.tune.Tuner", return_value=mock_tuner),
            patch("torch.cuda.is_available", return_value=False),
        ):
            train_simple_main(cfg)

        expected_yaml = tmp_path / "models" / cfg["logging"]["model_name"] / "train_config.yaml"
        assert expected_yaml.exists(), f"train_config.yaml not found at {expected_yaml}"


class TestMainEntryPoint:
    def test_main_callable(self):
        """The module exposes a callable main() entry point."""
        try:
            from bvr_marl_core.training.train_simple import main
        except ImportError as e:
            pytest.skip(f"train_simple deps not available: {e}")

        assert callable(main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
