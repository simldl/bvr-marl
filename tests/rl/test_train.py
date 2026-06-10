"""
Tests for bvr_marl_core/training/train.py module.

Tests the training script configuration, environment setup, and Ray/RLlib integration.
Uses default RLlib PPO — no custom neural network.
"""

import os
import platform
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces


class MockMultiAgentEnv(gym.Env):
    """Mock multi-agent environment for testing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        mock_obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        mock_act_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = {"A1": mock_obs_space, "B1": mock_obs_space}
        self.action_space = {"A1": mock_act_space, "B1": mock_act_space}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {"A1": np.zeros(10, dtype=np.float32), "B1": np.zeros(10, dtype=np.float32)}
        return obs, {"A1": {}, "B1": {}}

    def step(self, actions):
        obs = {"A1": np.zeros(10, dtype=np.float32), "B1": np.zeros(10, dtype=np.float32)}
        return (
            obs,
            {"A1": 0.0, "B1": 0.0},
            {"A1": False, "B1": False, "__all__": False},
            {"A1": False, "B1": False, "__all__": False},
            {"A1": {}, "B1": {}},
        )


@pytest.fixture
def sample_config():
    return {
        "seed": 42,
        "num_gpus": 0,
        "training": {"steps": 10},
        "logging": {
            "model_name": "test_model",
            "log_dir": "/tmp/test_logs",
            "save_dir": "/tmp/test_models",
        },
        "env": {
            "max_steps": 100,
            "num_agents_per_team": 1,
            "aircraft_config": {"agent_type": "F22", "opponent_type": "Eurofighter"},
        },
        "ppo": {
            "gamma": 0.995,
            "lambda": 0.95,
            "train_batch_size_per_learner": 512,
            "num_epochs": 10,
            "lr": 3e-4,
            "num_env_runners": 0,
        },
    }


class TestTrainModule:
    def test_environment_variables_set(self):
        """Critical env vars are set on module import."""
        try:
            import bvr_marl_core.training.train  # noqa: F401

            assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
            assert os.environ.get("OMP_NUM_THREADS") == "1"
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_main_basic_flow(self, sample_config):
        """train_main runs without error with mocked Ray."""
        try:
            from bvr_marl_core.training.train import train_main

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "bvr_marl_core.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = Mock()
                mock_tuner.return_value = mock_tuner_instance

                train_main(sample_config)

                mock_tuner.assert_called_once()
                mock_tuner_instance.fit.assert_called_once()
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_checkpoint_frequency_calculation(self, sample_config):
        """Checkpoint frequency = max(1, int(0.05 * steps))."""
        try:
            from bvr_marl_core.training.train import train_main

            cfg = dict(sample_config)
            cfg["training"] = {"steps": 100}

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "bvr_marl_core.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = Mock()
                mock_tuner.return_value = mock_tuner_instance

                train_main(cfg)

                run_config = mock_tuner.call_args[1]["run_config"]
                assert run_config.checkpoint_config.checkpoint_frequency == 5
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
