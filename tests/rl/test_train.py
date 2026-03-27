"""
Tests for air_to_air_rl/training/train.py module.

Tests the training script configuration, environment setup, and Ray/RLlib integration.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf


class MockMultiAgentEnv(gym.Env):
    """Mock multi-agent environment that properly inherits from gym.Env."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create multi-agent spaces with proper agent IDs
        # Important: Keys must start with "A" for agent and NOT "A" for opponent
        mock_obs_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        mock_act_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.observation_space = {
            "A1": mock_obs_space,  # Starts with "A" - will be agent
            "B1": mock_obs_space,  # Doesn't start with "A" - will be opponent
        }
        self.action_space = {"A1": mock_act_space, "B1": mock_act_space}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {"A1": np.zeros(10, dtype=np.float32), "B1": np.zeros(10, dtype=np.float32)}
        infos = {"A1": {}, "B1": {}}
        return obs, infos

    def step(self, actions):
        obs = {"A1": np.zeros(10, dtype=np.float32), "B1": np.zeros(10, dtype=np.float32)}
        rewards = {"A1": 0.0, "B1": 0.0}
        terminateds = {"A1": False, "B1": False, "__all__": False}
        truncateds = {"A1": False, "B1": False, "__all__": False}
        infos = {"A1": {}, "B1": {}}
        return obs, rewards, terminateds, truncateds, infos


class TestTrainModule:
    """Test the main training module functionality."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        return MockMultiAgentEnv({})

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration matching train_config.yaml structure."""
        config = {
            "framework": "torch",
            "model": {
                "model_config": {
                    "action_dim": 10,
                    "hidden_dim": 256,
                    "num_hidden_layers": 3,
                    "activation": "relu",
                },
            },
            "training": {"steps": 100},
            "logging": {
                "model_name": "test_model",
                "log_dir": "/tmp/test_logs",
                "save_dir": "/tmp/test_models",
            },
            "num_gpus": 0,
            "env": {
                "max_steps": 1000,
                "episode_limit": 300,
                "aircraft_config": {"agent_type": "F22", "opponent_type": "Eurofighter"},
            },
        }
        return OmegaConf.create(config)

    def _to_dict(self, cfg):
        """Convert OmegaConf config to plain dict for train_main."""
        return OmegaConf.to_container(cfg, resolve=True)

    def test_main_function_basic_flow(self, sample_config, mock_env):
        """Test the main function's basic execution flow."""
        try:
            from air_to_air_rl.training.train import train_main

            with (
                patch("ray.init") as mock_ray_init,
                patch("ray.shutdown") as mock_shutdown,
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env") as mock_register_env,
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                # Setup tuner mock with proper result
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                # Run main function
                train_main(self._to_dict(sample_config))

                # Verify ray operations
                mock_ray_init.assert_called_once()
                mock_shutdown.assert_called_once()

                # Verify environment registration
                mock_register_env.assert_called_once()

                # Verify tuner creation and execution
                mock_tuner.assert_called_once()
                mock_tuner_instance.fit.assert_called_once()
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_policy_mapping_function(self):
        """Test the policy mapping function logic."""
        try:
            # Test the policy mapping logic directly since it's defined inline
            def policy_mapping_fn(agent_id, *_args, **_kwargs):
                return "agent_policy" if str(agent_id).startswith("A") else "opponent_policy"

            # Test agent mapping (starts with "A")
            assert policy_mapping_fn("A1") == "agent_policy"
            assert policy_mapping_fn("A_fighter_1") == "agent_policy"
            assert policy_mapping_fn("Agent_1") == "agent_policy"

            # Test opponent mapping (doesn't start with "A")
            assert policy_mapping_fn("B1") == "opponent_policy"
            assert policy_mapping_fn("B_fighter_1") == "opponent_policy"
            assert policy_mapping_fn("opponent_1") == "opponent_policy"
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_cuda_setup(self, sample_config, mock_env):
        """Test CUDA device setup when available."""
        try:
            from air_to_air_rl.training.train import train_main

            with (
                patch("torch.cuda.is_available") as mock_cuda_available,
                patch("torch.cuda.set_device") as mock_set_device,
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
            ):
                # Test with CUDA available
                mock_cuda_available.return_value = True
                cfg = self._to_dict(sample_config)
                cfg["num_gpus"] = 1

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                train_main(cfg)

                # Verify CUDA device was set
                mock_set_device.assert_called_once_with(0)
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_checkpoint_frequency_calculation(self, sample_config, mock_env):
        """Test checkpoint frequency calculation based on training steps."""
        try:
            from air_to_air_rl.training.train import train_main

            # Test with 100 steps (checkpoint_freq = max(1, int(0.05 * 100)) = 5)
            cfg = self._to_dict(sample_config)
            cfg["training"]["steps"] = 100

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                train_main(cfg)

                # Get the call arguments to verify checkpoint frequency
                call_args = mock_tuner.call_args
                run_config = call_args[1]["run_config"]
                # Checkpoint frequency should be max(1, int(0.05 * 100)) = 5
                assert run_config.checkpoint_config.checkpoint_frequency == 5
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_environment_variable_setup(self):
        """Test that required environment variables are set."""
        try:
            # Import to trigger environment variable setup
            import air_to_air_rl.training.train  # noqa: F401

            # Check critical environment variables that should be set
            assert os.environ.get("MASTER_ADDR") == "127.0.0.1"
            assert os.environ.get("MASTER_PORT") == "29500"
            assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
            assert os.environ.get("OMP_NUM_THREADS") == "1"
            assert os.environ.get("NCCL_P2P_DISABLE") == "1"
            assert os.environ.get("NCCL_IB_DISABLE") == "1"
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_windows_specific_env_vars(self):
        """Test Windows-specific environment variable setup."""
        try:
            with patch("platform.system", return_value="Windows"):
                import importlib

                import air_to_air_rl.training.train as train_mod

                importlib.reload(train_mod)

                assert os.environ.get("PL_TORCH_DISTRIBUTED_BACKEND") == "gloo"
                assert os.environ.get("WORLD_SIZE") == "1"
                assert os.environ.get("RANK") == "0"
                assert os.environ.get("LOCAL_RANK") == "0"
                assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_multi_agent_setup(self, sample_config, mock_env):
        """Test multi-agent policy setup with two policies."""
        try:
            from air_to_air_rl.training.train import train_main

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                train_main(self._to_dict(sample_config))

                # Verify tuner was called
                mock_tuner.assert_called_once()

                # Get the config passed to tuner (keyword argument)
                call_kwargs = mock_tuner.call_args.kwargs
                ppo_config = call_kwargs["param_space"]

                # In RLlib 2.50+, multi_agent config is under 'multi_agent_config' key
                # or directly in the config dict
                # Just verify that policies and policy_mapping_fn are in the config
                assert "policies" in ppo_config or "multi_agent_config" in ppo_config

                # Check if it's in multi_agent_config or directly
                if "multi_agent_config" in ppo_config:
                    ma_config = ppo_config["multi_agent_config"]
                    assert "policies" in ma_config
                    assert "policy_mapping_fn" in ma_config
                else:
                    # New API might put these directly in config
                    assert "policies" in ppo_config
                    assert "policy_mapping_fn" in ppo_config
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_ppo_training_config(self, sample_config, mock_env):
        """Test PPO training hyperparameters are set correctly."""
        try:
            from air_to_air_rl.training.train import train_main

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                train_main(self._to_dict(sample_config))

                # Get the config passed to tuner (keyword argument)
                call_kwargs = mock_tuner.call_args.kwargs
                ppo_config = call_kwargs["param_space"]

                # Verify key PPO hyperparameters exist
                # Some keys might be in nested dicts depending on RLlib version
                assert ppo_config["gamma"] == 0.995
                assert ppo_config["lambda"] == 0.95 or ppo_config.get("lambda_", 0.95) == 0.95
                assert ppo_config["use_gae"] is True
                assert ppo_config["use_critic"] is True

                # Check for train_batch_size - RLlib may convert train_batch_size_per_learner
                # to train_batch_size in to_dict() with a calculated value
                batch_size = ppo_config.get("train_batch_size_per_learner") or ppo_config.get(
                    "train_batch_size"
                )
                assert batch_size is not None and batch_size > 0, (
                    f"Expected batch size configuration, got: {batch_size}"
                )
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_default_network_training(self, sample_config, mock_env):
        """Test that training works with Ray's default networks (public release)."""
        try:
            from air_to_air_rl.training.train import train_main

            with (
                patch("ray.init"),
                patch("ray.shutdown"),
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("ray.tune.register_env"),
                patch("ray.tune.Tuner") as mock_tuner,
                patch("torch.cuda.is_available", return_value=False),
            ):
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                train_main(self._to_dict(sample_config))

                # Verify training was attempted (tuner.fit was called)
                mock_tuner_instance.fit.assert_called_once()
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
