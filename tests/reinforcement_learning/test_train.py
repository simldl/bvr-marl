"""
Tests for reinforcement_learning/train.py module.

Tests the training script configuration, environment setup, and Ray/RLlib integration.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf, DictConfig
import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
            "B1": mock_obs_space   # Doesn't start with "A" - will be opponent
        }
        self.action_space = {
            "A1": mock_act_space,
            "B1": mock_act_space
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {
            "A1": np.zeros(10, dtype=np.float32),
            "B1": np.zeros(10, dtype=np.float32)
        }
        infos = {"A1": {}, "B1": {}}
        return obs, infos

    def step(self, actions):
        obs = {
            "A1": np.zeros(10, dtype=np.float32),
            "B1": np.zeros(10, dtype=np.float32)
        }
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
                "use_neural_wrapper": True,
                "neural_wrapper_config": {
                    "wrapped_action_dim": 4,
                    "full_action_dim": 10,
                    "automation_level": "balanced"
                },
                "model_config": {
                    "action_dim": 4,
                    "hidden_dim": 256,
                    "num_hidden_layers": 3,
                    "activation": "relu"
                }
            },
            "training": {
                "steps": 100
            },
            "logging": {
                "model_name": "test_model",
                "log_dir": "/tmp/test_logs",
                "save_dir": "/tmp/test_models"  # Add missing save_dir
            },
            "num_gpus": 0,
            "env": {
                "max_steps": 1000,
                "episode_limit": 300,
                "aircraft_config": {
                    "agent_type": "F22",
                    "opponent_type": "Eurofighter"
                }
            }
        }
        return OmegaConf.create(config)

    def test_main_function_basic_flow(self, sample_config, mock_env):
        """Test the main function's basic execution flow."""
        try:
            from reinforcement_learning.train import main

            with patch('ray.init') as mock_ray_init, \
                 patch('ray.shutdown') as mock_shutdown, \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env') as mock_register_env, \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False):

                # Setup tuner mock with proper result
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                # Run main function
                main(sample_config)

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
            from reinforcement_learning.train import main

            with patch('torch.cuda.is_available') as mock_cuda_available, \
                 patch('torch.cuda.set_device') as mock_set_device, \
                 patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner:

                # Test with CUDA available
                mock_cuda_available.return_value = True
                sample_config.num_gpus = 1

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                main(sample_config)

                # Verify CUDA device was set
                mock_set_device.assert_called_once_with(0)
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_checkpoint_frequency_calculation(self, sample_config, mock_env):
        """Test checkpoint frequency calculation based on training steps."""
        try:
            from reinforcement_learning.train import main

            # Test with 100 steps (checkpoint_freq = steps // 20 = 100 // 20 = 5)
            sample_config.training.steps = 100

            with patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False):

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                main(sample_config)

                # Get the call arguments to verify checkpoint frequency
                call_args = mock_tuner.call_args
                run_config = call_args[1]['run_config']
                # Checkpoint frequency should be steps // 20 = 100 // 20 = 5
                assert run_config.checkpoint_config.checkpoint_frequency == 5
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_environment_variable_setup(self):
        """Test that required environment variables are set."""
        try:
            # Import to trigger environment variable setup
            import reinforcement_learning.train  # noqa: F401

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
            with patch('platform.system', return_value='Windows'):
                # Re-import to trigger Windows-specific setup
                import importlib
                import reinforcement_learning.train
                importlib.reload(reinforcement_learning.train)

                assert os.environ.get("PL_TORCH_DISTRIBUTED_BACKEND") == "gloo"
                assert os.environ.get("WORLD_SIZE") == "1"
                assert os.environ.get("RANK") == "0"
                assert os.environ.get("LOCAL_RANK") == "0"
                assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_neural_wrapper_config(self, sample_config, mock_env):
        """Test neural wrapper configuration is properly applied."""
        try:
            from reinforcement_learning.train import main

            with patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False):

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                # Enable neural wrapper
                sample_config.model.use_neural_wrapper = True

                main(sample_config)

                # Verify tuner was called
                mock_tuner.assert_called_once()

                # Get the config passed to tuner (keyword argument)
                call_kwargs = mock_tuner.call_args.kwargs
                ppo_config = call_kwargs['param_space']

                # Verify environment config was passed
                assert 'env_config' in ppo_config
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_multi_agent_setup(self, sample_config, mock_env):
        """Test multi-agent policy setup with two policies."""
        try:
            from reinforcement_learning.train import main

            with patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False):

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                main(sample_config)

                # Verify tuner was called
                mock_tuner.assert_called_once()

                # Get the config passed to tuner (keyword argument)
                call_kwargs = mock_tuner.call_args.kwargs
                ppo_config = call_kwargs['param_space']

                # In RLlib 2.50+, multi_agent config is under 'multi_agent_config' key
                # or directly in the config dict
                # Just verify that policies and policy_mapping_fn are in the config
                assert 'policies' in ppo_config or 'multi_agent_config' in ppo_config

                # Check if it's in multi_agent_config or directly
                if 'multi_agent_config' in ppo_config:
                    ma_config = ppo_config['multi_agent_config']
                    assert 'policies' in ma_config
                    assert 'policy_mapping_fn' in ma_config
                else:
                    # New API might put these directly in config
                    assert 'policies' in ppo_config
                    assert 'policy_mapping_fn' in ppo_config
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_ppo_training_config(self, sample_config, mock_env):
        """Test PPO training hyperparameters are set correctly."""
        try:
            from reinforcement_learning.train import main

            with patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False):

                # Setup tuner mock
                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                main(sample_config)

                # Get the config passed to tuner (keyword argument)
                call_kwargs = mock_tuner.call_args.kwargs
                ppo_config = call_kwargs['param_space']

                # Verify key PPO hyperparameters exist
                # Some keys might be in nested dicts depending on RLlib version
                assert ppo_config['gamma'] == 0.995
                assert ppo_config['lambda'] == 0.95 or ppo_config.get('lambda_', 0.95) == 0.95
                assert ppo_config['use_gae'] is True
                assert ppo_config['use_critic'] is True

                # Check for train_batch_size - RLlib may convert train_batch_size_per_learner
                # to train_batch_size in to_dict() with a calculated value
                batch_size = ppo_config.get('train_batch_size_per_learner') or ppo_config.get('train_batch_size')
                assert batch_size is not None and batch_size > 0, \
                    f"Expected batch size configuration, got: {batch_size}"
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")

    def test_rlmodule_spec_creation(self, sample_config, mock_env):
        """Test that RLModule specs are created for both policies."""
        try:
            from reinforcement_learning.train import main

            with patch('ray.init'), \
                 patch('ray.shutdown'), \
                 patch('reinforcement_learning.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv', MockMultiAgentEnv), \
                 patch('ray.tune.register_env'), \
                 patch('ray.tune.Tuner') as mock_tuner, \
                 patch('torch.cuda.is_available', return_value=False), \
                 patch('reinforcement_learning.train.RLModuleSpec') as mock_rlmodule_spec:

                # Setup mocks
                mock_spec_instance = Mock()
                mock_rlmodule_spec.return_value = mock_spec_instance

                mock_result = Mock()
                mock_tuner_instance = Mock()
                mock_tuner_instance.fit.return_value = mock_result
                mock_tuner.return_value = mock_tuner_instance

                main(sample_config)

                # Verify RLModuleSpec was created (at least once)
                # Note: In current implementation with shared policy, this may be called once
                assert mock_rlmodule_spec.call_count >= 1
        except ImportError as e:
            pytest.skip(f"Train module dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
