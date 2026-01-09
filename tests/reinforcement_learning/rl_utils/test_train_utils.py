import pytest
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf
import torch


class TestTrainUtils:
    """Test training utility functions."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration."""
        config = {
            "env": {
                "max_steps": 500,
                "aircraft_config": {
                    "agent_type": "F22",
                    "opponent_type": "F22"
                }
            },
            "model": {
                "rl_module_cls": "reinforcement_learning.networks.custom_multi_agent_model.CustomMultiAgentRLModule",
                "model_config": {
                    "hidden_layers": [64, 32],
                    "activation": "relu"
                }
            },
            "training": {
                "n_envs": 2,
                "rollout_fragment_length": 128,
                "learning_rate": 3e-4,
                "batch_size": 512,
                "clip_param": 0.2,
                "entropy_coef": 0.01,
                "vf_loss_coeff": 0.5,
                "num_epochs": 3,
                "sgd_minibatch_size": 32,
                "grad_clip": 0.5
            }
        }
        return OmegaConf.create(config)

    def test_discover_env_and_spaces(self, sample_config):
        """Test environment and space discovery."""
        try:
            from reinforcement_learning.rl_utils.train_utils import discover_env_and_spaces
            
            # Mock environment creator
            def mock_env_creator(config):
                mock_env = Mock()
                mock_env.reset.return_value = ({}, {})
                mock_env.observation_space = {"agent_A1": Mock(), "opponent_B1": Mock()}
                mock_env.action_space = {"agent_A1": Mock(), "opponent_B1": Mock()}
                return mock_env
            
            env, agent_ids, per_agent_obs, per_agent_act = discover_env_and_spaces(
                mock_env_creator, sample_config.env
            )
            
            assert env is not None
            assert isinstance(agent_ids, list)
            assert len(agent_ids) > 0
            assert isinstance(per_agent_obs, dict)
            assert isinstance(per_agent_act, dict)
            
        except ImportError:
            pytest.skip("discover_env_and_spaces not available")

    def test_load_module_class(self):
        """Test module class loading utility."""
        try:
            from reinforcement_learning.rl_utils.train_utils import load_module_class
            
            # Test loading a standard library class
            list_class = load_module_class("builtins.list")
            assert list_class is list
            
            # Test loading torch class
            linear_class = load_module_class("torch.nn.Linear")
            assert linear_class is torch.nn.Linear
            
        except ImportError:
            pytest.skip("load_module_class not available")

    def test_build_multi_agent_module_spec(self):
        """Test multi-agent module spec building."""
        try:
            from reinforcement_learning.rl_utils.train_utils import build_multi_agent_module_spec
            from gymnasium.spaces import Box
            
            # Mock spaces
            per_agent_obs = {
                "agent_A1": Box(low=-1, high=1, shape=(10,)),
                "opponent_B1": Box(low=-1, high=1, shape=(10,))
            }
            per_agent_act = {
                "agent_A1": Box(low=-1, high=1, shape=(9,)),
                "opponent_B1": Box(low=-1, high=1, shape=(9,))
            }
            
            with patch('reinforcement_learning.rl_utils.train_utils.load_module_class') as mock_load:
                mock_load.return_value = Mock()
                
                spec = build_multi_agent_module_spec(
                    "test.module.Class",
                    per_agent_obs,
                    per_agent_act,
                    {"hidden_dim": 64}
                )
                
                assert spec is not None
                
        except ImportError:
            pytest.skip("build_multi_agent_module_spec not available")

    def test_build_policies_dict(self):
        """Test policies dictionary building."""
        try:
            from reinforcement_learning.rl_utils.train_utils import build_policies_dict
            from gymnasium.spaces import Box
            
            agent_ids = ["agent_A1", "opponent_B1"]
            per_agent_obs = {
                "agent_A1": Box(low=-1, high=1, shape=(10,)),
                "opponent_B1": Box(low=-1, high=1, shape=(10,))
            }
            per_agent_act = {
                "agent_A1": Box(low=-1, high=1, shape=(9,)),
                "opponent_B1": Box(low=-1, high=1, shape=(9,))
            }
            
            policies = build_policies_dict(agent_ids, per_agent_obs, per_agent_act)
            
            assert isinstance(policies, dict)
            assert len(policies) == len(agent_ids)
            for agent_id in agent_ids:
                assert agent_id in policies
                
        except ImportError:
            pytest.skip("build_policies_dict not available")

    def test_api_stack_toggles(self, sample_config):
        """Test API stack configuration toggles."""
        try:
            from reinforcement_learning.rl_utils.train_utils import _apply_api_stack_toggles
            from ray.rllib.algorithms.ppo import PPOConfig
            
            config = PPOConfig()
            
            # Should not raise an error
            updated_config = _apply_api_stack_toggles(config)
            assert updated_config is not None
            
        except ImportError:
            pytest.skip("API stack toggles not available")

    def test_runner_toggles(self):
        """Test runner configuration toggles."""
        try:
            from reinforcement_learning.rl_utils.train_utils import _apply_runner_toggles
            from ray.rllib.algorithms.ppo import PPOConfig
            
            config = PPOConfig()
            
            # Test with different parameters
            updated_config = _apply_runner_toggles(config, n_envs=2, rfl=128)
            assert updated_config is not None
            
            # Test with 'auto' rollout fragment length
            updated_config2 = _apply_runner_toggles(config, n_envs=4, rfl='auto')
            assert updated_config2 is not None
            
        except ImportError:
            pytest.skip("Runner toggles not available")

    def test_build_training_config(self, sample_config):
        """Test complete training configuration building."""
        try:
            from reinforcement_learning.rl_utils.train_utils import build_training_config
            
            # Mock environment creator
            def mock_env_creator(config):
                mock_env = Mock()
                mock_env.reset.return_value = ({}, {})
                mock_env.observation_space = {"agent_A1": Mock(), "opponent_B1": Mock()}
                mock_env.action_space = {"agent_A1": Mock(), "opponent_B1": Mock()}
                return mock_env
            
            with patch('reinforcement_learning.rl_utils.train_utils.build_multi_agent_module_spec') as mock_build_spec:
                mock_build_spec.return_value = Mock()
                
                ppo_config = build_training_config(
                    mock_env_creator,
                    None,  # obs_space (legacy parameter)
                    None,  # act_space (legacy parameter)
                    sample_config
                )
                
                assert ppo_config is not None
                
        except ImportError:
            pytest.skip("build_training_config not available")

    def test_train_algorithm(self):
        """Test algorithm training loop."""
        try:
            from reinforcement_learning.rl_utils.train_utils import train_algorithm
            
            # Mock algorithm
            mock_algo = Mock()
            mock_algo.train.return_value = {"episode_reward_mean": 100.0}
            
            # Mock config
            mock_config = Mock()
            mock_config.training.steps = 3
            
            # Should complete without errors
            train_algorithm(mock_algo, mock_config)
            
            # Verify train was called
            assert mock_algo.train.call_count == 3
            
        except ImportError:
            pytest.skip("train_algorithm not available")

    def test_termination_check(self):
        """Test training termination condition checking."""
        try:
            from reinforcement_learning.rl_utils.train_utils import check_termination
            
            # Test with normal result
            result = {"episode_reward_mean": 100.0}
            config = Mock()
            
            # Should not terminate by default
            should_terminate = check_termination(result, config)
            assert isinstance(should_terminate, bool)
            
        except ImportError:
            pytest.skip("check_termination not available")