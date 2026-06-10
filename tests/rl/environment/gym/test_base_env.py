from unittest.mock import Mock

import pytest

from bvr_marl_core.rl.environment.gym.base_env import BaseMultiAgentEnv


class TestBaseMultiAgentEnv:
    """Test the base multi-agent environment class."""

    def test_base_env_initialization(self):
        """Test base environment initialization."""
        config = {"simulator": Mock()}
        env = BaseMultiAgentEnv(config)

        assert env.config == config
        assert env.simulator == config["simulator"]

    def test_base_env_setters(self):
        """Test setter methods for environment components."""
        config = {"simulator": Mock()}
        env = BaseMultiAgentEnv(config)

        # Mock space managers
        obs_space_mgr = Mock()
        mock_obs_space = Mock()
        obs_space_mgr.all.return_value = {"A1": mock_obs_space}
        action_space_mgr = Mock()
        mock_action_space = Mock()
        action_space_mgr.all.return_value = {"A1": mock_action_space}

        # Test setters
        env.set_spaces(obs_space_mgr, action_space_mgr)
        env.set_obs_builder("obs_builder")
        env.set_action_processor("action_processor")
        env.set_reward_calculator("reward_calculator")

        # Test that the spaces were set correctly
        assert "A1" in env.observation_space
        assert env.observation_space["A1"] == mock_obs_space
        assert env.obs_builder == "obs_builder"
        assert env.action_processor == "action_processor"
        assert env.reward_calculator == "reward_calculator"

    def test_base_env_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        config = {"simulator": Mock()}
        env = BaseMultiAgentEnv(config)

        with pytest.raises(NotImplementedError):
            env.reset()

        with pytest.raises(NotImplementedError):
            env.step({})

        with pytest.raises(NotImplementedError):
            env._spawn_agents()

    def test_base_env_abstract_nature(self):
        """Test that BaseMultiAgentEnv is abstract and requires implementation."""
        # The base class should be instantiable but methods should raise NotImplementedError
        config = {"simulator": Mock()}
        env = BaseMultiAgentEnv(config)

        # Check that it has the expected abstract methods
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "_spawn_agents")

        # But they should not be implemented
        assert callable(env.reset)
        assert callable(env.step)
        assert callable(env._spawn_agents)
