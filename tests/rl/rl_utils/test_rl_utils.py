from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf

from air_to_air_rl.rl.utils.reward_wrapper import RewardNormalizationWrapper


class MockMultiAgentEnv(gym.Env):
    """Mock multi-agent environment that properly inherits from gym.Env."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create multi-agent spaces
        self.observation_space = {
            "agent_A1": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
            "opponent_B1": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        }
        self.action_space = {
            "agent_A1": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            "opponent_B1": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {
            "agent_A1": np.zeros(10, dtype=np.float32),
            "opponent_B1": np.zeros(10, dtype=np.float32),
        }
        infos = {"agent_A1": {}, "opponent_B1": {}}
        return obs, infos

    def step(self, actions):
        obs = {
            "agent_A1": np.zeros(10, dtype=np.float32),
            "opponent_B1": np.zeros(10, dtype=np.float32),
        }
        rewards = {"agent_A1": 0.0, "opponent_B1": 0.0}
        terminateds = {"agent_A1": False, "opponent_B1": False, "__all__": False}
        truncateds = {"agent_A1": False, "opponent_B1": False, "__all__": False}
        infos = {"agent_A1": {}, "opponent_B1": {}}
        return obs, rewards, terminateds, truncateds, infos


class TestRLUtils:
    """Test RL utility functions."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration."""
        config = {
            "env": {
                "aircraft_config": {"agent_type": "F22", "opponent_type": "Eurofighter"},
                "weapon_config": {"missile_types": ["AIM120_AMRAAM"]},
            },
            "model": {
                "custom_model": "test_model",
                "rl_module_cls": "test.module.TestModel",
                "model_config": {"hidden_dim": 64},
            },
            "training": {"learning_rate": 3e-4, "batch_size": 512},
        }
        return OmegaConf.create(config)

    def test_create_env_creator(self, sample_config):
        """Test environment creator factory function."""
        try:
            from air_to_air_rl.rl.utils.env_creator import create_env_creator

            with patch(
                "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                MockMultiAgentEnv,
            ):
                # Create env creator
                env_creator = create_env_creator(sample_config)

                # Test that it returns a callable
                assert callable(env_creator)

                # Test calling the env creator
                env_config = {"test_param": "test_value"}
                env_instance = env_creator(env_config)

                # Should have created a wrapped environment (RewardNormalizationWrapper)
                assert env_instance is not None
                assert isinstance(env_instance, RewardNormalizationWrapper)

                # The wrapped environment should be our MockMultiAgentEnv
                assert isinstance(env_instance.env, MockMultiAgentEnv)

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_aircraft_config_resolution(self, sample_config):
        """Test aircraft configuration resolution from strings to classes."""
        try:
            from air_to_air_rl.rl.utils.type_maps import (
                resolve_aircraft_config as _resolve_aircraft_config,
            )

            resolved_config = _resolve_aircraft_config(sample_config.env)

            assert "aircraft_types" in resolved_config
            assert "agent" in resolved_config["aircraft_types"]
            assert "opponent" in resolved_config["aircraft_types"]

            # Should resolve to actual classes
            agent_class = resolved_config["aircraft_types"]["agent"]
            opponent_class = resolved_config["aircraft_types"]["opponent"]

            assert callable(agent_class)  # Should be a class
            assert callable(opponent_class)

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_aircraft_type_validation(self):
        """Test aircraft type validation."""
        try:
            from air_to_air_rl.rl.utils.type_maps import (
                resolve_aircraft_config as _resolve_aircraft_config,
            )

            # Test with invalid aircraft type
            invalid_config = {
                "aircraft_config": {"agent_type": "NonexistentAircraft", "opponent_type": "F22"}
            }

            with pytest.raises(ValueError):
                _resolve_aircraft_config(invalid_config)

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_aircraft_type_mapping(self):
        """Test aircraft type mapping functionality."""
        try:
            from air_to_air_rl.rl.utils.type_maps import AIRCRAFT_TYPE_MAP

            # Should contain standard aircraft types
            assert "F22" in AIRCRAFT_TYPE_MAP
            assert "Eurofighter" in AIRCRAFT_TYPE_MAP
            assert "F35" in AIRCRAFT_TYPE_MAP
            assert "DebugPlane" in AIRCRAFT_TYPE_MAP

            # All values should be callable (classes)
            for aircraft_class in AIRCRAFT_TYPE_MAP.values():
                assert callable(aircraft_class)

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missile_type_mapping(self):
        """Test missile type mapping functionality."""
        try:
            from air_to_air_rl.rl.utils.type_maps import MISSILE_TYPE_MAP

            # Should contain missile types
            assert "AIM120_AMRAAM" in MISSILE_TYPE_MAP
            assert "DefaultMissile" in MISSILE_TYPE_MAP

            # All values should be callable (classes)
            for missile_class in MISSILE_TYPE_MAP.values():
                assert callable(missile_class)

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_simulator_integration(self, sample_config):
        """Test simulator integration in environment creation."""
        try:
            from air_to_air_rl.rl.utils.env_creator import create_env_creator

            with (
                patch(
                    "air_to_air_rl.rl.environment.gym.bvr_multi_agent_env.BVRMultiAgentEnv",
                    MockMultiAgentEnv,
                ),
                patch("air_to_air_rl.rl.utils.env_creator.Simulator") as mock_sim_class,
            ):
                mock_sim_instance = Mock()
                mock_sim_class.return_value = mock_sim_instance

                # Create config without simulator to ensure it gets created
                test_config = sample_config.copy()

                env_creator = create_env_creator(test_config)

                # Create environment without explicit simulator in env_config
                env_instance = env_creator({})

                # Check that the environment was created successfully
                assert env_instance is not None
                assert isinstance(env_instance, RewardNormalizationWrapper)

                # Check that simulator was created (since not provided in config)
                mock_sim_class.assert_called_once()

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")
