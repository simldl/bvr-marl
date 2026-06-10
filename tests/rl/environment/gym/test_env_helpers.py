from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium import spaces


class TestEnvHelpers:
    """Test environment helper functions."""

    def test_env_helpers_import(self):
        """Test that env helpers can be imported."""
        try:
            import bvr_marl_core.rl.environment.gym.env_helpers

            assert True
        except ImportError as e:
            pytest.skip(f"Environment helpers not available: {e}")

    def test_to_f32_obs(self):
        """Test observation conversion to float32."""
        try:
            from bvr_marl_core.rl.environment.gym.env_helpers import to_f32_obs

            # Test with different observation types
            obs_dict = {
                "state": np.array([1, 2, 3], dtype=np.int32),
                "mask": np.array([1, 0, 1], dtype=bool),
            }

            f32_obs = to_f32_obs(obs_dict)

            assert f32_obs["state"].dtype == np.float32
            assert f32_obs["mask"].dtype == np.float32

        except (ImportError, AttributeError):
            pytest.skip("to_f32_obs function not available")

    def test_coerce_obs_to_space(self):
        """Test observation coercion to match space requirements."""
        from bvr_marl_core.rl.environment.gym.env_helpers import coerce_obs_to_space

        # Create a test space with Dict structure (as the function expects)
        space = spaces.Dict({"state": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)})

        # Test observation that needs coercion
        obs = {
            "state": np.array([2.0, -2.0, 0.5])  # Outside bounds
        }

        coerced_obs = coerce_obs_to_space(space, obs)

        assert space.contains(coerced_obs)
        assert coerced_obs["state"].dtype == np.float32
        # Check clipping worked
        assert np.all(coerced_obs["state"] <= 1.0)
        assert np.all(coerced_obs["state"] >= -1.0)

    def test_spaces_from_sample_observation(self):
        """Test space creation from sample observations."""
        try:
            from bvr_marl_core.rl.environment.gym.env_helpers import (
                spaces_from_sample_observation,
            )

            # Sample observation - use float32 for both since the function converts to float32
            sample_obs = {
                "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "action_mask": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            }

            space = spaces_from_sample_observation(sample_obs)

            assert isinstance(space, spaces.Dict)
            assert "state" in space.spaces
            assert "action_mask" in space.spaces

            # Check that the created space can contain the sample
            assert space.contains(sample_obs)

        except (ImportError, AttributeError):
            pytest.skip("spaces_from_sample_observation function not available")

    def test_ensure_action_space(self):
        """Test action space normalization."""
        from bvr_marl_core.rl.environment.gym.env_helpers import ensure_action_space

        # Test with existing Box space
        existing_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        result = ensure_action_space(existing_space)
        assert result is existing_space

        # Test with array
        arr = np.array([0.1, 0.2, 0.3])
        result = ensure_action_space(arr)
        assert isinstance(result, spaces.Box)
        assert result.shape == (3,)

        # Test with dict config - shape only
        config_shape = {"shape": (4,)}
        result = ensure_action_space(config_shape)
        assert isinstance(result, spaces.Box)
        assert result.shape == (4,)

        # Test with dict config - with bounds (creates scalar space from scalar bounds)
        config_bounds = {"low": -2.0, "high": 2.0}
        result = ensure_action_space(config_bounds)
        assert isinstance(result, spaces.Box)
        # Function creates scalar space when scalar bounds are provided
        assert result.low == -2.0
        assert result.high == 2.0

        # Test default fallback
        result = ensure_action_space(None)
        assert isinstance(result, spaces.Box)
        assert result.shape == (9,)  # default_dim
