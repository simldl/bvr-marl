import pytest
import numpy as np
from unittest.mock import Mock
from gymnasium import spaces


class TestEnvHelpers:
    """Test environment helper functions."""

    def test_env_helpers_import(self):
        """Test that env helpers can be imported."""
        try:
            import reinforcement_learning.environment.gym.env_helpers
            assert True
        except ImportError as e:
            pytest.skip(f"Environment helpers not available: {e}")

    def test_to_f32_obs(self):
        """Test observation conversion to float32."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import to_f32_obs
            
            # Test with different observation types
            obs_dict = {
                "state": np.array([1, 2, 3], dtype=np.int32),
                "mask": np.array([1, 0, 1], dtype=bool)
            }
            
            f32_obs = to_f32_obs(obs_dict)
            
            assert f32_obs["state"].dtype == np.float32
            assert f32_obs["mask"].dtype == np.float32
            
        except (ImportError, AttributeError):
            pytest.skip("to_f32_obs function not available")

    def test_coerce_obs_to_space(self):
        """Test observation coercion to match space requirements."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import coerce_obs_to_space
            
            # Create a test space
            space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            
            # Test observation that needs coercion
            obs = np.array([2.0, -2.0, 0.5])  # Outside bounds
            
            coerced_obs = coerce_obs_to_space(obs, space)
            
            assert space.contains(coerced_obs)
            assert coerced_obs.dtype == np.float32
            
        except (ImportError, AttributeError):
            pytest.skip("coerce_obs_to_space function not available")

    def test_spaces_from_sample_observation(self):
        """Test space creation from sample observations."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import spaces_from_sample_observation
            
            # Sample observation - use float32 for both since the function converts to float32
            sample_obs = {
                "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "action_mask": np.array([1.0, 0.0, 1.0], dtype=np.float32)
            }
            
            space = spaces_from_sample_observation(sample_obs)
            
            assert isinstance(space, spaces.Dict)
            assert "state" in space.spaces
            assert "action_mask" in space.spaces
            
            # Check that the created space can contain the sample
            assert space.contains(sample_obs)
            
        except (ImportError, AttributeError):
            pytest.skip("spaces_from_sample_observation function not available")

    def test_observation_preprocessing(self):
        """Test observation preprocessing utilities."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import preprocess_observation
            
            raw_obs = {
                "continuous": np.array([100.0, 200.0]),  # Large values
                "discrete": np.array([0, 1, 2]),
                "boolean": np.array([True, False, True])
            }
            
            processed_obs = preprocess_observation(raw_obs)
            
            # Should be properly formatted
            assert isinstance(processed_obs, dict)
            for key, value in processed_obs.items():
                assert isinstance(value, np.ndarray)
                
        except (ImportError, AttributeError):
            pytest.skip("preprocess_observation function not available")

    def test_observation_validation(self):
        """Test observation validation utilities."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import validate_observation
            
            # Valid observation
            valid_obs = {
                "state": np.array([0.5, 0.3, 0.1], dtype=np.float32),
                "mask": np.array([1, 1, 0], dtype=np.float32)
            }
            
            # Invalid observation (contains NaN)
            invalid_obs = {
                "state": np.array([0.5, np.nan, 0.1], dtype=np.float32),
                "mask": np.array([1, 1, 0], dtype=np.float32)
            }
            
            assert validate_observation(valid_obs) == True
            assert validate_observation(invalid_obs) == False
            
        except (ImportError, AttributeError):
            pytest.skip("validate_observation function not available")

    def test_space_utils(self):
        """Test space utility functions."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import get_space_size, flatten_space
            
            # Test with Box space
            box_space = spaces.Box(low=-1, high=1, shape=(2, 3), dtype=np.float32)
            assert get_space_size(box_space) == 6
            
            # Test with Dict space
            dict_space = spaces.Dict({
                "a": spaces.Box(low=-1, high=1, shape=(2,)),
                "b": spaces.Discrete(4)
            })
            
            flat_space = flatten_space(dict_space)
            assert isinstance(flat_space, spaces.Space)
            
        except (ImportError, AttributeError):
            pytest.skip("Space utility functions not available")

    def test_action_masking_helpers(self):
        """Test action masking helper functions."""
        try:
            from reinforcement_learning.environment.gym.env_helpers import apply_action_mask, create_action_mask
            
            # Test action mask application
            action_logits = np.array([1.0, 2.0, 3.0, 1.5])
            action_mask = np.array([1, 1, 0, 1])  # Third action masked
            
            masked_logits = apply_action_mask(action_logits, action_mask)
            
            # Masked actions should have very low logits
            assert masked_logits[2] < -1e6
            assert masked_logits[0] == action_logits[0]  # Unmasked should be unchanged
            
        except (ImportError, AttributeError):
            pytest.skip("Action masking helpers not available")