"""
Neural network inference module.
Handles neural network prediction and output processing.
"""
import numpy as np
from typing import Dict, Callable


class NeuralInference:
    """Handle neural network inference for manual controls."""

    def __init__(self, neural_model: Callable = None):
        """
        Initialize neural inference.

        Args:
            neural_model: Neural network model callable
        """
        self.neural_model = neural_model

    def set_model(self, model: Callable):
        """Set or update the neural network model."""
        self.neural_model = model

    def get_neural_actions(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Get the 4 manual control actions from the neural network.

        Args:
            observation: State observation

        Returns:
            Dict with the 4 manual control values
        """
        try:
            # Neural network should return 4 values
            nn_output = self.neural_model(observation)

            # Ensure output is the right shape and normalized
            if isinstance(nn_output, (list, tuple)):
                nn_output = np.array(nn_output)

            if len(nn_output) != 4:
                raise ValueError(f"Neural network must output 4 actions, got {len(nn_output)}")

            return {
                'throttle': np.clip(nn_output[0], 0.0, 1.0),
                'yaw_change': np.clip(nn_output[1], 0.0, 1.0),
                'pitch_change': np.clip(nn_output[2], 0.0, 1.0),
                'missile_fire': np.clip(nn_output[3], 0.0, 1.0)
            }

        except Exception as e:
            print(f"Warning: Neural network error: {e}. Using default controls.")
            return self._get_default_controls()

    def _get_default_controls(self) -> Dict[str, float]:
        """Get safe default controls."""
        return {
            'throttle': 0.7,
            'yaw_change': 0.5,
            'pitch_change': 0.5,
            'missile_fire': 0.0
        }
