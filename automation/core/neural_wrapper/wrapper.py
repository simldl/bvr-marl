"""
Main neural wrapper implementation.
Combines neural network control with automated systems.
"""
import numpy as np
from typing import Dict, Any, Optional, Callable
from automation.core.auto_helper import AutoHelper, AutoHelperConfig
from .neural_inference import NeuralInference
from .metrics import MetricsTracker
from .observation_info import ObservationInfoProvider


class NeuralWrapper:
    """
    Wrapper that combines neural network control with automated systems.

    The neural network controls:
    - Throttle (action[0])
    - Yaw change (action[1])
    - Pitch change (action[2])
    - Missile firing (action[4])

    The AutoHelper controls:
    - Target selection (action[3])
    - Countermeasures (actions[5-8])
    """

    def __init__(self, aircraft, neural_model: Optional[Callable] = None,
                 config: Optional[AutoHelperConfig] = None):
        """
        Initialize the neural wrapper.

        Args:
            aircraft: The aircraft to control
            neural_model: Neural network model
            config: Configuration for automation
        """
        self.aircraft = aircraft
        self.auto_helper = AutoHelper(aircraft, config)
        self.neural_inference = NeuralInference(neural_model)
        self.metrics_tracker = MetricsTracker()
        self.obs_info_provider = ObservationInfoProvider()

    def set_neural_model(self, model: Callable):
        """Set or update the neural network model."""
        self.neural_inference.set_model(model)

    def get_action(self, observation: np.ndarray, simulator, dt: float) -> np.ndarray:
        """
        Get complete action combining neural and automated systems.

        Args:
            observation: Current state observation
            simulator: The simulation environment
            dt: Time delta since last update

        Returns:
            Complete action array
        """
        # Get manual controls from neural network
        if self.neural_inference.neural_model is not None:
            manual_controls = self.neural_inference.get_neural_actions(observation)
        else:
            manual_controls = self.neural_inference._get_default_controls()

        # Get complete action from AutoHelper
        complete_action = self.auto_helper.get_action_values(simulator, dt, manual_controls)

        # Update metrics
        self.metrics_tracker.update_from_action(complete_action)

        return complete_action

    def update_automation_config(self, **kwargs):
        """Update automation configuration parameters."""
        config = self.auto_helper.config

        if 'automation_level' in kwargs:
            self.auto_helper.set_automation_level(kwargs['automation_level'])

        if 'enable_countermeasures' in kwargs:
            config.enable_countermeasures = kwargs['enable_countermeasures']

        if 'enable_target_selection' in kwargs:
            config.enable_target_selection = kwargs['enable_target_selection']

        if 'threat_response_thresholds' in kwargs:
            config.threat_response_thresholds.update(kwargs['threat_response_thresholds'])

        # Update subsystems
        self.auto_helper.threat_assessment.update_config(config)
        self.auto_helper.target_manager.update_config(config)
        self.auto_helper.countermeasure_controller.update_config(config)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the wrapper system."""
        return {
            'neural_model_active': self.neural_inference.neural_model is not None,
            'automation_status': self.auto_helper.get_status(),
            'performance_metrics': self.metrics_tracker.get_metrics(),
            'config': {
                'automation_level': self.auto_helper.config.automation_level,
                'countermeasures_enabled': self.auto_helper.config.enable_countermeasures,
                'target_selection_enabled': self.auto_helper.config.enable_target_selection
            }
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics_tracker.reset()

    def get_observation_info(self) -> Dict[str, Any]:
        """Get information about observation structure."""
        return self.obs_info_provider.get_observation_info()
