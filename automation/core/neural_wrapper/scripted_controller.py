"""
Fully scripted controller without neural networks.
For fully automated system operation.
"""
import numpy as np
from typing import Optional
from automation.core.auto_helper import AutoHelper, AutoHelperConfig


class FullyScriptedController:
    """
    Fully scripted controller that handles all actions without neural networks.
    This is for the future fully automated system.
    """

    def __init__(self, aircraft, config: Optional[AutoHelperConfig] = None):
        """
        Initialize fully scripted controller.

        Args:
            aircraft: The aircraft to control
            config: Configuration for automation
        """
        self.aircraft = aircraft
        self.auto_helper = AutoHelper(aircraft, config)

        # Scripted behavior parameters
        self.engagement_range_km = 30
        self.defensive_range_km = 10
        self.preferred_altitude_m = 10000
        self.preferred_speed_mps = 300

    def get_action(self, simulator, dt: float) -> np.ndarray:
        """
        Get fully automated action.

        Args:
            simulator: The simulation environment
            dt: Time delta since last update

        Returns:
            Complete action array
        """
        # Placeholder for future full automation
        # Return conservative defaults with automation
        manual_controls = {
            'throttle': 0.7,
            'yaw_change': 0.5,
            'pitch_change': 0.5,
            'missile_fire': 0.0
        }

        return self.auto_helper.get_action_values(simulator, dt, manual_controls)
