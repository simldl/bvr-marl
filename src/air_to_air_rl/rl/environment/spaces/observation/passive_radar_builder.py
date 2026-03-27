"""
Passive Radar Builder - Extracts passive radar detections.

Builds observations for:
- Passive radar detections: relative angle, distance, and age
"""

import numpy as np

from .helpers import pad_generic


class PassiveRadarBuilder:
    """Builds passive radar observation components."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config

    def build(self, unit) -> tuple[np.ndarray, np.ndarray]:
        """
        Build passive radar observations.

        Extracts passive radar detections with angle, distance, and age.

        Args:
            unit: The unit to build passive radar observations for

        Returns:
            tuple of 2 arrays:
            - pr_states: Passive radar detections [pr_slots, 3] (angle, distance, age)
            - pr_mask: Validity mask [pr_slots]
        """
        t = self.simulator.elapsed_time_s
        unit.sensor.passive_radar.clear_old(t)
        dets = unit.sensor.passive_radar.get_observation(t)
        states = [[d["rel_angle_deg"], d["distance_m"], d["age_s"]] for d in dets]
        return pad_generic(states, self.config.pr_slots, 3)
