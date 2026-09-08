"""
Passive Radar Builder - Extracts passive radar detections.

Builds observations for:
- Passive radar detections: relative angle, distance, and age
"""

import numpy as np

from bvr_marl_core.rl.environment.spaces.observation.helpers import pad_tokens


class PassiveRadarBuilder:
    """Builds passive radar observation components."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config

    def build(self, unit) -> np.ndarray:
        """
        Build passive radar (ESM) detection tokens.

        Extracts passive radar detections with angle, distance, and age, with the
        validity mask folded into each token's last column.

        Args:
            unit: The unit to build passive radar observations for

        Returns:
            pr_tokens: [pr_slots, 4] = [rel_angle_deg, distance_m, age_s, mask]
        """
        t = self.simulator.elapsed_time_s
        unit.sensor.passive_radar.clear_old(t)
        dets = unit.sensor.passive_radar.get_observation(t)
        states = [[d["rel_angle_deg"], d["distance_m"], d["age_s"]] for d in dets]
        return pad_tokens(states, self.config.pr_slots, 3)
