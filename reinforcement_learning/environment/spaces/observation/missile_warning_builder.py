"""
Missile Warning Builder - Builds missile warning features with sector encoding.

Builds observations for:
- Warning flag: normalized count of incoming missile warnings
- Warning directions: threat level + one-hot sector encoding per warning
- Mask: validity mask for warning slots

Uses ObservationHelper and MissileWarner system for consistency with BT.
"""
from typing import Tuple
import numpy as np
from .helpers import pad_generic
from aircrafts.systems.observation_helper import ObservationHelper


class MissileWarningBuilder:
    """Builds missile warning observation components."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        # Cache ObservationHelpers per agent (created lazily)
        self._obs_helpers = {}

    def _get_obs_helper(self, unit) -> ObservationHelper:
        """Get or create ObservationHelper for unit (cached)."""
        if unit.id not in self._obs_helpers:
            self._obs_helpers[unit.id] = ObservationHelper(unit)
        return self._obs_helpers[unit.id]

    def build(self, unit) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Build missile warning observations.

        Uses ObservationHelper.get_threat_warnings() for consistency with BT.
        Encodes warnings as threat level + sector one-hot encoding.

        Args:
            unit: The unit to build warnings for

        Returns:
            Tuple of 3 elements:
            - warning_flag: Normalized count of warnings [0-1]
            - warning_dirs: Threat level + sector encoding [em_slots * (1 + warn_sectors)]
            - warning_mask: Validity mask [em_slots]
        """
        obs_helper = self._get_obs_helper(unit)

        # Get threat warnings from helper (same source as BT)
        threat_data = obs_helper.get_threat_warnings()
        num_warnings = threat_data.get('num_warnings', 0)
        warning_ids = threat_data.get('warning_ids', [])

        # Normalize warning count
        count = min(num_warnings, self.config.em_slots)
        count_norm = count / np.float32(self.config.em_slots) if self.config.em_slots > 0 else 0.0

        # Build sector-based warning info
        info_list = []
        for wid in warning_ids[:self.config.em_slots]:
            # Find the missile in active units
            missile = None
            for u in self.simulator.active_units.values():
                if u.id == wid:
                    missile = u
                    break

            if missile:
                tnorm = 1.0  # Threat level (could be enhanced with range/closure)
                # Calculate bearing to missile in own reference frame (compass-from-North)
                dE = (missile.position.lon - unit.position.lon) * 111_000.0 * np.cos(np.radians(unit.position.lat))
                dN = (missile.position.lat - unit.position.lat) * 111_000.0
                bearing_compass = (np.degrees(np.arctan2(dE, dN)) - unit.yaw_deg) % 360.0
                sector = int(bearing_compass // (360.0 / self.config.warn_sectors))
                sector = min(max(sector, 0), self.config.warn_sectors - 1)
                one_hot = [0.0] * self.config.warn_sectors
                one_hot[sector] = 1.0
                info_list.append([tnorm] + one_hot)
            else:
                # Warning ID but missile not found - maybe just died
                info_list.append([0.0] + [0.0] * self.config.warn_sectors)

        dim = 1 + self.config.warn_sectors
        info_arr, info_mask = pad_generic(info_list, self.config.em_slots, dim)

        return count_norm, info_arr.flatten(), info_mask
