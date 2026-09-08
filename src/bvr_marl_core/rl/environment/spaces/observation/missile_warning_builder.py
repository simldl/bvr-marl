"""
Missile Warning Builder - Builds missile warning features with sector encoding.

Builds observations for:
- Warning flag: normalized count of incoming missile warnings
- Warning directions: threat level + one-hot sector encoding per warning
- Mask: validity mask for warning slots

Uses ObservationHelper and MissileWarner system for consistency with BT.
"""

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.rl.environment.spaces.observation.constants import WARN_FEATURE_DIM
from bvr_marl_core.rl.environment.spaces.observation.helpers import pad_tokens
from bvr_marl_core.tactical import ObservationHelper


class MissileWarningBuilder:
    """Builds missile warning observation components."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self.information_mode = resolve_information_mode(
            getattr(config, "information_mode", None), default=InformationMode.SENSOR_LIMITED
        )
        # Cache ObservationHelpers per agent (created lazily)
        self._obs_helpers = {}

    def _get_obs_helper(self, unit) -> ObservationHelper:
        """Get or create ObservationHelper for unit (cached)."""
        if unit.id not in self._obs_helpers:
            self._obs_helpers[unit.id] = ObservationHelper(unit)
        return self._obs_helpers[unit.id]

    def build(self, unit) -> np.ndarray:
        """
        Build missile-warning tokens.

        Uses ObservationHelper.get_threat_warnings() for consistency with BT.
        Each warning is a token [urgency, sin(bearing), cos(bearing), mask]; the
        mask column (and the attention over these tokens) makes the old scalar
        "warning count" flag redundant, so it is no longer emitted.

        Args:
            unit: The unit to build warnings for

        Returns:
            warn_tokens: [em_slots, WARN_FEATURE_DIM + 1] with mask in the last column
        """
        if self.information_mode is InformationMode.SENSOR_LIMITED:
            return self._build_from_warning_and_track_estimates(unit)

        obs_helper = self._get_obs_helper(unit)

        # Get threat warnings from helper (same source as BT)
        threat_data = obs_helper.get_threat_warnings()
        warning_ids = threat_data.get("warning_ids", [])

        # Build sector-based warning info
        info_list = []
        for wid in warning_ids[: self.config.em_slots]:
            # Find the missile in active units
            missile = None
            for u in self.simulator.active_units.values():
                if u.id == wid:
                    missile = u
                    break

            if missile:
                # Threat level encodes urgency (1 = impact imminent, 0 = far/not
                # closing) from range + closing rate, so the policy can learn the
                # timing of a last-ditch break rather than only its direction.
                tnorm = self._threat_urgency(unit, missile)
                # Bearing to the missile relative to own nose, encoded as
                # (sin, cos) so it is continuous across the 0/360 wraparound —
                # more compact and learnable than the old N-way sector one-hot.
                dE = (
                    (missile.position.lon - unit.position.lon)
                    * 111_000.0
                    * np.cos(np.radians(unit.position.lat))
                )
                dN = (missile.position.lat - unit.position.lat) * 111_000.0
                bearing_rad = np.radians(np.degrees(np.arctan2(dE, dN)) - unit.yaw_deg)
                info_list.append([tnorm, float(np.sin(bearing_rad)), float(np.cos(bearing_rad))])
            else:
                # Warning ID but missile not found - maybe just died
                info_list.append([0.0, 0.0, 0.0])

        return pad_tokens(info_list, self.config.em_slots, WARN_FEATURE_DIM)

    def _build_from_warning_and_track_estimates(self, unit) -> np.ndarray:
        """Build warnings without resolving warning IDs through simulator truth."""
        sensor = getattr(unit, "sensor", None)
        warner = getattr(sensor, "missile_warner", None)
        warning_ids = (
            list(warner.get_current_warning_ids())
            if warner is not None and hasattr(warner, "get_current_warning_ids")
            else []
        )
        if not warning_ids:
            return pad_tokens([], self.config.em_slots, WARN_FEATURE_DIM)

        contacts = []
        converter = getattr(sensor, "tactical_contact", TacticalContact.from_track_snapshot)
        for track in getattr(sensor, "sensor_tracks", ()) or ():
            contact = converter(track)
            if contact.is_missile and not contact.suspect_deception:
                contacts.append(contact)
        contacts.sort(key=lambda contact: str(contact.track_id))

        info_list = [self._contact_warning_features(unit, contact) for contact in contacts]
        # The current warning model exposes only IDs when no radar track exists.
        # Preserve the warning itself without inventing a truth-derived bearing.
        missing = max(0, len(warning_ids) - len(info_list))
        info_list.extend([[1.0, 0.0, 0.0]] * missing)
        return pad_tokens(info_list[: len(warning_ids)], self.config.em_slots, WARN_FEATURE_DIM)

    def _contact_warning_features(self, unit, contact: TacticalContact) -> list[float]:
        d_east, d_north, d_up, missile_vx, missile_vy, missile_vz = contact.state
        bearing_rad = np.arctan2(d_east, d_north) - np.radians(float(unit.yaw_deg))
        own_vx, own_vy, own_vz = self._vel_xyz(getattr(unit, "velocity", None))
        distance = float(np.sqrt(d_east**2 + d_north**2 + d_up**2))
        if distance < 1e-6:
            urgency = 1.0
        else:
            closing = (
                -(
                    (missile_vx - own_vx) * d_east
                    + (missile_vy - own_vy) * d_north
                    + (missile_vz - own_vz) * d_up
                )
                / distance
            )
            urgency = (
                0.0
                if closing <= 0.0
                else float(np.clip(1.0 - distance / closing / self._TTI_HORIZON_S, 0.0, 1.0))
            )
        return [urgency, float(np.sin(bearing_rad)), float(np.cos(bearing_rad))]

    @staticmethod
    def _vel_xyz(v) -> tuple[float, float, float]:
        """ENU components from a velocity (namedtuple/array), or zeros."""
        if v is None:
            return (0.0, 0.0, 0.0)
        try:
            return (
                float(getattr(v, "vx", v[0])),
                float(getattr(v, "vy", v[1])),
                float(getattr(v, "vz", v[2])),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    # Time-to-impact horizon (s) over which urgency ramps from 0 (far) to 1 (now).
    _TTI_HORIZON_S = 60.0

    def _threat_urgency(self, unit, missile) -> float:
        """Normalized incoming-missile urgency in [0, 1].

        1.0 == impact imminent, 0.0 == far away or not closing. Derived from the
        range and LOS closing rate between the unit and the warning missile.
        """
        try:
            cos_lat = np.cos(np.radians(unit.position.lat))
            dE = (missile.position.lon - unit.position.lon) * 111_000.0 * cos_lat
            dN = (missile.position.lat - unit.position.lat) * 111_000.0
            dU = missile.position.alt - unit.position.alt
            rng = float(np.sqrt(dE * dE + dN * dN + dU * dU))
            if rng < 1e-6:
                return 1.0
            mvx, mvy, mvz = self._vel_xyz(getattr(missile, "velocity", None))
            ovx, ovy, ovz = self._vel_xyz(getattr(unit, "velocity", None))
            # Closing rate = -d(range)/dt = -((v_missile - v_unit) . r_hat).
            closing = -((mvx - ovx) * dE + (mvy - ovy) * dN + (mvz - ovz) * dU) / rng
            if closing <= 0.0:
                return 0.0  # missile not closing (beaten/defeated/off-track)
            tti_s = rng / closing
            return float(np.clip(1.0 - tti_s / self._TTI_HORIZON_S, 0.0, 1.0))
        except Exception:
            return 1.0  # safe default: treat as urgent
