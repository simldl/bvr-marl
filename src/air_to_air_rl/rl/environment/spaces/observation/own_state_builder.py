"""
Own state builder for agent observations.
Builds comprehensive self-state including SQI, DLZ zone, envelope, radar, and boundary information.
"""

import numpy as np

from air_to_air_rl.aircrafts.systems.observation_helper import ObservationHelper

from .constants import d_OWN


class OwnStateBuilder:
    """Build own state observation vector."""

    def __init__(self, simulator, config):
        """
        Initialize own state builder.

        Args:
            simulator: Simulator instance
            config: Environment configuration
        """
        self.simulator = simulator
        self.config = config
        self._obs_helpers = {}

    def _get_obs_helper(self, unit) -> ObservationHelper:
        """Get or create ObservationHelper for unit (cached)."""
        if unit.id not in self._obs_helpers:
            self._obs_helpers[unit.id] = ObservationHelper(unit)
        return self._obs_helpers[unit.id]

    def build(self, unit) -> np.ndarray:
        """
        Build own state vector using ObservationHelper for consistency.

        Returns 22-element vector (reduced from 26, removed DLZ range redundancy):
        [0-5]:   Position/orientation (lat, lon, alt, yaw, speed, pitch)
        [6]:     Remaining missiles (normalized)
        [7]:     Ps (specific excess power, kW normalized)
        [8]:     SQI (shot quality index) [0-1]
        [9]:     DLZ zone encoded [0-1] (R1=0.25, R2=0.5, R3=0.75, R4=1.0)
        [10-11]: Envelope scalars (s_factor, c_factor)
        [12]:    Radar max range (normalized to 150km)
        [13]:    Radar FOV (normalized to 180deg)
        [14]:    Own RCS (normalized to 10 m²)
        [15]:    Radar power (normalized)
        [16]:    Radar lock quality [0-1]
        [17-18]: Boundary distances normalized [0-1] (min of left/right, min of bottom/top)
        [19-21]: Reserved for future use (set to 0)

        REMOVED (moved to per-fighter NEZ or redundant):
        - nez_visible (redundant with per-fighter NEZ)
        - DLZ ranges (r_min, r_tr, r_pi, r_aero) - redundant with per-fighter data
        - slant_range_to_target - can be computed from enemy fighter relative position

        Args:
            unit: Aircraft unit

        Returns:
            np.ndarray: 22-element state vector
        """
        obs_helper = self._get_obs_helper(unit)
        target = getattr(unit, "target", None)

        # Get SQI and DLZ zone
        sqi, dlz_zone = self._get_sqi_dlz_features(obs_helper, target)

        # Get Ps (specific excess power)
        ps_normalized = self._get_specific_excess_power(unit)

        # Get envelope scalars
        envelope_features = self._get_envelope_features(unit)

        # Get radar features
        radar_features = self._get_radar_features(unit, obs_helper, target)

        # Get boundary features (compressed to 2 values)
        boundary_features = self._get_boundary_features_compressed(unit)

        # Build complete state vector
        state = np.array(
            [
                unit.position.lat,
                unit.position.lon,
                unit.position.alt,
                unit.yaw_deg,
                unit.speed,
                unit.pitch_deg,
                unit.max_missiles - len(unit.missiles),  # Remaining missiles
                ps_normalized,
                sqi,
                dlz_zone,
                *envelope_features,
                *radar_features,
                *boundary_features,
                0.0,
                0.0,
                0.0,  # Reserved slots
            ],
            dtype=np.float32,
        )

        assert len(state) == d_OWN, f"OwnState dimension mismatch: {len(state)} != {d_OWN}"
        return state

    def _get_sqi_dlz_features(self, obs_helper, target) -> tuple:
        """Extract SQI and DLZ zone only (2 elements)."""
        dlz_nez = obs_helper.get_dlz_nez_features(target)

        sqi = dlz_nez.get("sqi", 0.0)
        dlz_zone_map = {"R1": 0.25, "R2": 0.5, "R3": 0.75, "R4": 1.0}
        dlz_zone = dlz_zone_map.get(dlz_nez.get("zone", "R4"), 1.0)

        return sqi, dlz_zone

    def _get_specific_excess_power(self, unit) -> float:
        """
        Calculate specific excess power (Ps) normalized.

        Ps = (T - D) * V / W
        where T=thrust, D=drag, V=velocity, W=weight

        Normalize to typical range: [-50 to +200 kW] -> [0, 1]
        """
        try:
            if hasattr(unit, "get_specific_excess_power"):
                ps_kw = unit.get_specific_excess_power()
            elif hasattr(unit, "get_energy_state"):
                energy_state = unit.get_energy_state()
                ps_kw = energy_state.get("ps_kw", 0.0)
            else:
                # Fallback: approximate from speed and altitude changes
                ps_kw = 0.0

            # Normalize: typical range is -50 to +200 kW
            ps_normalized = np.clip((ps_kw + 50.0) / 250.0, 0.0, 1.0)
            return ps_normalized
        except Exception:
            return 0.5  # Neutral value if calculation fails

    def _get_envelope_features(self, unit) -> list:
        """Get envelope scalars from action processor (2 elements)."""
        s_factor = 1.0
        c_factor = 1.0
        try:
            if hasattr(self.simulator, "action_processor"):
                action_proc = self.simulator.action_processor
                if hasattr(action_proc, "get_envelope_scalars"):
                    envelope_data = action_proc.get_envelope_scalars(unit.id)
                    s_factor = envelope_data.get("s_factor", 1.0)
                    c_factor = envelope_data.get("c_factor", 1.0)
        except Exception:
            pass
        return [s_factor, c_factor]

    def _get_radar_features(self, unit, obs_helper, target) -> list:
        """Get radar-specific features (5 elements)."""
        radar_max_range_norm = 0.0
        radar_fov_norm = 0.0
        own_rcs_norm = 0.0
        radar_power_norm = 0.0
        radar_lock_quality = 0.0

        if hasattr(unit, "radar"):
            radar = unit.radar
            radar_max_range_norm = getattr(radar, "max_range_m", 0.0) / 150000.0
            radar_fov_norm = getattr(radar, "h_fov_deg", 0.0) / 180.0
            tx_p = getattr(
                radar, "tx_power_w", getattr(radar, "power_w", getattr(radar, "power", 0.0))
            )
            radar_power_norm = tx_p / 1000.0

        if hasattr(unit, "rcs"):
            own_rcs_norm = getattr(unit, "rcs", 0.0) / 10.0

        if target:
            lock_data = obs_helper.get_lock_quality(target)
            radar_lock_quality = lock_data.get("lock_strength", 0.0)

        return [
            radar_max_range_norm,
            radar_fov_norm,
            own_rcs_norm,
            radar_power_norm,
            radar_lock_quality,
        ]

    def _get_boundary_features_compressed(self, unit) -> list:
        """
        Get compressed boundary distance features (2 elements).

        Instead of 4 separate distances, compute:
        - min(left, right) - how close to horizontal boundaries
        - min(bottom, top) - how close to vertical boundaries

        This reduces dimensionality while preserving boundary proximity information.
        """
        boundary_dist_horizontal = 0.5
        boundary_dist_vertical = 0.5

        if hasattr(self.simulator, "map_limits"):
            map_limits = self.simulator.map_limits
            lat = unit.position.lat
            lon = unit.position.lon

            lon_extent = map_limits.longitude_extent()
            lat_extent = map_limits.latitude_extent()

            if lon_extent > 0:
                dist_left = (lon - map_limits.left_lon) / lon_extent
                dist_right = (map_limits.right_lon - lon) / lon_extent
                boundary_dist_horizontal = min(dist_left, dist_right)

            if lat_extent > 0:
                dist_bottom = (lat - map_limits.bottom_lat) / lat_extent
                dist_top = (map_limits.top_lat - lat) / lat_extent
                boundary_dist_vertical = min(dist_bottom, dist_top)

            boundary_dist_horizontal = np.clip(boundary_dist_horizontal, 0.0, 1.0)
            boundary_dist_vertical = np.clip(boundary_dist_vertical, 0.0, 1.0)

        return [boundary_dist_horizontal, boundary_dist_vertical]
