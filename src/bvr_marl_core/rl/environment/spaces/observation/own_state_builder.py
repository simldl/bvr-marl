"""
Own state builder for agent observations.
Builds compact self-state: energy/orientation, weapon/envelope readiness,
radar lock, and directional horizontal + altitude boundary awareness.
"""

import math

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    OWN_EMCON_TOGGLE_REF_STEPS,
    own_state_dim,
)
from bvr_marl_core.tactical import ObservationHelper

# Endurance reference (s) for normalizing station time (~1 hour).
_STATION_TIME_REF_S = 3600.0


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
        self.information_mode = resolve_information_mode(
            getattr(config, "information_mode", None), default=InformationMode.SENSOR_LIMITED
        )
        # Active sensing: widen the ownship vector with own radar-emission state.
        self.emcon_action_enabled = bool(getattr(config, "emcon_action_enabled", False))
        self._own_dim = own_state_dim(self.emcon_action_enabled)
        self._obs_helpers = {}

    def _get_obs_helper(self, unit) -> ObservationHelper:
        """Get or create ObservationHelper for unit (cached)."""
        if unit.id not in self._obs_helpers:
            self._obs_helpers[unit.id] = ObservationHelper(unit)
        return self._obs_helpers[unit.id]

    def build(self, unit) -> np.ndarray:
        """
        Build the own-state vector (``d_OWN`` features, all normalized to ~[-1, 1]).

        See ``constants.d_OWN`` for the authoritative index layout. Summary:
        heading sin/cos, speed, pitch, remaining missiles, Ps, radar-lock quality,
        episode time remaining, can-fire flag, horizontal-boundary cues (3),
        altitude-boundary cues (2), fuel fraction, and station time.

        The envelope scalars (s_factor, c_factor) and the in-envelope flag were
        dropped in the observation-space rework: they are redundant with the
        enemy-fighter shot cues (DLZ zone / SQI) and the can-fire flag.

        Args:
            unit: Aircraft unit

        Returns:
            np.ndarray: ``d_OWN``-element state vector
        """
        obs_helper = self._get_obs_helper(unit)
        target = (
            self._locked_contact(unit)
            if self.information_mode is InformationMode.SENSOR_LIMITED
            else getattr(unit, "target", None)
        )

        ps_normalized = self._get_specific_excess_power(unit)
        radar_lock_quality = self._get_radar_lock_quality(unit, obs_helper, target)
        tactical_cues = self._get_tactical_cues(obs_helper, target)
        boundary_direction_features = self._get_boundary_direction_features(unit)
        altitude_boundary_features = self._get_altitude_boundary_features(unit)

        weapons_state = obs_helper.get_weapons_state()
        remaining_raw = float(weapons_state.get("remaining_missiles", 0.0))
        max_missiles = float(getattr(unit, "max_missiles", 0.0)) or 8.0
        remaining_missiles = float(np.clip(remaining_raw / max_missiles, 0.0, 1.0))

        # Normalize orientation/energy: heading as sin/cos (wraparound-safe),
        # speed by max speed, pitch by 90deg.
        yaw_rad = math.radians(float(getattr(unit, "yaw_deg", 0.0)))
        max_speed = float(getattr(unit, "max_speed_mps", 0.0)) or 600.0
        speed_norm = float(np.clip(float(getattr(unit, "speed", 0.0)) / max_speed, 0.0, 1.0))
        pitch_norm = float(np.clip(float(getattr(unit, "pitch_deg", 0.0)) / 90.0, -1.0, 1.0))

        fuel_system = getattr(unit, "fuel", None)
        fuel_fraction = float(fuel_system.fuel_fraction) if fuel_system is not None else 1.0
        # On-station endurance normalized by a 1-hour reference, clipped to [0, 1].
        station_time = (
            float(np.clip(fuel_system.station_time_s / _STATION_TIME_REF_S, 0.0, 1.0))
            if fuel_system is not None
            else 1.0
        )

        features = [
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            speed_norm,
            pitch_norm,
            remaining_missiles,
            ps_normalized,
            radar_lock_quality,
            *tactical_cues,
            *boundary_direction_features,
            *altitude_boundary_features,
            fuel_fraction,
            station_time,
        ]
        if self.emcon_action_enabled:
            features.extend(self._emcon_features(unit))

        state = np.array(features, dtype=np.float32)
        assert len(state) == self._own_dim, (
            f"OwnState dimension mismatch: {len(state)} != {self._own_dim}"
        )
        return state

    def _emcon_features(self, unit) -> list[float]:
        """Own radar-emission state: [emitting flag, duty cycle, time since toggle]."""
        emitting = 1.0 if bool(getattr(unit, "radar_emitting", True)) else 0.0
        tracker = getattr(unit, "emission_tracker", None)
        if tracker is None:
            return [emitting, 1.0, 1.0]
        duty = float(np.clip(tracker.duty_cycle, 0.0, 1.0))
        since_toggle = float(
            np.clip(tracker.steps_since_toggle / OWN_EMCON_TOGGLE_REF_STEPS, 0.0, 1.0)
        )
        return [emitting, duty, since_toggle]

    @staticmethod
    def _locked_contact(unit) -> TacticalContact | None:
        """Return a lock-grade operational contact for ownship tactical cues."""
        sensor = getattr(unit, "sensor", None)
        try:
            locked_ids = set(sensor.get_locked_targets() or ())
        except (AttributeError, TypeError, ValueError):
            return None
        for track in getattr(sensor, "sensor_tracks", ()) or ():
            if track.track_id in locked_ids:
                try:
                    contact = sensor.tactical_contact(track)
                except (TypeError, ValueError):
                    continue
                if contact.engageable and not contact.suspect_deception and not contact.is_missile:
                    return contact
        return None

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

    def _get_radar_lock_quality(self, unit, obs_helper, target) -> float:
        """Radar lock quality on the selected target [0-1] (dynamic).

        Static radar parameters (max range, FOV, RCS, tx power) are deliberately
        omitted: they are constant per aircraft type and carry no per-step signal.
        """
        if target:
            lock_data = obs_helper.get_lock_quality(target)
            return float(lock_data.get("lock_strength", 0.0))
        return 0.0

    def _get_tactical_cues(self, obs_helper, target) -> list:
        """Two actionable tactical scalars (episode time, fire readiness).

        - time_remaining [0-1]: fraction of the episode left (1=fresh, 0=timeout
          imminent), so the policy can reason about committing before a timeout.
        - can_fire [0/1]: every launch gate passed *right now* (lock, FOV, gimbal,
          range, inventory, cooldown) — the actionable "may I shoot" signal that
          remaining-missile count alone does not convey.

        The former in-envelope cue was removed from the observation, so it is no
        longer computed here (it triggered an otherwise-unneeded DLZ dict build).
        """
        episode_max_s = float(getattr(self.simulator, "episode_max_s", 0.0))
        elapsed_s = float(getattr(self.simulator, "elapsed_time_s", 0.0))
        time_remaining = (
            float(np.clip(1.0 - elapsed_s / episode_max_s, 0.0, 1.0))
            if episode_max_s > 0.0
            else 1.0
        )

        can_fire = 0.0
        if target:
            try:
                can_fire = (
                    1.0
                    if obs_helper.get_fire_feasibility(target, simulator=self.simulator).get(
                        "can_fire"
                    )
                    else 0.0
                )
            except Exception:
                can_fire = 0.0

        return [time_remaining, can_fire]

    def _get_boundary_direction_features(self, unit) -> list:
        """Directional horizontal-boundary cues (3 elements).

        Encodes *which* edge is near and whether the aircraft is heading at it —
        the information a policy needs to avoid a boundary death:
          - signed horizontal position in box [-1,1]  (-1=left edge, +1=right edge)
          - signed vertical position in box   [-1,1]  (-1=bottom edge, +1=top edge)
          - closing rate toward the NEAREST horizontal edge, normalized by max
            speed, in [-1,1] (>0 = flying toward that edge / about to exit).

        Returns neutral zeros when map limits are unavailable.
        """
        signed_h = 0.0
        signed_v = 0.0
        closing_norm = 0.0

        map_limits = getattr(self.simulator, "map_limits", None)
        if map_limits is None:
            return [signed_h, signed_v, closing_norm]

        lat = unit.position.lat
        lon = unit.position.lon
        lon_extent = map_limits.longitude_extent()
        lat_extent = map_limits.latitude_extent()

        if lon_extent > 0:
            center_lon = 0.5 * (map_limits.left_lon + map_limits.right_lon)
            signed_h = float(np.clip(2.0 * (lon - center_lon) / lon_extent, -1.0, 1.0))
        if lat_extent > 0:
            center_lat = 0.5 * (map_limits.bottom_lat + map_limits.top_lat)
            signed_v = float(np.clip(2.0 * (lat - center_lat) / lat_extent, -1.0, 1.0))

        if lon_extent > 0 and lat_extent > 0:
            # Geographic yaw convention: 0deg = North, 90deg = East
            # (see physics.flying_objects), so velocity decomposes as:
            yaw_rad = math.radians(getattr(unit, "yaw_deg", 0.0))
            pitch_rad = math.radians(getattr(unit, "pitch_deg", 0.0))
            horiz_speed = float(getattr(unit, "speed", 0.0)) * math.cos(pitch_rad)
            v_north = horiz_speed * math.cos(yaw_rad)  # toward +lat
            v_east = horiz_speed * math.sin(yaw_rad)  # toward +lon

            # Normalized distance and outward-closing speed (m/s) for each edge.
            # Outward closing > 0 means the velocity points toward that edge.
            edges = (
                ((lon - map_limits.left_lon) / lon_extent, -v_east),  # left
                ((map_limits.right_lon - lon) / lon_extent, v_east),  # right
                ((lat - map_limits.bottom_lat) / lat_extent, -v_north),  # bottom
                ((map_limits.top_lat - lat) / lat_extent, v_north),  # top
            )
            _, closing_mps = min(edges, key=lambda e: e[0])
            max_speed = float(getattr(unit, "max_speed_mps", 0.0)) or 600.0
            closing_norm = float(np.clip(closing_mps / max_speed, -1.0, 1.0))

        return [signed_h, signed_v, closing_norm]

    def _get_altitude_boundary_features(self, unit) -> list:
        """Altitude-boundary cues (2 elements), the vertical analogue of the
        horizontal boundary features:
          - signed altitude in box [-1,1] (-1=floor, +1=ceiling); also a
            monotonic, well-conditioned stand-in for raw altitude
          - closing rate toward the nearest floor/ceiling, normalized by max
            speed, in [-1,1] (>0 = descending into the floor / climbing into the
            ceiling, whichever is nearer).

        Floor/ceiling are taken from the aircraft clamp bounds, falling back to
        map limits then a default span. Returns zeros if no span is resolvable.
        """
        signed_alt = 0.0
        alt_closing = 0.0
        alt = float(getattr(unit.position, "alt", 0.0))

        min_alt = getattr(unit, "min_alt_m", None)
        max_alt = getattr(unit, "max_alt_m", None)
        if min_alt is None or max_alt is None:
            map_limits = getattr(self.simulator, "map_limits", None)
            if map_limits is not None:
                min_alt = getattr(map_limits, "min_alt", min_alt)
                max_alt = getattr(map_limits, "max_alt", max_alt)
        if min_alt is None or max_alt is None or max_alt <= min_alt:
            return [signed_alt, alt_closing]

        extent = max_alt - min_alt
        center = 0.5 * (min_alt + max_alt)
        signed_alt = float(np.clip(2.0 * (alt - center) / extent, -1.0, 1.0))

        pitch_rad = math.radians(getattr(unit, "pitch_deg", 0.0))
        v_up = float(getattr(unit, "speed", 0.0)) * math.sin(pitch_rad)  # + = climbing
        dist_floor = (alt - min_alt) / extent
        dist_ceiling = (max_alt - alt) / extent
        # Closing toward the nearer altitude bound (>0 = approaching it).
        closing_mps = -v_up if dist_floor <= dist_ceiling else v_up
        max_speed = float(getattr(unit, "max_speed_mps", 0.0)) or 600.0
        alt_closing = float(np.clip(closing_mps / max_speed, -1.0, 1.0))

        return [signed_alt, alt_closing]
