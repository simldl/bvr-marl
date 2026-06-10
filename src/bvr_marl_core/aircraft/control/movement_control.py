import numpy as np

from bvr_marl_core.simulator.utils.angles import clamp_pitch_deg, normalize_angle


class AircraftControlSystem:
    def __init__(self, parent):
        self.parent = parent
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.yaw_deg = parent.yaw_deg
        self.throttle = 1.0
        self.desired_yaw_deg = parent.yaw_deg
        self.desired_pitch_deg = 0.0

    def set_yaw_deg(self, new_yaw_deg: float):
        self.desired_yaw_deg = normalize_angle(new_yaw_deg)

    def set_pitch_deg(self, new_pitch_deg: float):
        self.desired_pitch_deg = clamp_pitch_deg(float(new_pitch_deg))

    def set_throttle(self, throttle: float):
        self.throttle = np.clip(throttle, 0.0, 1.0)

    def _check_boundary_violation(self, position):
        """Check if aircraft has violated lat/lon boundaries (not altitude)."""
        m = self.parent
        lat_violated = position.lat < m.map_limits.bottom_lat or position.lat > m.map_limits.top_lat
        lon_violated = position.lon < m.map_limits.left_lon or position.lon > m.map_limits.right_lon
        return lat_violated or lon_violated

    def _clamp_position_to_boundary(self, position):
        pos = position
        m = self.parent
        pos.alt = np.clip(pos.alt, m.min_alt_m, m.max_alt_m)
        return pos

    def update_movement(self, tick_secs: float):
        m = self.parent

        # Emergency energy override: when the aircraft is at the altitude floor AND its
        # speed has fallen below the minimum, the agent's throttle command may not provide
        # enough thrust to climb.  Force maximum throttle for this tick so the aircraft
        # can physically escape the floor.  The stored throttle (agent command) is unchanged.
        effective_throttle = self.throttle
        if m.position.alt <= m.min_alt_m and m.speed < m.min_speed_mps:
            effective_throttle = 1.0

        lat, lon, alt, spd, new_yaw, new_pitch, new_roll = m.physics.compute_movement(
            m.position,
            self.yaw_deg,
            self.desired_yaw_deg,
            self.pitch_deg,
            self.desired_pitch_deg,
            m.speed,
            effective_throttle,
            tick_secs,
        )
        self.pitch_deg = new_pitch
        self.yaw_deg = new_yaw
        self.roll_deg = new_roll
        m.speed = np.clip(spd, m.min_speed_mps, m.max_speed_mps)

        m.yaw_deg = new_yaw
        m.pitch_deg = new_pitch
        m.roll_deg = new_roll
        m.position.lat, m.position.lon, m.position.alt = lat, lon, alt

        boundary_violated = self._check_boundary_violation(m.position)
        if boundary_violated and not m.boundary_violation_active:
            m.boundary_violation_active = True
            m.boundary_violation_countdown = 5
            m.removal_reason = "boundary_violation"

        m.position = self._clamp_position_to_boundary(m.position)
