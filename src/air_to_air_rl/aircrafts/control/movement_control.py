import numpy as np

from air_to_air_rl.simulator.utils.angles import normalize_angle


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
        self.desired_pitch_deg = normalize_angle(new_pitch_deg)

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
        lat, lon, alt, spd, new_yaw, new_pitch, new_roll = self.parent.physics.compute_movement(
            self.parent.position,
            self.yaw_deg,
            self.desired_yaw_deg,
            self.pitch_deg,
            self.desired_pitch_deg,
            self.parent.speed,
            self.throttle,
            tick_secs,
        )
        self.pitch_deg = new_pitch
        self.yaw_deg = new_yaw
        self.roll_deg = new_roll
        self.parent.speed = np.clip(spd, self.parent.min_speed_mps, self.parent.max_speed_mps)

        self.parent.yaw_deg = new_yaw
        self.parent.pitch_deg = new_pitch
        self.parent.roll_deg = new_roll
        self.parent.position.lat, self.parent.position.lon, self.parent.position.alt = lat, lon, alt

        boundary_violated = self._check_boundary_violation(self.parent.position)
        if boundary_violated and not self.parent.boundary_violation_active:
            self.parent.boundary_violation_active = True
            self.parent.boundary_violation_countdown = 5
            self.parent.removal_reason = "boundary_violation"

        self.parent.position = self._clamp_position_to_boundary(self.parent.position)
