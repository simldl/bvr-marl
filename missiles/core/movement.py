import numpy as np

class MissileMovement:
    def __init__(self, missile, physics):
        self.missile = missile
        self.physics = physics

    def update(self, tick_secs):
        m = self.missile
        lat, lon, alt, spd, new_yaw, new_pitch, new_roll = self.physics.compute_movement(
            m.position,
            m.yaw_deg,
            m.desired_yaw_deg,
            m.pitch_deg,
            m.desired_pitch_deg,
            m.speed,
            1.0,
            tick_secs
        )
        m.position.lat, m.position.lon, m.position.alt = lat, lon, alt
        m.speed = np.clip(spd, 0, m.max_speed_mps)
        m.yaw_deg = new_yaw
        m.pitch_deg = new_pitch
        m.roll_deg = new_roll
        m.position = self._clamp_position(m.position)

    def _clamp_position(self, pos):
        m = self.missile
        pos.lat = np.clip(pos.lat, m.map_limits.bottom_lat, m.map_limits.top_lat)
        pos.lon = np.clip(pos.lon, m.map_limits.left_lon, m.map_limits.right_lon)
        pos.alt = np.clip(pos.alt, m.map_limits.min_alt, m.map_limits.max_alt)
        return pos