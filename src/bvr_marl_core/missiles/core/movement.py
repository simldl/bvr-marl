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
            tick_secs,
        )
        m.position.lat, m.position.lon, m.position.alt = lat, lon, alt
        m.speed = min(max(spd, 0.0), m.max_speed_mps)
        m.yaw_deg = new_yaw
        m.pitch_deg = new_pitch
        m.roll_deg = new_roll
        m.position = self._clamp_position(m.position)

    def _clamp_position(self, pos):
        m = self.missile
        limits = m.map_limits
        pos.lat = min(max(pos.lat, limits.bottom_lat), limits.top_lat)
        pos.lon = min(max(pos.lon, limits.left_lon), limits.right_lon)
        pos.alt = min(max(pos.alt, limits.min_alt), limits.max_alt)
        return pos
