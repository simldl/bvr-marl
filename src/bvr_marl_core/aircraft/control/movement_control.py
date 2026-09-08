from bvr_marl_core.physics.flight_controller import LiftVectorAutopilot
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
        #: Positions actually swept during the most recent tick, oldest first.
        #: Consumed by missile guidance and CPA resolution so they walk the real
        #: arc instead of reconstructing one from the tick's two endpoints.
        self.tick_path: list[tuple[float, float, float]] = []
        #: Inner loop. Dormant until an outer loop commands a lift vector, so
        #: the scripted desired-heading callers keep their previous behaviour.
        self.autopilot = LiftVectorAutopilot(parent)

    def set_lift_vector(
        self,
        n_cmd: float,
        phi_cmd_deg: float,
        s_factor: float = 1.0,
        c_factor: float = 1.0,
    ) -> None:
        """Command a lift vector for the coming control tick."""
        self.autopilot.set_lift_vector(n_cmd, phi_cmd_deg, s_factor, c_factor)

    def set_yaw_deg(self, new_yaw_deg: float):
        self.desired_yaw_deg = normalize_angle(new_yaw_deg)

    def set_pitch_deg(self, new_pitch_deg: float):
        self.desired_pitch_deg = clamp_pitch_deg(float(new_pitch_deg))

    def set_throttle(self, throttle: float):
        self.throttle = min(max(throttle, 0.0), 1.0)

    def _check_boundary_violation(self, position):
        """Check if aircraft has violated lat/lon boundaries (not altitude)."""
        m = self.parent
        lat_violated = position.lat < m.map_limits.bottom_lat or position.lat > m.map_limits.top_lat
        lon_violated = position.lon < m.map_limits.left_lon or position.lon > m.map_limits.right_lon
        return lat_violated or lon_violated

    def _clamp_position_to_boundary(self, position):
        pos = position
        m = self.parent
        if getattr(m, "keep_inside_boundary", False):
            pos.lat = min(max(pos.lat, m.map_limits.bottom_lat), m.map_limits.top_lat)
            pos.lon = min(max(pos.lon, m.map_limits.left_lon), m.map_limits.right_lon)
        pos.alt = min(max(pos.alt, m.min_alt_m), m.max_alt_m)
        return pos

    def update_movement(self, tick_secs: float):
        """Advance the airframe across one control tick.

        The outer loop's command is held fixed for the whole tick; the inner
        loop is integrated in sub-steps beneath it. At a one-second tick a jet
        turning 17 deg/s would otherwise be flown as a single 300 m straight
        chord through what is physically an arc.
        """
        n_sub = self.autopilot.substep_count(tick_secs) if self.autopilot.active else 1
        dt_sub = tick_secs / n_sub
        self.tick_path = [self._pose()]
        for _ in range(n_sub):
            self._integrate_substep(dt_sub)
            self.tick_path.append(self._pose())
        self._update_boundary_state()

    def _pose(self) -> tuple[float, float, float]:
        p = self.parent.position
        return (float(p.lat), float(p.lon), float(p.alt))

    def _integrate_substep(self, dt_s: float):
        m = self.parent

        # Emergency energy override: when the aircraft is at the altitude floor AND its
        # speed has fallen below the minimum, the agent's throttle command may not provide
        # enough thrust to climb.  Force maximum throttle for this tick so the aircraft
        # can physically escape the floor.  The stored throttle (agent command) is unchanged.
        effective_throttle = self.throttle
        if m.position.alt <= m.min_alt_m and m.speed < m.min_speed_mps:
            effective_throttle = 1.0

        # Uncontrolled descent: a fuel flame-out OR a mortally-hit (dying) jet
        # produces no thrust and spirals down, to be removed at the floor / at its
        # scheduled death. Energy protections (which would inject phantom recovery
        # thrust) are disabled, a nose-down glide is commanded, and the low-speed
        # floor is relaxed so it actually descends.
        fuel = getattr(m, "fuel", None)
        flamed_out = fuel is not None and fuel.enabled and fuel.flamed_out
        uncontrolled = flamed_out or bool(getattr(m, "is_mortally_hit", False))
        if uncontrolled:
            effective_throttle = 0.0
            m.physics.enable_energy_protections = False
            self.desired_pitch_deg = min(self.desired_pitch_deg, -10.0)

        # Run the inner loop first so the attitude targets this sub-step reflect
        # the bank and load factor the aircraft has actually reached by now.
        # Only the autopilot path passes the extra keywords, so the legacy
        # desired-heading call signature is untouched for every platform and
        # test double that has not been migrated to a lift-vector command.
        inner_loop_kwargs = {}
        if self.autopilot.active and not uncontrolled:
            desired_yaw, desired_pitch, omega_max = self.autopilot.advance(dt_s)
            self.set_yaw_deg(desired_yaw)
            self.set_pitch_deg(desired_pitch)
            inner_loop_kwargs = {
                "omega_max_deg_s": omega_max,
                "roll_deg_override": self.autopilot.phi_deg,
            }

        lat, lon, alt, spd, new_yaw, new_pitch, new_roll = m.physics.compute_movement(
            m.position,
            self.yaw_deg,
            self.desired_yaw_deg,
            self.pitch_deg,
            self.desired_pitch_deg,
            m.speed,
            effective_throttle,
            dt_s,
            **inner_loop_kwargs,
        )
        self.pitch_deg = new_pitch
        self.yaw_deg = new_yaw
        self.roll_deg = new_roll
        speed_floor = 30.0 if uncontrolled else m.min_speed_mps
        m.speed = min(max(spd, speed_floor), m.max_speed_mps)

        m.yaw_deg = new_yaw
        m.pitch_deg = new_pitch
        m.roll_deg = new_roll
        m.position.lat, m.position.lon, m.position.alt = lat, lon, alt

        # Burn fuel for the thrust actually commanded this sub-step, then update mass.
        if fuel is not None and fuel.enabled:
            thrust_N = m.physics.get_engine_force(m.speed, m.position.alt, effective_throttle)
            fuel.consume(thrust_N, dt_s)

        m.position = self._clamp_position_to_boundary(m.position)

    def _update_boundary_state(self):
        m = self.parent
        boundary_violated = self._check_boundary_violation(m.position)
        if getattr(m, "keep_inside_boundary", False) or getattr(m, "is_support_asset", False):
            m.boundary_violation_active = False
            m.boundary_violation_countdown = 0
            if getattr(m, "removal_reason", None) == "boundary_violation":
                m.removal_reason = None
        elif boundary_violated and not m.boundary_violation_active:
            m.boundary_violation_active = True
            m.boundary_violation_countdown = 5
            m.removal_reason = "boundary_violation"
