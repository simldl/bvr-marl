from __future__ import annotations

import math

from air_to_air_rl.physics.physics import (
    AIRSPEED_OF_SOUND,
    BasePhysics,
    ForceModel,
    PhysicsParams,
    get_speed_of_sound,
)
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.utils.angles import (
    normalize_angle,
    signed_yaw_deg_diff,
    yaw_geo_to_math,
    yaw_math_to_geo,
)
from air_to_air_rl.simulator.utils.geodesics import geodetic_direct


class FlyingPhysics(BasePhysics):
    def __init__(self, params: PhysicsParams):
        super().__init__(params)
        self.mass_kg = params.mass_kg
        self.A_m2 = params.reference_area_m2
        self.g = params.gravity_m_s2
        self.air = params.air
        self.cd0 = params.cd0
        self.AR = params.aspect_ratio
        self.eps = params.oswald_e
        self.K_ind = 1.0 / (math.pi * self.AR * self.eps)
        self.max_speed = params.max_speed_mps
        self.n_max = params.n_max
        self.engine_prof = params.engine_thrust_profile
        self.descent_multiplier = params.descent_multiplier
        self.stall0_mps = params.stall0_mps
        self.service_ceiling = params.service_ceiling_m
        self.stall_sink_pitch = params.stall_sink_pitch_deg
        self.speed_full_thrust = params.speed_full_thrust

        self.enable_energy_protections = True
        self.stall_margin_mps = getattr(params, "stall_margin_mps", 20.0)
        self.energy_relief_gain = getattr(params, "energy_relief_gain", 2.0)

        self.in_stall_recovery: bool = False
        self.stall_recovery_start_time: float = 0.0
        self.speed_at_stall_entry: float = 0.0
        self.recovery_time_required: float = getattr(params, "stall_recovery_min_time_s", 1.5)

        self.ground_clearance_taper_m = getattr(params, "ground_clearance_taper_m", 500.0)
        self._elapsed_time_s: float = 0.0

        self.force_model = ForceModel(
            mass_kg=self.mass_kg,
            reference_area_m2=self.A_m2,
            gravity=self.g,
            air=self.air,
            cd0=self.cd0,
            K_ind=self.K_ind,
            get_base_drag_cd=self.get_base_drag_cd,
            get_engine_force=self.get_engine_force,
        )

    def get_engine_force(self, v_mps: float, alt_m: float, throttle: float) -> float:
        return 0.0

    def get_base_drag_cd(self, v_mps: float, alt_m: float) -> float:
        return self.cd0

    def get_pitch_limit_deg(self, pos, speed_mps) -> float:
        return 90.0

    def get_ca_max(self, v_mps: float, alt_m: float) -> float:
        return getattr(self, "CA_max", 1.2)

    def _engine_speed_factor(self, v_mps: float) -> float:
        v1 = self.engine_prof.get("full_thrust_speed", 0.0)
        if v_mps <= v1:
            fv = 1.0
        else:
            a = AIRSPEED_OF_SOUND
            raw = 1.0 + 0.7 * ((v_mps / a) ** 2 - (self.speed_full_thrust / a) ** 2)
            fv = max(1.0, min(raw, 2.4))
        return fv

    def update_yaw_deg(
        self,
        pos: Position,
        yaw_deg,
        desired_yaw_deg,
        speed_mps,
        dt,
        throttle=1.0,
        use_sustained=False,
    ) -> tuple[float, float]:
        """
        Update yaw using physics-based turn rate modeling.

        Args:
            pos: Position (for altitude-dependent calculations)
            yaw_deg: Current yaw angle (degrees)
            desired_yaw_deg: Desired yaw angle (degrees)
            speed_mps: True airspeed (m/s)
            dt: Time step (s)
            throttle: Throttle setting [0,1] (for sustained turn rate)
            use_sustained: If True, use sustained turn rate; if False, use instantaneous

        Returns:
            tuple of (new_yaw_deg, omega_rad)
        """
        if speed_mps < 1e-3:
            return yaw_deg, 0.0

        # Use physics-based turn rate calculation if available (for AircraftPhysics)
        if hasattr(self, "compute_instantaneous_turn_rate") and hasattr(
            self, "compute_sustained_turn_rate"
        ):
            if use_sustained:
                omega_max_deg = self.compute_sustained_turn_rate(speed_mps, pos.alt, throttle)
            else:
                omega_max_deg = self.compute_instantaneous_turn_rate(speed_mps, pos.alt)
        else:
            g = self.g
            S = self.A_m2
            m = self.mass_kg
            W = m * g
            n_max = self.n_max

            rho = self.air.get_density(pos.alt)
            CA_max = self.get_ca_max(speed_mps, pos.alt)
            n_aero = 0.5 * rho * speed_mps**2 * S * CA_max / max(W, 1e-6)
            n = min(n_aero, n_max)

            if n < 1.0:
                return yaw_deg, 0.0

            omega_max_rad = g * math.sqrt(n**2 - 1) / speed_mps
            omega_max_deg = math.degrees(omega_max_rad)

        if omega_max_deg <= 0.0:
            return yaw_deg, 0.0

        delta = signed_yaw_deg_diff(yaw_deg, desired_yaw_deg)
        max_delta = omega_max_deg * dt

        if abs(delta) <= max_delta:
            omega_rad = math.radians(signed_yaw_deg_diff(desired_yaw_deg, yaw_deg)) / dt
            return desired_yaw_deg, omega_rad

        new_yaw_deg = normalize_angle(yaw_deg + max_delta if delta > 0 else yaw_deg - max_delta)
        omega_rad = math.radians(signed_yaw_deg_diff(new_yaw_deg, yaw_deg)) / dt
        return new_yaw_deg, omega_rad

    def update_pitch_deg(self, pitch_deg, desired_pitch_deg, speed_mps, dt, pos=None):
        if pos is not None:
            pitch_limit = self.get_pitch_limit_deg(pos, speed_mps)
            pitch_floor = self._ground_proximity_pitch_floor(pos.alt)
        else:
            pitch_limit = self.get_pitch_limit_deg(None, speed_mps)
            pitch_floor = -pitch_limit
        limited_desired_pitch = max(min(desired_pitch_deg, pitch_limit), pitch_floor)

        max_pitch_rate_deg_s = getattr(self, "max_pitch_rate_deg_s", 20.0)
        delta = limited_desired_pitch - pitch_deg
        max_delta = max_pitch_rate_deg_s * dt
        if abs(delta) <= max_delta:
            return limited_desired_pitch
        return pitch_deg + (max_delta if delta > 0 else -max_delta)

    def update_roll(self, speed_mps: float, omega_rad: float, pitch_deg: float) -> float:
        pitch_rad = math.radians(pitch_deg)
        return math.degrees(math.atan((speed_mps * omega_rad * math.cos(pitch_rad)) / self.g))

    def _dynamic_stall(self, alt_m: float) -> float:
        if self.stall0_mps is None:
            return float("inf")
        rho_rel = self.air.rho0 / max(self.air.get_density(alt_m), 1e-4)
        return self.stall0_mps * math.sqrt(rho_rel)

    def is_in_stall_condition(
        self, speed_mps: float, alt_m: float, load_factor: float = 1.0
    ) -> bool:
        """
        Determine if aircraft is in stall condition.
        Stall occurs when:
        1. Speed drops below dynamic stall speed adjusted for load factor
        2. Or when load factor demand exceeds available lift at current speed
        """
        if self.stall0_mps is None:
            return False

        # Dynamic stall speed increases with load factor (square root relation)
        v_stall_dynamic = self._dynamic_stall(alt_m)
        v_stall_adjusted = v_stall_dynamic * math.sqrt(abs(load_factor))

        return speed_mps < v_stall_adjusted

    def should_enter_stall_recovery(
        self, speed_mps: float, alt_m: float, load_factor: float = 1.0, current_time: float = 0.0
    ) -> bool:
        """Determine if stall recovery should be initiated."""
        if self.in_stall_recovery:
            return False  # Already in recovery

        # Enter recovery if in stall condition
        if self.is_in_stall_condition(speed_mps, alt_m, load_factor):
            self.in_stall_recovery = True
            self.stall_recovery_start_time = current_time
            self.speed_at_stall_entry = speed_mps
            return True

        return False

    def should_exit_stall_recovery(
        self, speed_mps: float, alt_m: float, current_time: float = 0.0
    ) -> bool:
        """Determine if stall recovery should be exited."""
        if not self.in_stall_recovery:
            return False

        # Minimum recovery time requirement
        recovery_duration = current_time - self.stall_recovery_start_time
        if recovery_duration < self.recovery_time_required:
            return False

        # Speed recovery requirement (must be well above stall + margin)
        v_stall = self._dynamic_stall(alt_m)
        speed_margin = 1.2  # 20% margin above stall speed
        min_recovery_speed = v_stall * speed_margin

        # Also require speed to be improving from stall entry
        speed_improvement = speed_mps > (self.speed_at_stall_entry * 1.1)

        if speed_mps >= min_recovery_speed and speed_improvement:
            self.in_stall_recovery = False
            return True

        return False

    def compute_stall_recovery_controls(
        self, current_pitch_deg: float, current_throttle: float
    ) -> tuple[float, float]:
        """
        Compute automatic stall recovery control inputs.
        Returns: (recovery_pitch_deg, recovery_throttle)
        """
        if not self.in_stall_recovery:
            return current_pitch_deg, current_throttle

        # Aggressive pitch down for recovery
        recovery_pitch = min(current_pitch_deg, -15.0)  # Force nose down at least 15 degrees

        # Maximum throttle for recovery
        recovery_throttle = 1.0

        return recovery_pitch, recovery_throttle

    def _service_ceiling_pitch_limit(self, alt_m: float) -> float:
        """
        Ceiling taper: φ_ceil(h) according to specification
        """
        if self.service_ceiling is None:
            return 90.0

        h = alt_m
        h_ceiling = self.service_ceiling

        if h <= 0.8 * h_ceiling:
            return 90.0
        elif h >= h_ceiling:
            return 0.0
        else:
            # Linear taper: 90° * (h_ceiling - h) / (0.2 * h_ceiling)
            return 90.0 * (h_ceiling - h) / (0.2 * h_ceiling)

    def _ground_proximity_pitch_floor(self, alt_m: float) -> float:
        """
        Floor pitch limit: prevents nose-down pitch near the ground.
        Mirrors _service_ceiling_pitch_limit for the lower boundary.
        Returns the minimum (most negative) allowed pitch in degrees.
        """
        taper = self.ground_clearance_taper_m
        if taper <= 0.0 or alt_m >= taper:
            return -90.0  # No restriction at altitude
        elif alt_m <= 0.0:
            return 0.0  # No negative pitch at/below ground level
        else:
            return -90.0 * (alt_m / taper)  # Linear taper: 0° at ground → -90° at taper altitude

    def _update_orientation(
        self,
        pos: Position,
        pitch_deg: float,
        desired_pitch_deg: float,
        yaw_deg: float,
        desired_yaw_deg: float,
        speed_mps: float,
        dt: float,
    ) -> tuple[float, float, float]:
        pitch_limit = self._service_ceiling_pitch_limit(pos.alt)
        pitch_deg = self.update_pitch_deg(
            pitch_deg, min(desired_pitch_deg, pitch_limit), speed_mps, dt, pos
        )
        yaw_deg, omega_rad = self.update_yaw_deg(pos, yaw_deg, desired_yaw_deg, speed_mps, dt)
        return pitch_deg, yaw_deg, omega_rad

    def _compute_motion_vectors(
        self, pitch_rad: float, yaw_deg_geo: float, speed_mps: float, dt: float
    ):
        """
        Convert geographic yaw -> mathematical, compute motion, then return a
        geographic ground track for geodetic projection.
        """
        # Convert: geographic (0°=N, 90°=E) -> math (0°=E, 90°=N)
        psi_math_deg = yaw_geo_to_math(yaw_deg_geo)
        psi = math.radians(psi_math_deg)
        theta = pitch_rad

        v = speed_mps
        vx = v * math.cos(theta) * math.cos(psi)
        vy = v * math.cos(theta) * math.sin(psi)
        vz = v * math.sin(theta)

        dx = vx * dt
        dy = vy * dt
        dz = vz * dt

        # Track angle is currently mathematical → convert back to geographic
        ground_track_math_deg = math.degrees(math.atan2(vy, vx))
        ground_track_geo_deg = yaw_math_to_geo(normalize_angle(ground_track_math_deg))
        ground_dist = math.hypot(dx, dy)
        return ground_track_geo_deg, ground_dist, dz

    def _integrate_energy_and_speed(
        self,
        pos: Position,
        alt2: float,
        speed_mps: float,
        pitch_rad: float,
        n: float,
        throttle: float,
        dt: float,
    ) -> float:
        Ps = self.force_model.compute_specific_energy_rate(
            speed_mps, 0.5 * (pos.alt + alt2), n, throttle
        )
        h_e = pos.alt + speed_mps**2 / (2.0 * self.g) + Ps * dt
        h_e = max(0.0, h_e)
        return math.sqrt(max(0.0, 2.0 * self.g * (h_e - alt2)))

    def compute_movement(
        self,
        pos: Position,
        yaw_deg: float,
        desired_hdg_deg: float,
        pitch_deg: float,
        desired_pitch_deg: float,
        speed_mps: float,
        throttle: float,
        dt: float,
    ) -> tuple[float, float, float, float, float, float, float]:

        # Advance internal time for stall helpers
        self._elapsed_time_s += dt

        # --- Afterburner spool update (if available) ---
        if hasattr(self, "afterburner") and self.afterburner is not None:
            try:
                self.afterburner.update_spool(dt, speed_mps, pos.alt)
            except Exception:
                pass

        # 1) Apply ceiling taper + global cap and yaw/pitch slews (existing logic)
        pitch_deg, yaw_deg, omega_rad = self._update_orientation(
            pos, pitch_deg, desired_pitch_deg, yaw_deg, desired_hdg_deg, speed_mps, dt
        )

        # Load factor from commanded turn rate (for kinematics)
        n_turn = math.sqrt(1.0 + (speed_mps * omega_rad / self.g) ** 2)
        n_turn = min(self.n_max, max(1.0, n_turn))

        # Load factor for drag calculation (allows unloads n < 1 to reduce induced drag)
        n_drag = n_turn
        n_ext = getattr(self, "n_external", None)
        if n_ext is not None:
            # Use external n for drag without 1.0 floor, allowing push-overs to cut drag
            n_drag = min(self.n_max, max(0.1, float(n_ext)))  # Small floor to avoid division issues
            self.n_external = None  # one-shot consumption

        # --- Stall recovery: enter/maintain before moving, so motion reflects recovery attitude ---
        if getattr(self, "enable_energy_protections", True):
            # Decide entry using current state (speed at start of step, current altitude)
            _ = self.should_enter_stall_recovery(
                speed_mps=speed_mps,
                alt_m=pos.alt,
                load_factor=n_drag,
                current_time=self._elapsed_time_s,
            )
            # If we are in recovery now, override pitch & throttle
            if self.in_stall_recovery:
                pitch_deg, throttle = self.compute_stall_recovery_controls(
                    current_pitch_deg=pitch_deg, current_throttle=throttle
                )

        # Ground proximity floor: clamp after any override (stall recovery, guard band, etc.)
        # so the aircraft can never pitch nose-down into the ground.
        if getattr(self, "enable_energy_protections", True):
            pitch_floor = self._ground_proximity_pitch_floor(pos.alt)
            pitch_deg = max(pitch_deg, pitch_floor)

        pitch_rad = math.radians(pitch_deg)

        # 2) Motion vectors + geodetic projection (bearing already geographic now)
        ground_track, ground_dist, dz = self._compute_motion_vectors(
            pitch_rad, yaw_deg, speed_mps, dt
        )
        lat2, lon2, alt2 = geodetic_direct(pos.lat, pos.lon, pos.alt, ground_track, ground_dist, dz)

        # 3) Energy/speed integration
        v_new_mps = self._integrate_energy_and_speed(
            pos, alt2, speed_mps, pitch_rad, n_drag, throttle, dt
        )

        if getattr(self, "enable_energy_protections", True):
            v_stall = self._dynamic_stall(alt2)

            # Maintain recovery state and check for exit after we know v_new
            if self.in_stall_recovery:
                # Re-check exit using the post-integration speed and new altitude
                _ = self.should_exit_stall_recovery(
                    speed_mps=v_new_mps, alt_m=alt2, current_time=self._elapsed_time_s
                )
                # While still in recovery, force max throttle and ensure nose-down bias is respected
                if self.in_stall_recovery:
                    throttle = 1.0
                    # Optionally (light touch): bias pitch down a bit further each step toward sink
                    if pitch_deg > self.stall_sink_pitch:
                        pitch_deg = self.update_pitch_deg(
                            pitch_deg, self.stall_sink_pitch, speed_mps, dt, pos
                        )
                        pitch_rad = math.radians(pitch_deg)
                        v_new_mps = self._integrate_energy_and_speed(
                            pos, alt2, speed_mps, pitch_rad, n_drag, throttle, dt
                        )
            else:
                # Not in explicit recovery: apply the soft energy guard band near stall (paper Eq. 21)
                alt_mid = 0.5 * (pos.alt + alt2)
                Ps = self.force_model.compute_specific_energy_rate(
                    speed_mps, alt_mid, n_drag, throttle
                )
                stall_margin = self.stall_margin_mps
                if (Ps < 0.0) and math.isfinite(v_stall) and (speed_mps < v_stall + stall_margin):
                    throttle = 1.0
                    k_E = self.energy_relief_gain
                    phi_tgt = max(pitch_deg - k_E * (-Ps), self.stall_sink_pitch)
                    # Respect ceiling/global caps and rate limits via helper
                    pitch_deg = self.update_pitch_deg(pitch_deg, phi_tgt, speed_mps, dt, pos)
                    pitch_rad = math.radians(pitch_deg)
                    v_new_mps = self._integrate_energy_and_speed(
                        pos, alt2, speed_mps, pitch_rad, n_drag, throttle, dt
                    )

        # 4) Overspeed guards (Mach & EAS) — unchanged
        if getattr(self, "enable_energy_protections", True):
            alt_mid = 0.5 * (pos.alt + alt2)
            a_local = get_speed_of_sound(alt_mid)

            M_NE = getattr(self, "mach_ne", 2.2)  # thermal/structural Mach limit
            V_EAS_NE = getattr(self, "eas_ne_mps", 410.0)  # ~800 KCAS -> ~411 m/s

            # Convert EAS limit to TAS at this altitude
            rho = self.air.get_density(alt_mid)
            rho0 = self.air.rho0
            V_ne_mach = M_NE * a_local
            V_ne_qbar = V_EAS_NE / math.sqrt(max(rho / rho0, 1e-6))

            v_new_mps = min(v_new_mps, V_ne_mach, V_ne_qbar)

        # 5) Roll from turn rate & pitch (existing)
        roll_deg = self.update_roll(v_new_mps, omega_rad, pitch_deg)

        return lat2, lon2, alt2, v_new_mps, yaw_deg, pitch_deg, roll_deg
