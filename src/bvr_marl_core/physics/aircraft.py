from __future__ import annotations

import math
from dataclasses import dataclass, field

from bvr_marl_core.physics.afterburner import Afterburner, AfterburnerParams
from bvr_marl_core.physics.flying_objects import FlyingPhysics
from bvr_marl_core.physics.physics import PhysicsParams, get_speed_of_sound


class AircraftPhysics(FlyingPhysics):
    @dataclass
    class Params(PhysicsParams):
        aspect_ratio: float = 7.5
        oswald_e: float = 0.82
        max_speed_mps: float = 680.0
        n_max: float = 9.0
        cd0_lo: float = 0.009
        cd0_mid1: float = 0.012
        cd0_mid2: float = 0.036
        cd0_hi: float = 0.031
        mach_t1: float = 0.8
        mach_t2: float = 0.95
        mach_t3: float = 1.05
        mach_t4: float = 2.5
        stall0_mps: float = 56.6
        service_ceiling_m: float = 18_000
        engine_thrust_profile: dict = field(
            default_factory=lambda: {"full_thrust_speed": 154.3, "optimal_speed": 257.2}
        )

        # Maximum lift coefficient parameters (Mach-dependent)
        cl_max_low: float = 1.6  # CL_max at low speed
        cl_max_high: float = 0.8  # CL_max at high Mach
        cl_max_mach_transition: float = 0.8  # Mach number where transition begins

        # Overspeed limits (Mach and EAS)
        mach_ne: float = 2.2  # Never-exceed Mach number (thermal/structural)
        eas_ne_mps: float = 410.0  # Never-exceed EAS in m/s (~800 KCAS)

    def __init__(self, params: Params):
        super().__init__(params)
        self.cd0_lo = params.cd0_lo
        self.cd0_mid1 = params.cd0_mid1
        self.cd0_mid2 = params.cd0_mid2
        self.cd0_hi = params.cd0_hi
        self.mach_t1 = params.mach_t1
        self.mach_t2 = params.mach_t2
        self.mach_t3 = params.mach_t3
        self.mach_t4 = params.mach_t4

        self.cl_max_low = params.cl_max_low
        self.cl_max_high = params.cl_max_high
        self.cl_max_mach_transition = params.cl_max_mach_transition

        self.mach_ne = params.mach_ne
        self.eas_ne_mps = params.eas_ne_mps

        self.afterburner = Afterburner(
            air=self.air,
            mass_kg=self.mass_kg,
            g=self.g,
            engine_prof=getattr(self, "engine_prof", {}),
            speed_full_thrust=getattr(self, "speed_full_thrust", 340.0),
            get_speed_of_sound_fn=get_speed_of_sound,
            params=AfterburnerParams(
                enabled=True,
                thrust_mult=1.6,
                density_exp=0.8,
                min_mach=0.70,
                tau_on_s=0.8,
                tau_off_s=0.4,
                delta_cd0=0.002,
                sfc_mil_kgps_per_N=1.8e-5,
                sfc_ab_kgps_per_N=3.8e-5,
            ),
        )

    @staticmethod
    def smoothstep(t: float) -> float:
        u = max(0.0, min(1.0, t))
        return u * u * (3.0 - 2.0 * u)

    def get_base_drag_cd(self, v_mps: float, alt_m: float) -> float:
        """Mach-scheduled parasite drag, augmented by AB ΔCD0 when spooled in."""
        mach = v_mps / get_speed_of_sound(alt_m)
        if mach < self.mach_t1:
            base = self.cd0_lo
        elif mach < self.mach_t2:
            s = self.smoothstep((mach - self.mach_t1) / (self.mach_t2 - self.mach_t1))
            base = self.cd0_lo + (self.cd0_mid1 - self.cd0_lo) * s
        elif mach < self.mach_t3:
            s = self.smoothstep((mach - self.mach_t2) / (self.mach_t3 - self.mach_t2))
            base = self.cd0_mid1 + (self.cd0_mid2 - self.cd0_mid1) * s
        elif mach < self.mach_t4:
            s = self.smoothstep((mach - self.mach_t3) / (self.mach_t4 - self.mach_t3))
            base = self.cd0_mid2 - (self.cd0_mid2 - self.cd0_hi) * s
        else:
            base = self.cd0_hi
        if hasattr(self, "afterburner") and self.afterburner is not None:
            return self.afterburner.effective_cd0(base)
        return base

    def get_engine_force(self, v_mps: float, alt_m: float, throttle: float) -> float:
        return self.afterburner.engine_force(v_mps, alt_m, throttle)

    def get_pitch_limit_deg(self, pos, speed_mps) -> float:
        phi_cap = 60.0
        if pos is None:
            return phi_cap
        phi_ceil = self._service_ceiling_pitch_limit(pos.alt)
        return min(phi_cap, max(0.0, phi_ceil))

    def get_ca_max(self, v_mps: float, alt_m: float) -> float:
        a = get_speed_of_sound(alt_m)
        mach = v_mps / max(a, 1e-6)
        if mach < 0.85:
            return 1.7
        elif mach < 1.2:
            return 1.3
        else:
            return 1.1

    def get_cl_max(self, v_mps: float, alt_m: float) -> float:
        M = v_mps / max(get_speed_of_sound(alt_m), 1e-6)
        CL_sub, CL_trans, CL_sup = 1.7, 1.3, 1.1

        def s(x, x0, w):
            return 1.0 / (1.0 + math.exp(-(x - x0) / w))

        w1 = 1.0 - s(M, 0.85, 0.04)
        w2 = s(M, 0.85, 0.04) * (1.0 - s(M, 1.20, 0.06))
        w3 = s(M, 1.20, 0.06)
        return w1 * CL_sub + w2 * CL_trans + w3 * CL_sup

    def compute_instantaneous_load_factor(self, v_mps: float, alt_m: float) -> float:
        """
        Compute instantaneous load factor limit from aerodynamic and structural constraints
        n_max_phys = min(n_aero, n_struct)
        where n_aero = (q * S * CL_max) / W
        """
        if v_mps < 1e-3:
            return 1.0

        rho = self.air.get_density(alt_m)
        q = 0.5 * rho * v_mps**2
        cl_max = self.get_cl_max(v_mps, alt_m)
        n_aero = (q * self.A_m2 * cl_max) / (self.mass_kg * self.g)
        n_max_phys = min(n_aero, self.n_max)
        return max(1.0, n_max_phys)

    def compute_sustained_load_factor(
        self, v_mps: float, alt_m: float, throttle: float = 1.0
    ) -> float:
        """
        Compute sustained load factor limit from thrust/drag considerations
        n_prop = sqrt(max(0, ((T_avail - q*S*CD0) * q*S) / (k * W^2)))
        n_sust = min(n_max_phys, n_prop)
        """
        if v_mps < 1e-3:
            return 1.0

        rho = self.air.get_density(alt_m)
        q = 0.5 * rho * v_mps**2
        n_max_phys = self.compute_instantaneous_load_factor(v_mps, alt_m)
        T_avail = self.get_engine_force(v_mps, alt_m, throttle)
        CD0 = self.get_base_drag_cd(v_mps, alt_m)
        k = self.K_ind
        W = self.mass_kg * self.g
        numerator = (T_avail - q * self.A_m2 * CD0) * q * self.A_m2
        denominator = k * W**2
        if numerator <= 0 or denominator <= 0:
            n_prop = 1.0
        else:
            n_prop = math.sqrt(numerator / denominator)
        return min(n_max_phys, max(1.0, n_prop))

    def compute_instantaneous_turn_rate(self, v_mps: float, alt_m: float) -> float:
        """
        Compute maximum instantaneous turn rate (deg/s)
        χ̇_max_inst = (g/V) * sqrt(n_max_phys^2 - 1)
        """
        if v_mps < 1e-3:
            return 0.0

        n_max_phys = self.compute_instantaneous_load_factor(v_mps, alt_m)
        if n_max_phys <= 1.0:
            return 0.0
        omega_rad = (self.g / v_mps) * math.sqrt(n_max_phys**2 - 1.0)
        return math.degrees(omega_rad)

    def compute_sustained_turn_rate(
        self, v_mps: float, alt_m: float, throttle: float = 1.0
    ) -> float:
        """
        Compute maximum sustained turn rate (deg/s)
        χ̇_max_sust = (g/V) * sqrt(n_sust^2 - 1)
        """
        if v_mps < 1e-3:
            return 0.0

        n_sust = self.compute_sustained_load_factor(v_mps, alt_m, throttle)
        if n_sust <= 1.0:
            return 0.0
        omega_rad = (self.g / v_mps) * math.sqrt(n_sust**2 - 1.0)
        return math.degrees(omega_rad)
