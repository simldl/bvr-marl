from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bvr_marl_core.physics.flying_objects import FlyingPhysics
from bvr_marl_core.physics.physics import PhysicsParams, get_speed_of_sound


class MissilePhysics(FlyingPhysics):
    @dataclass
    class Params(PhysicsParams):
        aspect_ratio: float = 3.0
        oswald_e: float = 0.75
        n_max: float = 30.0
        max_speed_mps: float = 1440.0
        cd0_s1: float = 0.08
        cd0_s2: float = 0.33
        cd0_s3: float = 0.30
        cd0_s4: float = 0.23
        cd0_hi: float = 0.17
        mach_t1: float = 1.1
        mach_t2: float = 1.6
        mach_t3: float = 2.4
        mach_t4: float = 3.2
        constant_engine_F: float = 20000.0
        engine_thrust_profile: dict = field(
            default_factory=lambda: {"full_thrust_speed": 1030.0, "optimal_speed": 1280.0}
        )
        stall0_mps: float = 129.0
        service_ceiling_m: float | None = None

    def __init__(self, params: Params):
        super().__init__(params)
        self.cd0_s1 = params.cd0_s1
        self.cd0_s2 = params.cd0_s2
        self.cd0_s3 = params.cd0_s3
        self.cd0_s4 = params.cd0_s4
        self.cd0_hi = params.cd0_hi
        self.mach_t1 = params.mach_t1
        self.mach_t2 = params.mach_t2
        self.mach_t3 = params.mach_t3
        self.mach_t4 = params.mach_t4
        self.constant_engine_F = params.constant_engine_F

        # IMPORTANT: missiles keep legacy behavior (no aircraft energy protections)
        self.enable_energy_protections = False

    @staticmethod
    def smoothstep(t: float) -> float:
        u = max(0.0, min(1.0, t))
        return u * u * (3.0 - 2.0 * u)

    def get_base_drag_cd(self, v_mps: float, alt_m: float) -> float:
        mach = v_mps / get_speed_of_sound(alt_m)
        if mach < self.mach_t1:
            s = self.smoothstep(mach / self.mach_t1)
            return self.cd0_s1 + (self.cd0_s2 - self.cd0_s1) * s
        elif mach < self.mach_t2:
            s = self.smoothstep((mach - self.mach_t1) / (self.mach_t2 - self.mach_t1))
            return self.cd0_s2 - (self.cd0_s2 - self.cd0_s3) * s
        elif mach < self.mach_t3:
            s = self.smoothstep((mach - self.mach_t2) / (self.mach_t3 - self.mach_t2))
            return self.cd0_s3 - (self.cd0_s3 - self.cd0_s4) * s
        elif mach < self.mach_t4:
            s = self.smoothstep((mach - self.mach_t3) / (self.mach_t4 - self.mach_t3))
            return self.cd0_s4 - (self.cd0_s4 - self.cd0_hi) * s
        else:
            # Linear extrapolation beyond mach_t4, floored at half cd0_hi to prevent negative drag.
            return max(self.cd0_hi * 0.5, self.cd0_hi - 0.01 * (mach - self.mach_t4))

    def get_engine_force(self, v_mps: float, alt_m: float, throttle: float) -> float:
        return self.constant_engine_F

    def get_pitch_limit_deg(self, pos, speed_mps) -> float:
        return 75.0

    def get_ca_max(self, v_mps: float, alt_m: float) -> float:
        # speed of sound a(alt); rho(alt) is used elsewhere for n(M)
        a = get_speed_of_sound(alt_m)
        M = max(v_mps / max(a, 1e-6), 0.0)

        # Regime anchors (AMRAAM-like, fin-tail, clean):
        C_sub = 2.9  # strong subsonic authority (AGARD/APL-consistent)
        C_low = 1.65  # low-supersonic (M ~1–2.5)
        C_high = 1.20  # high-supersonic (M >~ 3–4)

        # Smooth step helper: s(x) ~ 0..1 over width w centered at x0
        def s(x, x0, w):
            # logistic with small width gives a crisp but C¹-smooth transition
            from math import exp

            return 1.0 / (1.0 + exp(-(x - x0) / max(w, 1e-6)))

        # Blend weights across two transitions near M≈0.9 and M≈2.7
        w1 = 1.0 - s(M, 0.9, 0.15)  # subsonic weight
        w2 = s(M, 0.9, 0.15) * (1.0 - s(M, 2.7, 0.25))  # low-supersonic weight
        w3 = s(M, 2.7, 0.25)  # high-supersonic weight

        # Hard floor for very low M: no meaningful authority < ~0.2
        if M < 0.20:
            return 0.0

        # Optional gentle ramp from 0.20 → 0.60 to avoid a jump
        if 0.20 <= M < 0.60:
            # linear ramp to the subsonic plateau C_sub at M=0.60
            return C_sub * (M - 0.20) / (0.60 - 0.20)

        # Smooth blend of the three regimes
        return w1 * C_sub + w2 * C_low + w3 * C_high
