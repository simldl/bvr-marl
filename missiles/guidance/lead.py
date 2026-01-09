# missiles/guidance/lead.py
import math
import numpy as np
from missiles.guidance.base import BaseGuidanceMode
from missiles.guidance.direct import DirectPursuitGuidance
from radar.core.utils import geodetic_to_enu

class LeadInterceptGuidance(BaseGuidanceMode):
    """
    Constant-velocity lead pursuit.
    Solves (||r + v_t t|| = V_m t) for t >= 0 and aims at p_t + v_t t.
    Frame: ENU (x=E, y=N, z=U). Yaw is clockwise from North.
    """

    TGO_MIN = 0.2   # s
    TGO_MAX = 25.0  # s

    def compute(self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs):
        # Relative vector r (target - missile) in ENU
        r = np.asarray(geodetic_to_enu(
            target_position.lat,  target_position.lon,  target_position.alt,
            missile_position.lat, missile_position.lon, missile_position.alt
        ), dtype=float)
        # Target velocity from provider (ENU m/s); fallback → None
        v_t = None
        try:
            v_t = getattr(self.missile.target_provider, "get_guidance_velocity", lambda: None)()
            if v_t is not None:
                v_t = np.asarray(v_t, dtype=float).reshape(3)
        except Exception:
            v_t = None

        # If we don't have velocity, fall back to direct pursuit
        if v_t is None or not np.isfinite(v_t).all():
            return DirectPursuitGuidance(self.missile).compute(
                current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
            )

        V_m = float(max(getattr(self.missile, "speed", 0.0), 1.0))  # current missile speed

        # Solve (v_t·v_t - V_m^2) t^2 + 2 (r·v_t) t + r·r = 0
        a = float(np.dot(v_t, v_t) - V_m * V_m)
        b = float(2.0 * np.dot(r, v_t))
        c = float(np.dot(r, r))

        t = None
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                t = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                s = math.sqrt(disc)
                # pick the smallest positive root
                roots = [(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
                t = min([rt for rt in roots if rt > 0.0], default=None)

        if t is None:
            # Not fast enough or geometry infeasible → direct pursuit
            return DirectPursuitGuidance(self.missile).compute(
                current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
            )

        # Clamp t_go and compute intercept point in ENU
        t = max(self.TGO_MIN, min(self.TGO_MAX, float(t)))
        p_int = r + v_t * t

        # Map ENU vector to yaw/pitch setpoints:
        # yaw = atan2(E, N), pitch = atan2(U, sqrt(E^2+N^2))
        x, y, z = float(p_int[0]), float(p_int[1]), float(p_int[2])
        yaw_deg   = (math.degrees(math.atan2(x, y)) % 360.0)
        pitch_deg = math.degrees(math.atan2(z, max(1e-9, math.hypot(x, y))))
        # Clip pitch to numeric bounds (your movement model already limits rates)
        pitch_deg = max(min(pitch_deg, 89.0), -89.0)

        return yaw_deg, pitch_deg