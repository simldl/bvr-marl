# missiles/guidance/lead.py
import logging
import math

import numpy as np

from bvr_marl_core.missiles.guidance.base import BaseGuidanceMode
from bvr_marl_core.missiles.guidance.direct import DirectPursuitGuidance
from bvr_marl_core.radar.core.utils import geodetic_to_enu

logger = logging.getLogger(__name__)


class LeadInterceptGuidance(BaseGuidanceMode):
    """
    Constant-velocity lead pursuit.
    Solves (||r + v_t t|| = V_m t) for t >= 0 and aims at p_t + v_t t.
    Frame: ENU (x=E, y=N, z=U). Yaw is clockwise from North.

    Both r and v_t must be in the same ENU frame — missile-centred ENU.
    The preferred source is tracker_manager.get_target_state_for_pn() which
    returns a coherent (p, v) pair from the same KF snapshot, both already
    transformed to missile ENU.  The cached provider velocity is used as a
    fallback and is rotated into missile ENU via
    get_guidance_velocity_in_missile_enu().
    """

    TGO_MIN = 0.2  # s
    TGO_MAX = 25.0  # s
    _debug_vel_check = False  # enable finite-difference velocity validation

    def __init__(self, missile):
        super().__init__(missile)
        self._prev_p_t_enu = None  # previous tick's target position in missile ENU

    def compute(
        self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
    ):
        # Relative vector r (target - missile) in missile ENU; replaced by coherent tracker pos if available.
        r = np.asarray(
            geodetic_to_enu(
                target_position.lat,
                target_position.lon,
                target_position.alt,
                missile_position.lat,
                missile_position.lon,
                missile_position.alt,
            ),
            dtype=float,
        )

        # Coherent (p, v_t) in missile ENU from the same KF snapshot eliminates
        # frame skew between the geodetic position and the tracker velocity.
        v_t = None
        prov = getattr(self.missile, "target_provider", None)
        radar = getattr(self.missile, "radar", None)

        tid = prov.get_guidance_tid() if prov is not None else None
        if tid is None and prov is not None:
            get_coast_tid = getattr(prov, "get_coast_guidance_tid", None)
            if callable(get_coast_tid):
                try:
                    tid = get_coast_tid()
                except Exception:
                    tid = None
        if tid is not None and radar is not None:
            tm = getattr(radar, "tracker_manager", None)
            if tm is not None and hasattr(tm, "get_target_state_for_pn"):
                state_tuple = tm.get_target_state_for_pn(tid, missile_position)
                if (
                    state_tuple is not None
                    and isinstance(state_tuple, (tuple, list))
                    and len(state_tuple) >= 2
                ):
                    p_raw, v_raw = state_tuple[0], state_tuple[1]  # (p, v[, P6]) in missile ENU
                    if p_raw is not None:
                        p = np.asarray(p_raw, dtype=float).reshape(-1)
                        if (
                            p.size == 3
                            and np.isfinite(p).all()
                            and 1.0 < float(np.linalg.norm(p)) < 300_000.0
                        ):
                            r = p  # coherent position in missile ENU
                    if v_raw is not None:
                        v = np.asarray(v_raw, dtype=float).reshape(-1)
                        v_mag = float(np.linalg.norm(v))
                        if v.size == 3 and np.isfinite(v).all() and 0.0 < v_mag < 1400.0:
                            v_t = v

        if v_t is None and prov is not None:
            raw = getattr(prov, "get_guidance_velocity_in_missile_enu", lambda _: None)(
                missile_position
            )
            if raw is None:
                raw = getattr(prov, "get_guidance_velocity", lambda: None)()
            if raw is not None:
                v = np.asarray(raw, dtype=float).reshape(3)
                if np.isfinite(v).all():
                    v_t = v

        # Debug: finite-difference velocity cross-check (enabled via _debug_vel_check)
        if self._debug_vel_check and v_t is not None and self._prev_p_t_enu is not None:
            dt = float(tick_secs)
            if dt > 1e-9:
                v_hat = (r - self._prev_p_t_enu) / dt
                v_hat_mag = float(np.linalg.norm(v_hat))
                v_t_mag = float(np.linalg.norm(v_t))
                if v_t_mag > 10.0 and v_hat_mag > 10.0:
                    cos_angle = float(np.dot(v_hat, v_t) / (v_hat_mag * v_t_mag))
                    mag_ratio = v_hat_mag / max(v_t_mag, 1e-9)
                    if cos_angle < 0.7 or mag_ratio > 3.0 or mag_ratio < 0.33:
                        logger.debug(
                            "LeadInterceptGuidance: v_t vs FD divergence -- "
                            "cos=%.2f mag_ratio=%.2f |v_t|=%.1f |v_hat|=%.1f",
                            cos_angle,
                            mag_ratio,
                            v_t_mag,
                            v_hat_mag,
                        )
        self._prev_p_t_enu = r.copy()

        if v_t is None or not np.isfinite(v_t).all():
            return DirectPursuitGuidance(self.missile).compute(
                current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
            )

        V_m = float(max(getattr(self.missile, "speed", 0.0), 1.0))

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
            return DirectPursuitGuidance(self.missile).compute(
                current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
            )

        t = max(self.TGO_MIN, min(self.TGO_MAX, float(t)))
        p_int = r + v_t * t

        # yaw = atan2(E, N), pitch = atan2(U, sqrt(E^2+N^2))
        x, y, z = float(p_int[0]), float(p_int[1]), float(p_int[2])
        yaw_deg = math.degrees(math.atan2(x, y)) % 360.0
        pitch_deg = math.degrees(math.atan2(z, max(1e-9, math.hypot(x, y))))
        pitch_deg = max(min(pitch_deg, 89.0), -89.0)

        return yaw_deg, pitch_deg
