# Proportional Navigation implementation with phase-aware scheduling.
# Midcourse: True PN (TPN) or Pure PN (energy-conserving), N≈4
# Terminal:  ZEM shaping with higher N (e.g., 5.5–6)
#
# Frame discipline:
#   • All vectors are in the missile's ENU frame (at the missile position).
#   • Target velocity is fetched from tracker_manager.get_target_state_for_pn(tid, missile_ref)
#     or rotated from target ENU into missile ENU as a fallback.
#
# Yaw convention:
#   • Geodetic yaw: 0° = North, clockwise positive (East = +90°).
#   • Internal ENU computations use this convention consistently.
#
# Scheduling gate:
#   • ALTITUDE_THRESHOLD is used as a *slant-range gate* for terminal scheduling
#     (kept for backward compatibility with your constructor). See alias property
#     `terminal_range_m` for a clearer name.

import math
from typing import Optional

import numpy as np

from air_to_air_rl.radar.core.utils import geodetic_to_enu
from air_to_air_rl.simulator.utils.angles import normalize_angle


class PnPropNavGuidance:
    """
    PN/ZEM navigator with simple scheduling:
      phase = "pn"       → True/Pure PN with nominal N (≈4)
      phase = "terminal" → ZEM shaping with higher N (≈5.5–6)
      otherwise          → use constructor pn_law and N

    Returns crisp yaw/pitch setpoints by integrating commanded yaw/pitch rates.

    Notation:
      • LOS rate            Ω = (R × V_rel) / ||R||^2
      • True PN             a_cmd ∝ N * Vc * (Ω × û_LOS)
      • Pure PN (energy)    a_cmd ∝ N * Vc * (Ω × v̂_M)
      • ZEM shaping         a_cmd = N * ZEM_⊥ / t_go^2
      • Guidance preservation (lateral-only): project a_cmd to plane ⟂ v_M
    """

    PN_TRUE = 1
    PN_PURE = 2
    PN_ZEM = 3

    def __init__(
        self,
        missile,
        pn_n: float = 4.0,
        pn_law: int = PN_TRUE,
        altitude_threshold: float = 2000.0,  # kept for backward-compat; acts as slant-range gate
    ):
        self.missile = missile
        self.PN_N = float(pn_n)
        self.pn_law = int(pn_law)

        # NOTE: despite the historical name, this is used as a slant-range gate (meters).
        self.ALTITUDE_THRESHOLD = float(altitude_threshold)

        # Hold-last-valid target velocity (used when tracker temporarily unavailable)
        self._last_valid_vt: np.ndarray | None = None
        self._last_valid_vt_age: float = 0.0  # seconds since last fresh velocity
        self._HOLD_VT_MAX_AGE: float = 3.0  # drop held velocity after this many seconds

        self.g = 9.80665
        self.debug = False

    @property
    def terminal_range_m(self) -> float:
        """Alias for ALTITUDE_THRESHOLD; clearer name."""
        return self.ALTITUDE_THRESHOLD

    @terminal_range_m.setter
    def terminal_range_m(self, val: float) -> None:
        self.ALTITUDE_THRESHOLD = float(val)

    def compute(
        self,
        current_yaw_deg: float,
        current_pitch_deg: float,
        missile_position,
        target_position,
        tick_secs: float,
        phase: str | None = None,
    ):
        """
        Compute absolute (yaw_deg, pitch_deg) commands in geodetic yaw convention.
        Inputs:
          • current_yaw_deg, current_pitch_deg: missile attitude (deg)
          • missile_position, target_position: geodetic objects with lat/lon/alt
          • tick_secs: integration step [s]
          • phase: "pn" or "terminal" to force a mode; otherwise default law
        """

        Vm_vec = self._missile_velocity_vec(
            yaw_deg=current_yaw_deg,
            pitch_deg=current_pitch_deg,
            speed_mps=float(getattr(self.missile, "speed", 0.0)),
        )
        Vm_mag = max(float(np.linalg.norm(Vm_vec)), 1.0)
        Vmh = Vm_vec / Vm_mag

        # Coherent (p, v) from same KF snapshot eliminates position/velocity time skew.
        tracker_p, tracker_v = self._get_target_state_enu(missile_position, target_position)

        # Prefer tracker position (coherent with velocity); fall back to geodetic conversion.
        if tracker_p is not None:
            R_vec = tracker_p
        else:
            R_vec = np.asarray(
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
        R = max(float(np.linalg.norm(R_vec)), 1e-9)
        Ulos = self._unit(R_vec)

        # Hold-last-valid: avoids zero-velocity fallback (pure-pursuit for lateral movers).
        # Never finite-difference R across ENU frames (frame-inconsistent).
        if tracker_v is not None:
            self._last_valid_vt = tracker_v.copy()
            self._last_valid_vt_age = 0.0
            Vt_vec = tracker_v
        else:
            self._last_valid_vt_age += tick_secs
            if self._last_valid_vt is not None and self._last_valid_vt_age < self._HOLD_VT_MAX_AGE:
                Vt_vec = self._last_valid_vt
            else:
                Vt_vec = np.zeros(3, dtype=float)

        Vrel = Vt_vec - Vm_vec
        Vc = max(1e-6, -float(np.dot(Vrel, Ulos)))  # closing speed (positive if closing)
        Omega = np.cross(R_vec, Vrel) / (R * R)  # LOS rate [rad/s]

        if phase == "terminal" or R < self.ALTITUDE_THRESHOLD:
            law = self.PN_ZEM
            N_nav = max(self.PN_N, 5.5)  # bump N in terminal
        elif phase == "pn":
            law = self.pn_law
            N_nav = self.PN_N
        else:
            law = self.pn_law
            N_nav = self.PN_N

        if law == self.PN_ZEM:
            a_cmd = self._zem_guidance(R_vec, Vrel, Ulos, Vc, N_nav)
        elif law == self.PN_TRUE:
            a_cmd = N_nav * Vc * np.cross(Omega, Ulos)  # True PN
        elif law == self.PN_PURE:
            a_cmd = N_nav * Vc * np.cross(Omega, Vmh)  # Pure (energy-conserving) PN
        else:
            a_cmd = N_nav * Vc * np.cross(Omega, Ulos)  # default to True PN

        # Lateral-only guidance command (remove along-velocity component)
        a_cmd = self._lateral_only(a_cmd, Vmh)
        a_cmd = self._limit_acceleration(a_cmd)

        # Velocity-aligned frame: f (forward), u (up ⟂ f), l (left)
        k_world = np.array([0.0, 0.0, 1.0])
        f = Vmh
        u = k_world - np.dot(k_world, f) * f
        u_norm = np.linalg.norm(u)
        u = np.array([0.0, 0.0, 1.0]) if u_norm < 1e-9 else (u / u_norm)
        lateral = np.cross(u, f)  # left lateral

        a_lat = float(np.dot(a_cmd, lateral))  # lateral → yaw
        a_vert = float(np.dot(a_cmd, u))  # vertical → pitch

        # Small-angle mapping via velocity "nudge" (keeps signs consistent)
        vhat_p = [
            f[0] + (a_lat * lateral[0] + a_vert * u[0]) / Vm_mag * tick_secs,
            f[1] + (a_lat * lateral[1] + a_vert * u[1]) / Vm_mag * tick_secs,
            f[2] + (a_lat * lateral[2] + a_vert * u[2]) / Vm_mag * tick_secs,
        ]
        vn = max(1e-9, float(np.linalg.norm(vhat_p)))
        vhat_p = [vhat_p[0] / vn, vhat_p[1] / vn, vhat_p[2] / vn]

        # CW-from-North yaw: atan2(E, N)
        desired_yaw_deg = normalize_angle(math.degrees(math.atan2(vhat_p[0], vhat_p[1])))
        desired_pitch_deg = float(math.degrees(math.asin(max(-1.0, min(1.0, vhat_p[2])))))
        desired_pitch_deg = max(min(desired_pitch_deg, 89.0), -89.0)

        if self.debug and hasattr(self.missile, "elapsed_time_s"):
            t = float(getattr(self.missile, "elapsed_time_s"))
            if (t % 2.0) < tick_secs:
                print(f"[PN] t={t:6.2f}s R={R:7.1f}m Vc={Vc:6.1f} m/s law={law} N={N_nav:.1f}")

        return float(desired_yaw_deg), float(desired_pitch_deg)

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else np.zeros(3, dtype=float)

    def _time_to_go(self, R_vec: np.ndarray, Ulos: np.ndarray, Vrel: np.ndarray) -> float:
        Vc = max(1e-6, -float(np.dot(Vrel, Ulos)))
        return float(np.linalg.norm(R_vec)) / Vc

    def _zem_guidance(
        self, R_vec: np.ndarray, Vrel: np.ndarray, Ulos: np.ndarray, Vc: float, N: float
    ) -> np.ndarray:
        """
        Zero-Effort Miss normal to LOS with GCP applied later:
          ZEM = R + Vrel * t_go,  ZEM_⊥ = ZEM - (ZEM·û_LOS)û_LOS,
          a_cmd = N * ZEM_⊥ / t_go^2
        """
        tgo = self._time_to_go(R_vec, Ulos, Vrel)
        if tgo <= 1e-6:
            return np.zeros(3, dtype=float)
        ZEM = R_vec + Vrel * tgo
        ZEM_r = np.dot(ZEM, Ulos) * Ulos
        ZEM_n = ZEM - ZEM_r
        return (N * ZEM_n) / (tgo * tgo)

    def _lateral_only(self, a_cmd: np.ndarray, Vmh: np.ndarray) -> np.ndarray:
        """Project acceleration onto plane orthogonal to velocity."""
        return a_cmd - np.dot(a_cmd, Vmh) * Vmh

    def _limit_acceleration(self, a_cmd: np.ndarray) -> np.ndarray:
        n_max = float(getattr(self.missile, "n_max", 30.0))
        a_max = n_max * self.g
        mag = float(np.linalg.norm(a_cmd))
        return (a_max / mag) * a_cmd if mag > a_max else a_cmd

    def _missile_velocity_vec(
        self, yaw_deg: float, pitch_deg: float, speed_mps: float
    ) -> np.ndarray:
        """
        Map (yaw, pitch, speed) to ENU velocity.
        Geodetic yaw: 0° = North, clockwise positive.
        ENU axes ordering: [E, N, U].
        """
        sy = math.sin(math.radians(yaw_deg))
        cy = math.cos(math.radians(yaw_deg))
        sp = math.sin(math.radians(pitch_deg))
        cp = math.cos(math.radians(pitch_deg))
        return np.array([cp * sy * speed_mps, cp * cy * speed_mps, sp * speed_mps], dtype=float)

    def _get_target_state_enu(self, missile_position, target_position):
        """
        Return (p_or_None, v_or_None) in missile ENU, from the same tracker KF snapshot.

        Priority:
          1. tracker_manager.get_target_state_for_pn(tid, missile_pos)
             — returns coherent (p_tgt_msl, v_tgt_msl) from the same KF state,
               both already in missile ENU. Using this position as R_vec eliminates
               systematic time skew between the filtered velocity and the geodetic position.
          2. target_provider.get_guidance_velocity() — velocity only (p=None).
             Position falls back to geodetic_to_enu in compute().
        """
        radar = getattr(self.missile, "radar", None)

        target_id = None
        prov = getattr(self.missile, "target_provider", None)
        if prov is not None:
            target_id = getattr(prov, "get_guidance_tid", lambda: None)()

        # Fallback: radar.get_tracker_info() (backward-compat)
        if target_id is None and radar is not None:
            getter = getattr(radar, "get_tracker_info", None)
            if getter is not None:
                info = getter() or {}
                target_id = info.get("tid")

        if target_id is not None and radar is not None:
            tracker_manager = getattr(radar, "tracker_manager", None)
            if tracker_manager is not None and hasattr(tracker_manager, "get_target_state_for_pn"):
                state_tuple = tracker_manager.get_target_state_for_pn(target_id, missile_position)
                if (
                    state_tuple is not None
                    and isinstance(state_tuple, (tuple, list))
                    and len(state_tuple) >= 2
                ):
                    p_raw, v_raw = state_tuple[0], state_tuple[1]  # (p, v[, P6]) in missile ENU
                    p = np.asarray(p_raw, dtype=float).reshape(-1)
                    v = np.asarray(v_raw, dtype=float).reshape(-1)
                    p_valid = (
                        p.size == 3
                        and np.isfinite(p).all()
                        and 1.0 < float(np.linalg.norm(p)) < 300_000.0
                    )
                    v_mag = float(np.linalg.norm(v))
                    v_valid = v.size == 3 and np.isfinite(v).all() and 0.0 < v_mag < 1400.0
                    return (p if p_valid else None, v if v_valid else None)

        # Provider velocity fallback: rotate cached velocity into current missile ENU.
        # get_guidance_velocity_in_missile_enu() applies the ENU(vel_ref)->ENU(missile)
        # rotation automatically; fall back to the raw accessor only for older providers
        # that predate the frame-aware API.
        if prov is not None:
            get_vel_fn = getattr(prov, "get_guidance_velocity_in_missile_enu", None)
            if get_vel_fn is not None:
                v = get_vel_fn(missile_position)
            else:
                v = getattr(prov, "get_guidance_velocity", lambda: None)()
            if v is not None:
                v = np.asarray(v, dtype=float).reshape(-1)
                if v.size == 3 and np.isfinite(v).all():
                    s = float(np.linalg.norm(v))
                    if 0.0 < s < 1400.0:
                        return None, v

        return None, None

    def _enu_rotation(self, from_lat, from_lon, to_lat, to_lon):
        """
        Rotation matrix mapping vectors from ENU(from_lat,from_lon) into ENU(to_lat,to_lon):
          v_to = R @ v_from, with R ≈ argmin ||R - (B_to B_from^T)|| subject to R ∈ SO(3).
        """
        B_from = self._enu_basis(from_lat, from_lon)
        B_to = self._enu_basis(to_lat, to_lon)
        R = B_to.T @ B_from  # correct: v_to = R @ v_from maps ENU(from) → ENU(to)
        # Project to nearest proper rotation (SVD-based)
        try:
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0.0:
                U[:, -1] *= -1.0
                R = U @ Vt
        except Exception:
            pass
        return R

    def _enu_basis(self, lat_deg, lon_deg):
        """
        Orthonormal ENU basis at (lat,lon) in ECEF; columns are [e, n, u].
        Vectors are orthonormal by construction; no QR needed (QR introduces sign flips).
        """
        lat = math.radians(float(lat_deg))
        lon = math.radians(float(lon_deg))
        e = np.array([-math.sin(lon), math.cos(lon), 0.0])
        n = np.array(
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)]
        )
        u = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
        return np.column_stack([e, n, u])
