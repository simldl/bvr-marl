from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from bvr_marl_core.simulator.utils.angles import deg2rad


@dataclass
class HitParams:
    fuse_radius_m: float = 500.0
    target_radius_m: float = 0.0
    max_consider_range_m: float = 50_000.0


def _latlonalt_to_local_m(
    lat: float, lon: float, alt: float, lat0: float, lon0: float
) -> np.ndarray:
    """Coarse conversion to local ENU (meters)."""
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(deg2rad(lat0))
    x = (lon - lon0) * lon_scale
    y = (lat - lat0) * lat_scale
    z = alt
    return np.array([x, y, z], dtype=float)


def _closest_approach_time(r0: np.ndarray, v_rel: np.ndarray) -> float:
    """Minimizes ||r0 + t * v_rel||^2 over t∈R → t* = - (r0·v_rel) / ||v_rel||^2"""
    denom = float(v_rel @ v_rel)
    if denom <= 0.0:
        return 0.0
    t_star = -float(r0 @ v_rel) / denom
    return t_star


class MissileHitCalculator:
    """
    Continuous Collision Detection (CCD) between missile and targets.

    Approach:
      r(t) = (p_m0 + t*vm) - (p_t0 + t*vt) = r0 + t * v_rel, t∈[0,1]
    Find t* of minimal distance, clamp to [0,1], and check
    d_min(t*) <= (fuse_radius + target_radius).

    Returns: (hit: bool, victim, t_hit_fraction∈[0,1] or None)
    """

    def __init__(self, params: HitParams | None = None):
        self.params = params or HitParams()
        self._preposes: dict[int, tuple[float, float, float]] = {}
        self._enu_origin: tuple[float, float] | None = None

    def begin_tick_snapshot(self, units: Iterable) -> None:
        """Call at *beginning* of a tick before any units are moved.
        Stores t0 poses and sets an ENU origin if available.
        """
        self._preposes.clear()
        lat0, lon0 = None, None
        for u in units:
            if hasattr(u, "position") and u.position is not None:
                lat = float(u.position.lat)
                lon = float(u.position.lon)
                alt = float(u.position.alt)
                self._preposes[u.id] = (lat, lon, alt)
                if lat0 is None:
                    lat0, lon0 = lat, lon
        if lat0 is not None:
            self._enu_origin = (lat0, lon0)

    def check_and_maybe_hit(
        self,
        missile,
        potential_targets: Iterable,
        tick_secs: float,
    ) -> tuple[bool, object | None, float | None]:
        """Call after *all* movement updates of the tick (t1 is set).

        Returns:
            (True, target_obj, t_hit_fraction∈[0,1]) on hit
            (False, None, None) otherwise
        """
        self._ensure_enu_origin(missile)
        lat0_ref, lon0_ref = self._enu_origin  # type: ignore[assignment]

        p_m0, p_m1, v_m = self._compute_motion_vectors(missile, lat0_ref, lon0_ref)

        max_r = self.params.max_consider_range_m
        fuse = self.params.fuse_radius_m
        eff_radius = fuse + self.params.target_radius_m
        for tgt in potential_targets:
            if tgt is missile or getattr(tgt, "is_destroyed", False):
                continue

            p_t0, p_t1, v_t = self._compute_motion_vectors(tgt, lat0_ref, lon0_ref)

            if self._early_reject(p_m0, p_m1, p_t0, p_t1, max_r):
                continue

            r0, v_rel = self._relative_motion(p_m0, v_m, p_t0, v_t)

            t_star = self._clamped_closest_time(r0, v_rel)

            d_min = self._distance_at_time(r0, v_rel, t_star)

            if d_min <= eff_radius:
                return True, tgt, t_star

        return False, None, None

    def check_segment_hit(
        self,
        m0: tuple[float, float, float],
        m1: tuple[float, float, float],
        t0: tuple[float, float, float],
        t1: tuple[float, float, float],
    ) -> tuple[bool, float | None, float]:
        """Self-contained CCD over a single substep.

        Unlike :meth:`check_and_maybe_hit`, this does NOT read the per-tick
        ``begin_tick_snapshot`` poses, so it is safe to call repeatedly inside a
        substep loop where both objects move several times within one tick.

        Args:
            m0, m1: missile geodetic pose ``(lat, lon, alt)`` at the start/end of
                the substep.
            t0, t1: target geodetic pose ``(lat, lon, alt)`` at the start/end of
                the substep.

        Returns:
            ``(hit, t_frac, d_min)`` where ``d_min`` is the closest-approach
            distance (m) over the substep and ``t_frac in [0, 1]`` is the
            fraction of the substep at closest approach when a hit occurs
            (``t_frac`` is ``None`` on a miss). ``d_min`` is always returned so
            callers can model a range-dependent kill probability.
        """
        if self._enu_origin is not None:
            lat0, lon0 = self._enu_origin
        else:
            lat0, lon0 = m0[0], m0[1]

        p_m0 = _latlonalt_to_local_m(m0[0], m0[1], m0[2], lat0, lon0)
        p_m1 = _latlonalt_to_local_m(m1[0], m1[1], m1[2], lat0, lon0)
        p_t0 = _latlonalt_to_local_m(t0[0], t0[1], t0[2], lat0, lon0)
        p_t1 = _latlonalt_to_local_m(t1[0], t1[1], t1[2], lat0, lon0)

        v_m = p_m1 - p_m0
        v_t = p_t1 - p_t0
        r0, v_rel = self._relative_motion(p_m0, v_m, p_t0, v_t)
        t_star = self._clamped_closest_time(r0, v_rel)
        d_min = self._distance_at_time(r0, v_rel, t_star)

        eff_radius = self.params.fuse_radius_m + self.params.target_radius_m
        return (d_min <= eff_radius), t_star, d_min

    def _ensure_enu_origin(self, missile) -> None:
        """Set ENU origin to missile position if needed (fallback)."""
        if self._enu_origin is None:
            self._enu_origin = (
                float(missile.position.lat),
                float(missile.position.lon),
            )

    def _get_t0_pose(self, obj) -> tuple[float, float, float] | None:
        """Read stored t0 pose or None if not available."""
        return self._preposes.get(obj.id, None)

    def _current_pose(self, obj) -> tuple[float, float, float]:
        return (
            float(obj.position.lat),
            float(obj.position.lon),
            float(obj.position.alt),
        )

    def _compute_motion_vectors(
        self, obj, lat0_ref: float, lon0_ref: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute p0, p1, v for an object (missile or target)."""
        t0_pose = self._get_t0_pose(obj)
        if t0_pose is None:
            t0_pose = self._current_pose(obj)
        t1_pose = self._current_pose(obj)
        p0 = _latlonalt_to_local_m(*t0_pose, lat0_ref, lon0_ref)
        p1 = _latlonalt_to_local_m(*t1_pose, lat0_ref, lon0_ref)
        v = p1 - p0
        return p0, p1, v

    @staticmethod
    def _early_reject(
        p_m0: np.ndarray,
        p_m1: np.ndarray,
        p_t0: np.ndarray,
        p_t1: np.ndarray,
        max_r: float,
    ) -> bool:
        """Simple distance coarse filter at endpoints."""
        return bool(np.linalg.norm(p_m0 - p_t0) > max_r and np.linalg.norm(p_m1 - p_t1) > max_r)

    @staticmethod
    def _relative_motion(
        p_m0: np.ndarray,
        v_m: np.ndarray,
        p_t0: np.ndarray,
        v_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r0 = p_m0 - p_t0
        v_rel = v_m - v_t
        return r0, v_rel

    @staticmethod
    def _clamped_closest_time(r0: np.ndarray, v_rel: np.ndarray) -> float:
        t_star = _closest_approach_time(r0, v_rel)
        return max(0.0, min(1.0, float(t_star)))

    @staticmethod
    def _distance_at_time(r0: np.ndarray, v_rel: np.ndarray, t: float) -> float:
        r_min = r0 + t * v_rel
        return float(np.linalg.norm(r_min))
