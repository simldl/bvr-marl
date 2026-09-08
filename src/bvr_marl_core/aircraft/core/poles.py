"""Lightweight A-pole/F-pole estimates for BVR timeline metrics.

The helper keeps the accepted terminology separate:

* A-pole is shooter-target slant range when an active missile goes active.
* F-pole is shooter-target slant range when the missile intercepts.

This is still an estimate, not a full flyout integration. It solves active and
intercept event times from current relative motion, then evaluates the
shooter-target range at those event times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg


@dataclass(frozen=True, slots=True)
class LaunchPoleEstimate:
    valid: bool
    slant_range_m: float = 0.0
    a_pole_range_m: float = 0.0
    f_pole_range_m: float = 0.0
    active_range_m: float = 0.0
    time_to_active_s: float = 0.0
    time_to_impact_s: float = 0.0
    missile_avg_speed_mps: float = 0.0
    shooter_target_closure_mps: float = 0.0
    missile_target_closure_mps: float = 0.0
    intercept_possible: bool = False
    active_supported: bool = False
    requires_lock_until_impact: bool = False
    fox_type: int = 0
    is_estimate: bool = True
    reason: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _number(obj: Any, name: str, default: float) -> float:
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return value if math.isfinite(value) else default


def _radar_range_m(params: Any, default: float) -> float:
    radar = params.get("radar") if isinstance(params, dict) else getattr(params, "radar", None)
    if radar is None:
        return _number(params, "active_range_m", default)
    return _number(radar, "max_range_m", default)


def _resolve_params(own: Any, missile_params: Any | None) -> Any | None:
    if missile_params is not None:
        return missile_params

    wez = getattr(own, "wez", None)
    if wez is not None and hasattr(wez, "_get_best_missile_params"):
        params = wez._get_best_missile_params(own)
        if params is not None:
            return params

    try:
        return NoEscapeZoneCalculator(own)._get_best_missile_params(own)
    except Exception:
        return None


def _effective_active_range_m(params: Any) -> float:
    fox_type = int(_number(params, "fox_type", 3.0))
    if fox_type != 3:
        return 0.0

    max_range = max(_number(params, "max_range_m", 75_000.0), 1.0)
    seeker_range = _radar_range_m(params, min(20_000.0, max_range))

    # Public missile configs often expose seeker/radar detection range, not the
    # commanded terminal-active handoff range. Cap the estimate to a terminal
    # fraction of kinematic reach while preserving smaller seeker limits.
    terminal_cap = max(15_000.0, 0.25 * max_range)
    return _clamp(seeker_range, 0.0, min(max_range, terminal_cap))


def _average_missile_speed_mps(params: Any, own: Any) -> float:
    max_speed = max(_number(params, "max_speed_mps", 1000.0), 1.0)
    launch_speed = max(float(getattr(own, "speed", 0.0) or 0.0), 0.0)
    estimate = 0.65 * max_speed + 0.35 * launch_speed
    return _clamp(estimate, min(300.0, max_speed), max_speed)


def _closing_components_mps(own: Any, target: Any) -> tuple[float, float]:
    own_to_target = geodetic_bearing_deg(
        own.position.lat,
        own.position.lon,
        target.position.lat,
        target.position.lon,
    )
    target_to_own = geodetic_bearing_deg(
        target.position.lat,
        target.position.lon,
        own.position.lat,
        own.position.lon,
    )

    own_speed = float(getattr(own, "speed", 0.0) or 0.0)
    target_speed = float(getattr(target, "speed", 0.0) or 0.0)
    own_yaw = float(getattr(own, "yaw_deg", 0.0) or 0.0)
    target_yaw = float(getattr(target, "yaw_deg", 0.0) or 0.0)

    own_closing = own_speed * math.cos(math.radians(signed_yaw_deg_diff(own_yaw, own_to_target)))
    target_closing = target_speed * math.cos(
        math.radians(signed_yaw_deg_diff(target_yaw, target_to_own))
    )
    return own_closing, target_closing


def estimate_launch_poles(
    own: Any,
    target: Any,
    missile_params: Any | None = None,
    *,
    lethal_radius_m: float = 100.0,
) -> LaunchPoleEstimate:
    """Estimate A-pole and F-pole from current geometry.

    The model assumes the missile flies down the current line of sight with an
    average speed derived from the missile's max speed and launch speed. Target
    and shooter continue with their current line-of-sight velocity components.
    """
    if own is None or target is None:
        return LaunchPoleEstimate(valid=False, reason="missing ownship or target")

    params = _resolve_params(own, missile_params)
    if params is None:
        return LaunchPoleEstimate(valid=False, reason="no missile parameters")

    try:
        slant_range = NoEscapeZoneCalculator._slant_range_m(own, target)
    except Exception as exc:
        return LaunchPoleEstimate(valid=False, reason=f"range calculation failed: {exc}")

    if slant_range <= 0.0 or not math.isfinite(slant_range):
        return LaunchPoleEstimate(valid=False, reason="invalid slant range")

    own_closing, target_closing = _closing_components_mps(own, target)
    shooter_target_closure = own_closing + target_closing

    missile_speed = _average_missile_speed_mps(params, own)
    missile_target_closure = missile_speed + target_closing
    if missile_target_closure <= 1.0:
        return LaunchPoleEstimate(
            valid=True,
            slant_range_m=slant_range,
            missile_avg_speed_mps=missile_speed,
            shooter_target_closure_mps=shooter_target_closure,
            missile_target_closure_mps=missile_target_closure,
            fox_type=int(_number(params, "fox_type", 0.0)),
            requires_lock_until_impact=int(_number(params, "fox_type", 3.0)) == 1,
            reason="target outruns missile closing geometry",
        )

    fox_type = int(_number(params, "fox_type", 3.0))
    active_range = _effective_active_range_m(params)
    time_to_impact = max(0.0, (slant_range - lethal_radius_m) / missile_target_closure)
    time_to_active = (
        max(0.0, (slant_range - active_range) / missile_target_closure)
        if active_range > 0.0
        else time_to_impact
    )

    def shooter_target_range_at(t_s: float) -> float:
        return max(0.0, slant_range - shooter_target_closure * t_s)

    a_pole = shooter_target_range_at(time_to_active) if active_range > 0.0 else 0.0
    f_pole = shooter_target_range_at(time_to_impact)

    path_length = missile_speed * time_to_impact
    max_range = _number(params, "max_range_m", 75_000.0)
    life_time = _number(params, "life_time_s", 120.0)
    intercept_possible = path_length <= max_range and time_to_impact <= life_time

    return LaunchPoleEstimate(
        valid=True,
        slant_range_m=slant_range,
        a_pole_range_m=a_pole,
        f_pole_range_m=f_pole,
        active_range_m=active_range,
        time_to_active_s=time_to_active,
        time_to_impact_s=time_to_impact,
        missile_avg_speed_mps=missile_speed,
        shooter_target_closure_mps=shooter_target_closure,
        missile_target_closure_mps=missile_target_closure,
        intercept_possible=intercept_possible,
        active_supported=fox_type == 3 and active_range > 0.0 and time_to_active < time_to_impact,
        requires_lock_until_impact=fox_type == 1,
        fox_type=fox_type,
        reason="" if intercept_possible else "estimated intercept exceeds missile reach",
    )
