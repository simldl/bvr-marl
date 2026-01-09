from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from simulator.core.helpers import Position

# Aircraft types (keep these imports in sync with your package layout)
from aircrafts.types.debug_plane import DebugPlane
from aircrafts.types.eurofighter import Eurofighter
from aircrafts.types.f22 import F22
from aircrafts.types.f35 import F35


# -----------------------------------------------------------------------------
# Registry and helpers
# -----------------------------------------------------------------------------

# Normalized keys -> class
AIRCRAFT_REGISTRY: Dict[str, type] = {
    # DebugPlane
    "debug": DebugPlane,
    "debugplane": DebugPlane,
    "debug-plane": DebugPlane,
    # Eurofighter
    "eurofighter": Eurofighter,
    # F-22
    "f22": F22,
    "f-22": F22,
    # F-35
    "f35": F35,
    "f-35": F35,
}


def _norm(name: str) -> str:
    """Normalize a human string to a registry key."""
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _resolve_aircraft_cls(env, aid: str, group: str):
    """
    Pick an aircraft class for (aid, group).
    Priority:
      1) env.aircraft_type_map[aid] if present and already a class (callable)
      2) env.aircraft_type_map[aid] if it's a string -> resolve via registry
      3) env.config['agent_aircraft'] / ['opponent_aircraft'] / ['aircraft'] -> registry
      4) default: DebugPlane
    """
    # 1) If the env already prepared a per-agent mapping and it's directly a class:
    if hasattr(env, "aircraft_type_map") and isinstance(getattr(env, "aircraft_type_map"), dict):
        maybe = env.aircraft_type_map.get(aid)
        if callable(maybe):
            return maybe
        if isinstance(maybe, str):
            cls = AIRCRAFT_REGISTRY.get(_norm(maybe))
            if cls is not None:
                return cls

    # 2) Fall back to config names
    cfg = getattr(env, "config", {}) or {}
    name: Optional[str] = None
    if group == "agent":
        name = cfg.get("agent_aircraft")
    elif group == "opponent":
        name = cfg.get("opponent_aircraft")
    if name is None:
        name = cfg.get("aircraft", "DebugPlane")

    cls = AIRCRAFT_REGISTRY.get(_norm(name))
    if cls is None:
        available = ", ".join(sorted(AIRCRAFT_REGISTRY.keys()))
        raise KeyError(f"Unknown aircraft type '{name}'. Available: {available}")
    return cls


# -----------------------------------------------------------------------------
# Spawn position and unit creation
# -----------------------------------------------------------------------------

def spawn_position(group: str, map_size_km: float, map_limits) -> Position:
    """
    Compute a random spawn position for the given group, splitting West/East by lon.
    Latitude/longitude bounds are derived from the map size in km.
    Altitude is sampled within [min_alt+100, min(max_alt, 10km)].
    """
    # Convert half map size to degrees (~111 km per degree)
    half_deg = (map_size_km / 2.0) / 111.0

    lat = float(np.random.uniform(-half_deg, half_deg))
    if group == "agent":
        lon = float(np.random.uniform(-half_deg, 0.0))
    elif group == "opponent":
        lon = float(np.random.uniform(0.0, half_deg))
    else:
        lon = float(np.random.uniform(-half_deg, half_deg))

    min_alt = float(getattr(map_limits, "min_alt", 0.0))
    max_alt = float(getattr(map_limits, "max_alt", 20000.0))
    alt_low = max(1000.0, min_alt + 100.0)
    alt_high = min(max_alt, 10000.0)
    alt = float(np.random.uniform(alt_low, alt_high))

    return Position(lat=lat, lon=lon, alt=alt)


def spawn_unit(env, aid: str, group: str) -> int:
    """
    Create and register a unit for agent-id `aid` in team `group` ("agent"|"opponent").
    Returns the simulator unit-id.
    """
    # 1) Position and basic kinematics
    map_size_km = float(getattr(env, "map_size_km", 300.0))
    pos = spawn_position(group, map_size_km, env.map_limits)

    # Point agents east and opponents west by default (so they face each other)
    yaw = 90.0 if group == "agent" else 270.0

    # Reasonable default speed; allow override via env.config["default_speed"]
    cfg = getattr(env, "config", {}) or {}
    speed = float(cfg.get("default_speed", 250.0))

    # 2) Altitude envelope from map limits (passed to the aircraft constructor)
    min_alt = float(getattr(env.map_limits, "min_alt", 0.0))
    max_alt = float(getattr(env.map_limits, "max_alt", 20000.0))

    # 3) Class resolution and instantiation
    aircraft_cls = _resolve_aircraft_cls(env, aid, group)
    unit = aircraft_cls(
        pos,            # Position
        yaw,            # heading deg
        speed,          # m/s
        group,          # "agent" | "opponent"
        env.map_limits, # map limits object
        min_alt,        # min_alt_m
        max_alt,        # max_alt_m
    )

    # 4) Register into the simulator and start tracing
    uid = env.simulator.add_unit(unit)
    if uid is None:
        raise RuntimeError(f"Could not add unit for {aid}")
    env.simulator.record_unit_trace(uid)
    return uid


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

class MapBoundaryChecker:
    @staticmethod
    def within_bounds(unit, map_limits) -> bool:
        return (
            map_limits.left_lon < unit.position.lon < map_limits.right_lon
            and map_limits.bottom_lat < unit.position.lat < map_limits.top_lat
        )