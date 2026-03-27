from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from air_to_air_rl.aircrafts.control.awacs_controller import OrbitConfig, create_orbit_controller
from air_to_air_rl.aircrafts.types.awacs import AWACS

# Aircraft types (keep these imports in sync with your package layout)
from air_to_air_rl.aircrafts.types.debug_plane import DebugPlane
from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.aircrafts.types.f35 import F35
from air_to_air_rl.aircrafts.types.su57 import Su57
from air_to_air_rl.rl.utils.config_resolvers import (
    resolve_datalink_config,
    resolve_loadout_config,
    resolve_missile_config,
)
from air_to_air_rl.simulator.core.helpers import Position

if TYPE_CHECKING:
    from air_to_air_rl.rl.environment.gym.gym_components.config import AWACSConfigData

# Normalized keys -> class
AIRCRAFT_REGISTRY: dict[str, type] = {
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
    # Su-57
    "su57": Su57,
    "su-57": Su57,
    # AWACS
    "awacs": AWACS,
    "e3": AWACS,
    "e-3": AWACS,
    "e7": AWACS,
    "e-7": AWACS,
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
    name: str | None = None
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
        pos,  # Position
        yaw,  # heading deg
        speed,  # m/s
        group,  # "agent" | "opponent"
        env.map_limits,  # map limits object
        min_alt,  # min_alt_m
        max_alt,  # max_alt_m
    )

    # 4) Apply missile configuration override from config
    _apply_missile_override(unit, group, cfg)

    # 5) Apply loadout configuration override from config
    _apply_loadout_override(unit, cfg)

    # 6) Apply datalink mode configuration
    _apply_datalink_mode(unit, cfg)

    # 7) Register into the simulator and start tracing
    uid = env.simulator.add_unit(unit)
    if uid is None:
        raise RuntimeError(f"Could not add unit for {aid}")
    env.simulator.record_unit_trace(uid)
    return uid


def _apply_missile_override(unit, group: str, cfg: dict) -> None:
    """
    Apply missile configuration override to an aircraft unit.

    If missile_config is specified in the configuration, overrides
    the aircraft's default missile types with the configured ones.
    """
    missile_config = cfg.get("missile_config")
    if missile_config is None:
        return

    resolved = resolve_missile_config(missile_config)

    if group == "agent" and resolved["agent_missiles"]:
        unit.missile_types = resolved["agent_missiles"]
        # Rebuild missile parameters cache for NEZ/DLZ calculations
        _rebuild_missile_params(unit)
    elif group == "opponent" and resolved["opponent_missiles"]:
        unit.missile_types = resolved["opponent_missiles"]
        _rebuild_missile_params(unit)


def _rebuild_missile_params(unit) -> None:
    """
    Rebuild the missile parameters cache after missile types are changed.
    This is needed for accurate NEZ/DLZ calculations.
    """
    from air_to_air_rl.missiles.missile_parameters import MissileParameters

    unit.missile_params = {}
    for missile_class in unit.missile_types:
        try:
            params = MissileParameters.from_missile_class(missile_class)
            unit.missile_params[missile_class.__name__] = params
        except Exception as e:
            print(f"Warning: Could not extract parameters for {missile_class.__name__}: {e}")


def _apply_loadout_override(unit, cfg: dict) -> None:
    """
    Apply loadout configuration override to an aircraft unit.

    Overrides max_missiles, flares, chaff, ecm, decoys if specified in config.
    """
    loadout_config = cfg.get("loadout_config")
    if loadout_config is None:
        return

    resolved = resolve_loadout_config(loadout_config)

    if resolved.get("max_missiles") is not None:
        unit.max_missiles = resolved["max_missiles"]
    if resolved.get("flares") is not None:
        unit.flares = resolved["flares"]
    if resolved.get("chaff") is not None:
        unit.chaff = resolved["chaff"]
    if resolved.get("ecm") is not None:
        unit.ecm = resolved["ecm"]
    if resolved.get("decoys") is not None:
        unit.decoys = resolved["decoys"]


def _apply_datalink_mode(unit, cfg: dict) -> None:
    """
    Apply datalink mode configuration to an aircraft unit.

    Sets the radar's datalink mode based on configuration.
    """
    datalink_mode = resolve_datalink_config(cfg)
    if hasattr(unit, "radar") and hasattr(unit.radar, "data_link"):
        unit.radar.data_link.set_mode(datalink_mode)


def spawn_unit_with_geometry(
    env,
    aid: str,
    group: str,
    position: Position,
    heading: float,
    speed: float,
) -> int:
    """
    Create and register a unit with explicit position, heading, and speed.

    Used for controlled geometry spawning in study scenarios.

    Args:
        env: Environment manager with simulator and config.
        aid: Agent ID string.
        group: "agent" or "opponent".
        position: Explicit spawn position.
        heading: Heading in degrees (0=North, 90=East).
        speed: Speed in m/s.

    Returns:
        Simulator unit-id.
    """
    cfg = getattr(env, "config", {}) or {}

    # Altitude envelope from map limits
    min_alt = float(getattr(env.map_limits, "min_alt", 0.0))
    max_alt = float(getattr(env.map_limits, "max_alt", 20000.0))

    # Class resolution and instantiation
    aircraft_cls = _resolve_aircraft_cls(env, aid, group)
    unit = aircraft_cls(
        position,  # Position
        heading,  # heading deg
        speed,  # m/s
        group,  # "agent" | "opponent"
        env.map_limits,  # map limits object
        min_alt,  # min_alt_m
        max_alt,  # max_alt_m
    )

    # Apply configuration overrides
    _apply_missile_override(unit, group, cfg)
    _apply_loadout_override(unit, cfg)
    _apply_datalink_mode(unit, cfg)

    # Register into the simulator
    uid = env.simulator.add_unit(unit)
    if uid is None:
        raise RuntimeError(f"Could not add unit for {aid}")
    env.simulator.record_unit_trace(uid)
    return uid


def spawn_with_geometry_config(env, geometry_data: dict) -> dict[str, int]:
    """
    Spawn all units based on a geometry configuration.

    Args:
        env: Environment manager with simulator, config, and bvr_config.
        geometry_data: Output from GeometrySampler.compute_positions().

    Returns:
        Dictionary mapping agent IDs to simulator unit IDs.
    """
    agent_to_unit_id = {}

    agent_ids = getattr(env, "bvr_config", env).agent_ids
    opponent_ids = getattr(env, "bvr_config", env).opponent_ids

    # Spawn agents
    for i, aid in enumerate(agent_ids):
        if i < len(geometry_data["agent_positions"]):
            pos = geometry_data["agent_positions"][i]
            heading = geometry_data["agent_headings"][i]
            speed = geometry_data["agent_speed"]
        else:
            # Fallback to random if more agents than positions
            pos = spawn_position("agent", env.map_size_km, env.map_limits)
            heading = 90.0
            speed = 250.0

        uid = spawn_unit_with_geometry(env, aid, "agent", pos, heading, speed)
        agent_to_unit_id[aid] = uid

    # Spawn opponents
    for i, oid in enumerate(opponent_ids):
        if i < len(geometry_data["opponent_positions"]):
            pos = geometry_data["opponent_positions"][i]
            heading = geometry_data["opponent_headings"][i]
            speed = geometry_data["opponent_speed"]
        else:
            pos = spawn_position("opponent", env.map_size_km, env.map_limits)
            heading = 270.0
            speed = 250.0

        uid = spawn_unit_with_geometry(env, oid, "opponent", pos, heading, speed)
        agent_to_unit_id[oid] = uid

    return agent_to_unit_id


def spawn_awacs(
    env,
    group: str,
    awacs_config: AWACSConfigData,
    fighter_center_lon: float = None,
) -> int | None:
    """
    Spawn an AWACS aircraft for the given team with a figure-8 orbit controller.

    AWACS are placed in fixed side zones adjacent to the combat area:
      - Blue / agent    AWACS: west side  (negative longitude)
      - Red  / opponent AWACS: east side  (positive longitude)

    Each AWACS flies a NS-oriented figure-8 pattern with radius orbit_radius_km.
    The figure-8 center is orbit_radius_km beyond the combat zone boundary so
    the inner edge of the orbit just touches the combat zone edge.

    The AWACS unit is given full_map_limits (if available on env) so it is NOT
    destroyed when flying in its side zone outside the combat boundaries.

    Args:
        env: Environment manager with simulator, map_limits, and optionally
             full_map_limits.
        group: "agent" (blue/west) or "opponent" (red/east).
        awacs_config: AWACSConfigData with orbit and lock settings.
        fighter_center_lon: Ignored – AWACS is placed at the fixed side-zone
                            position regardless of fighter spawn positions.

    Returns:
        Simulator unit-id for the AWACS, or None if not configured.
    """
    cfg = getattr(env, "config", {}) or {}
    km_per_deg = 111.0

    # Combat zone boundary in degrees
    combat_map_limits = getattr(env, "map_limits", None)
    if combat_map_limits is None:
        raise RuntimeError("env.map_limits is not set")
    combat_half_lon_deg = abs(combat_map_limits.left_lon)  # == right_lon

    # Full map limits (for AWACS – allows flight outside combat zone)
    awacs_map_limits = getattr(env, "full_map_limits", None) or combat_map_limits

    # AWACS figure-8 circle radius in degrees
    orbit_radius_deg = awacs_config.orbit_radius_km / km_per_deg

    # Position AWACS center orbit_radius_km beyond the combat zone boundary.
    # This places the inner circle tangent to the combat edge.
    awacs_lat = 0.0  # Centred on the EW axis for maximum NS coverage
    if group == "agent":
        # Blue AWACS – west side
        awacs_lon = -(combat_half_lon_deg + orbit_radius_deg)
        awacs_heading = 90.0  # Initially facing east (toward the fight)
    else:
        # Red AWACS – east side
        awacs_lon = combat_half_lon_deg + orbit_radius_deg
        awacs_heading = 270.0  # Initially facing west (toward the fight)

    awacs_pos = Position(
        lat=awacs_lat,
        lon=awacs_lon,
        alt=awacs_config.orbit_altitude_m,
    )

    # Altitude envelope from the AWACS map limits
    min_alt = float(getattr(awacs_map_limits, "min_alt", 0.0))
    max_alt = float(getattr(awacs_map_limits, "max_alt", 20000.0))

    # Create AWACS unit using its own (extended) map limits so boundary
    # violation logic does not fire while it flies its side-zone pattern.
    awacs_unit = AWACS(
        position=awacs_pos,
        yaw_deg=awacs_heading,
        speed_mps=awacs_config.orbit_speed_mps,
        group=group,
        map_limits=awacs_map_limits,
        min_alt_m=min_alt,
        max_alt_m=max_alt,
        name=f"AWACS_{group.upper()}",
        lock_fov_deg=awacs_config.lock_fov_deg,
    )

    # Override is_non_engageable based on config (AWACS defaults to non-engageable in __init__)
    awacs_unit.is_non_engageable = awacs_config.awacs_non_engageable

    # Create orbit controller centred on the AWACS spawn position
    orbit_controller = create_orbit_controller(
        pattern=awacs_config.orbit_pattern,
        center_lat=awacs_lat,
        center_lon=awacs_lon,
        leg_length_km=awacs_config.orbit_leg_length_km,
        turn_radius_km=awacs_config.orbit_radius_km,
        altitude_m=awacs_config.orbit_altitude_m,
        speed_mps=awacs_config.orbit_speed_mps,
        clockwise=awacs_config.orbit_clockwise,
        zone_min_lat=awacs_map_limits.bottom_lat,
        zone_max_lat=awacs_map_limits.top_lat,
        zone_min_lon=awacs_map_limits.left_lon,
        zone_max_lon=awacs_map_limits.right_lon,
    )
    awacs_unit.orbit_controller = orbit_controller

    # Apply datalink mode (AWACS should share data with teammates)
    _apply_datalink_mode(awacs_unit, cfg)

    # Register into the simulator
    uid = env.simulator.add_unit(awacs_unit)
    if uid is None:
        raise RuntimeError(f"Could not add AWACS unit for {group}")
    env.simulator.record_unit_trace(uid)

    return uid


def compute_awacs_orbit_center(
    group: str,
    geometry_data: dict | None,
    map_size_km: float,
    orbit_distance_km: float,
) -> float:
    """
    Compute the center longitude for AWACS orbit.

    With the new side-zone layout the orbit center is determined solely by the
    combat zone boundary and orbit_radius_km (passed here as orbit_distance_km
    for backward compatibility).  Fighter positions are no longer used.

    Args:
        group: "agent" (west/blue) or "opponent" (east/red).
        geometry_data: Unused – retained for API compatibility.
        map_size_km: Combat zone size in km (square).
        orbit_distance_km: Radius of the figure-8 circles in km.

    Returns:
        Longitude for AWACS orbit center in degrees.
    """
    km_per_deg = 111.0
    combat_half_deg = (map_size_km / 2.0) / km_per_deg
    orbit_radius_deg = orbit_distance_km / km_per_deg

    if group == "agent":
        return -(combat_half_deg + orbit_radius_deg)  # West side
    else:
        return combat_half_deg + orbit_radius_deg  # East side


class MapBoundaryChecker:
    @staticmethod
    def within_bounds(unit, map_limits) -> bool:
        return (
            map_limits.left_lon < unit.position.lon < map_limits.right_lon
            and map_limits.bottom_lat < unit.position.lat < map_limits.top_lat
        )
