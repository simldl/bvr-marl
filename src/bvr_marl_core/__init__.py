"""bvr_marl_core - BVR simulation engine public API.

This package exposes a curated flat convenience surface, so consumers can write
``from bvr_marl_core import Simulator, PlatformState`` for stable lightweight
symbols.

The actual symbols are imported lazily. This keeps package initialization cheap
and lets tooling import ``bvr_marl_core.public_api`` without pulling in numpy,
torch, ray, matplotlib, cartopy, or simulator implementation modules.

The authoritative, machine-readable allowlist of modules extension packages may
import lives in ``bvr_marl_core.public_api.PUBLIC_API_MODULES``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_PUBLIC_ATTRS: dict[str, tuple[str, str]] = {
    # Simulator
    "Simulator": ("bvr_marl_core.simulator", "Simulator"),
    "MapLimits": ("bvr_marl_core.simulator", "MapLimits"),
    "Position": ("bvr_marl_core.simulator", "Position"),
    "normalize_angle": ("bvr_marl_core.simulator", "normalize_angle"),
    "signed_yaw_deg_diff": ("bvr_marl_core.simulator", "signed_yaw_deg_diff"),
    "geodetic_bearing_deg": ("bvr_marl_core.simulator", "geodetic_bearing_deg"),
    "geodetic_distance_km": ("bvr_marl_core.simulator", "geodetic_distance_km"),
    "UnitDestroyedEvent": ("bvr_marl_core.simulator", "UnitDestroyedEvent"),
    "UnitRegisteredEvent": ("bvr_marl_core.simulator", "UnitRegisteredEvent"),
    "UnitRemovedEvent": ("bvr_marl_core.simulator", "UnitRemovedEvent"),
    # Domain
    "PlatformState": ("bvr_marl_core.domain", "PlatformState"),
    "SensorState": ("bvr_marl_core.domain", "SensorState"),
    "WeaponState": ("bvr_marl_core.domain", "WeaponState"),
    "TrackState": ("bvr_marl_core.domain", "TrackState"),
    "EngagementState": ("bvr_marl_core.domain", "EngagementState"),
    "SimMetadata": ("bvr_marl_core.domain", "SimMetadata"),
    "ControlCommand": ("bvr_marl_core.domain", "ControlCommand"),
    "FireCommand": ("bvr_marl_core.domain", "FireCommand"),
    "ManeuverCommand": ("bvr_marl_core.domain", "ManeuverCommand"),
    "DetectionEvent": ("bvr_marl_core.domain", "DetectionEvent"),
    "LaunchEvent": ("bvr_marl_core.domain", "LaunchEvent"),
    "HitEvent": ("bvr_marl_core.domain", "HitEvent"),
    "KillEvent": ("bvr_marl_core.domain", "KillEvent"),
    "LockChangeEvent": ("bvr_marl_core.domain", "LockChangeEvent"),
    "ScenarioStartEvent": ("bvr_marl_core.domain", "ScenarioStartEvent"),
    "ScenarioEndEvent": ("bvr_marl_core.domain", "ScenarioEndEvent"),
    # Interfaces
    "Controller": ("bvr_marl_core.interfaces", "Controller"),
    "ScriptedController": ("bvr_marl_core.interfaces", "ScriptedController"),
    "ControllerFactory": ("bvr_marl_core.interfaces", "ControllerFactory"),
    "ObservationHook": ("bvr_marl_core.interfaces", "ObservationHook"),
    "RewardFunction": ("bvr_marl_core.interfaces", "RewardFunction"),
    "SensorPlugin": ("bvr_marl_core.interfaces", "SensorPlugin"),
    "WeaponPlugin": ("bvr_marl_core.interfaces", "WeaponPlugin"),
    "FlightModelPlugin": ("bvr_marl_core.interfaces", "FlightModelPlugin"),
    "VisualizerPlugin": ("bvr_marl_core.interfaces", "VisualizerPlugin"),
    "ProtocolAdapter": ("bvr_marl_core.interfaces", "ProtocolAdapter"),
    "FMUPlugin": ("bvr_marl_core.interfaces", "FMUPlugin"),
    # Schema
    "SCHEMA_VERSION": ("bvr_marl_core.schema", "SCHEMA_VERSION"),
    "ScenarioConfig": ("bvr_marl_core.schema", "ScenarioConfig"),
    "SimulationConfig": ("bvr_marl_core.schema", "SimulationConfig"),
    "AgentConfig": ("bvr_marl_core.schema", "AgentConfig"),
    "TeamConfig": ("bvr_marl_core.schema", "TeamConfig"),
    "WeaponConfig": ("bvr_marl_core.schema", "WeaponConfig"),
    "SensorConfig": ("bvr_marl_core.schema", "SensorConfig"),
    "validate_config": ("bvr_marl_core.schema", "validate_config"),
    "migrate_config": ("bvr_marl_core.schema", "migrate_config"),
    # Registry
    "get_aircraft_class": ("bvr_marl_core.registry", "get_aircraft_class"),
    "get_missile_class": ("bvr_marl_core.registry", "get_missile_class"),
    # Tactical
    "NoEscapeZoneCalculator": ("bvr_marl_core.tactical", "NoEscapeZoneCalculator"),
    "TrackPrioritySystem": ("bvr_marl_core.tactical", "TrackPrioritySystem"),
    "ObservationHelper": ("bvr_marl_core.tactical", "ObservationHelper"),
    "OrbitConfig": ("bvr_marl_core.tactical", "OrbitConfig"),
    "create_orbit_controller": ("bvr_marl_core.tactical", "create_orbit_controller"),
    "MissileParameters": ("bvr_marl_core.tactical", "MissileParameters"),
    # Observation-space fixed slot counts (the obs/action spaces are fixed-size, so
    # these cap how many fighters per side a scenario can represent). Exposed on the
    # flat API so extensions can size team/scenario settings without a deep import
    # and without pulling the heavyweight RL stack. Lazily loaded; ray-free.
    "K_FF": ("bvr_marl_core.rl.environment.spaces.observation.constants", "K_FF"),
    "K_EF": ("bvr_marl_core.rl.environment.spaces.observation.constants", "K_EF"),
}

__all__ = list(_PUBLIC_ATTRS)


def __getattr__(name: str):
    """Resolve flat public API symbols, and submodules, on first access.

    Flat symbols (Simulator, PlatformState, …) come from ``_PUBLIC_ATTRS``.
    Submodule attribute access (e.g. ``bvr_marl_core.interfaces``) is supported
    too: under eager init those bindings appeared as a side effect of importing
    the package; under lazy init we import the submodule on demand instead.
    """
    attr = _PUBLIC_ATTRS.get(name)
    if attr is not None:
        module_name, attr_name = attr
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    # Fall back to a genuine submodule. find_spec first so a missing submodule
    # yields AttributeError, while an import error *inside* a real submodule
    # propagates instead of being masked.
    if not name.startswith("_") and find_spec(f"{__name__}.{name}") is not None:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
