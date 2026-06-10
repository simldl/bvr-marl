"""bvr_marl_core — BVR simulation engine public API.

This module re-exports every symbol that external consumers (e.g.
extension packages) are permitted to import.  Only items listed here are
considered stable public API.

Allowed public surface
----------------------
Simulator             — simulation loop entry point
MapLimits             — map boundary configuration
domain                — canonical state / command / event DTO model
interfaces            — abstract Protocols for plugins and controllers
schema                — versioned config/scenario models and validators
registry              — aircraft and missile type lookup
tactical              — engagement-geometry helpers, NEZ, target priority
physics.constraints   — pure-math flight envelope / energy calculators
utils                 — shared geometry and configuration helpers

DO NOT import from any other sub-package directly.  Anything not listed
above is an internal implementation detail and may change without notice.
"""

# ── Simulator entry point ────────────────────────────────────────────────────
# ── Domain model ─────────────────────────────────────────────────────────────
from bvr_marl_core.domain import (  # noqa: F401
    ControlCommand,
    DetectionEvent,
    EngagementState,
    FireCommand,
    HitEvent,
    KillEvent,
    LaunchEvent,
    LockChangeEvent,
    ManeuverCommand,
    PlatformState,
    ScenarioEndEvent,
    ScenarioStartEvent,
    SensorState,
    SimMetadata,
    TrackState,
    WeaponState,
)

# ── Interfaces / Protocols ────────────────────────────────────────────────────
from bvr_marl_core.interfaces import (  # noqa: F401
    Controller,
    ControllerFactory,
    FlightModelPlugin,
    FMUPlugin,
    ObservationHook,
    ProtocolAdapter,
    RewardFunction,
    ScriptedController,
    SensorPlugin,
    VisualizerPlugin,
    WeaponPlugin,
)

# ── Registry ──────────────────────────────────────────────────────────────────
from bvr_marl_core.registry import get_aircraft_class, get_missile_class  # noqa: F401

# ── Schema / config ───────────────────────────────────────────────────────────
from bvr_marl_core.schema import (  # noqa: F401
    SCHEMA_VERSION,
    AgentConfig,
    ScenarioConfig,
    SensorConfig,
    SimulationConfig,
    TeamConfig,
    WeaponConfig,
    migrate_config,
    validate_config,
)
from bvr_marl_core.simulator import (  # noqa: F401
    MapLimits,
    Position,
    Simulator,
    UnitDestroyedEvent,
    UnitRegisteredEvent,
    UnitRemovedEvent,
    geodetic_bearing_deg,
    geodetic_distance_km,
    normalize_angle,
    signed_yaw_deg_diff,
)

# ── Tactical helpers ──────────────────────────────────────────────────────────
from bvr_marl_core.tactical import (  # noqa: F401
    MissileParameters,
    NoEscapeZoneCalculator,
    ObservationHelper,
    OrbitConfig,
    TrackPrioritySystem,
    create_orbit_controller,
)

__all__ = [
    # Simulator
    "Simulator",
    "MapLimits",
    "Position",
    "normalize_angle",
    "signed_yaw_deg_diff",
    "geodetic_bearing_deg",
    "geodetic_distance_km",
    "UnitDestroyedEvent",
    "UnitRegisteredEvent",
    "UnitRemovedEvent",
    # Domain
    "PlatformState",
    "SensorState",
    "WeaponState",
    "TrackState",
    "EngagementState",
    "SimMetadata",
    "ControlCommand",
    "FireCommand",
    "ManeuverCommand",
    "DetectionEvent",
    "LaunchEvent",
    "HitEvent",
    "KillEvent",
    "LockChangeEvent",
    "ScenarioStartEvent",
    "ScenarioEndEvent",
    # Interfaces
    "Controller",
    "ScriptedController",
    "ControllerFactory",
    "ObservationHook",
    "RewardFunction",
    "SensorPlugin",
    "WeaponPlugin",
    "FlightModelPlugin",
    "VisualizerPlugin",
    "ProtocolAdapter",
    "FMUPlugin",
    # Schema
    "SCHEMA_VERSION",
    "ScenarioConfig",
    "SimulationConfig",
    "AgentConfig",
    "TeamConfig",
    "WeaponConfig",
    "SensorConfig",
    "validate_config",
    "migrate_config",
    # Registry
    "get_aircraft_class",
    "get_missile_class",
    # Tactical
    "NoEscapeZoneCalculator",
    "TrackPrioritySystem",
    "ObservationHelper",
    "OrbitConfig",
    "create_orbit_controller",
    "MissileParameters",
]
