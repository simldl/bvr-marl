"""Type stub for the bvr_marl_core flat public API.

At runtime these symbols are resolved lazily via module ``__getattr__`` (PEP 562)
so importing the package stays cheap.  This stub mirrors that surface statically
so type checkers and IDEs can see it.

It must stay in lock-step with ``_PUBLIC_ATTRS`` in ``__init__.py``; the
``test_public_api_stub_matches_manifest`` smoke test fails if they drift.
"""

from .domain import (
    ControlCommand as ControlCommand,
    DetectionEvent as DetectionEvent,
    EngagementState as EngagementState,
    FireCommand as FireCommand,
    HitEvent as HitEvent,
    KillEvent as KillEvent,
    LaunchEvent as LaunchEvent,
    LockChangeEvent as LockChangeEvent,
    ManeuverCommand as ManeuverCommand,
    PlatformState as PlatformState,
    ScenarioEndEvent as ScenarioEndEvent,
    ScenarioStartEvent as ScenarioStartEvent,
    SensorState as SensorState,
    SimMetadata as SimMetadata,
    TrackState as TrackState,
    WeaponState as WeaponState,
)
from .interfaces import (
    Controller as Controller,
    ControllerFactory as ControllerFactory,
    FlightModelPlugin as FlightModelPlugin,
    FMUPlugin as FMUPlugin,
    ObservationHook as ObservationHook,
    ProtocolAdapter as ProtocolAdapter,
    RewardFunction as RewardFunction,
    ScriptedController as ScriptedController,
    SensorPlugin as SensorPlugin,
    VisualizerPlugin as VisualizerPlugin,
    WeaponPlugin as WeaponPlugin,
)
from .registry import (
    get_aircraft_class as get_aircraft_class,
    get_missile_class as get_missile_class,
)
from .schema import (
    SCHEMA_VERSION as SCHEMA_VERSION,
    AgentConfig as AgentConfig,
    ScenarioConfig as ScenarioConfig,
    SensorConfig as SensorConfig,
    SimulationConfig as SimulationConfig,
    TeamConfig as TeamConfig,
    WeaponConfig as WeaponConfig,
    migrate_config as migrate_config,
    validate_config as validate_config,
)
from .simulator import (
    MapLimits as MapLimits,
    Position as Position,
    Simulator as Simulator,
    UnitDestroyedEvent as UnitDestroyedEvent,
    UnitRegisteredEvent as UnitRegisteredEvent,
    UnitRemovedEvent as UnitRemovedEvent,
    geodetic_bearing_deg as geodetic_bearing_deg,
    geodetic_distance_km as geodetic_distance_km,
    normalize_angle as normalize_angle,
    signed_yaw_deg_diff as signed_yaw_deg_diff,
)
from .tactical import (
    MissileParameters as MissileParameters,
    NoEscapeZoneCalculator as NoEscapeZoneCalculator,
    ObservationHelper as ObservationHelper,
    OrbitConfig as OrbitConfig,
    TrackPrioritySystem as TrackPrioritySystem,
    create_orbit_controller as create_orbit_controller,
)
from .rl.environment.spaces.observation.constants import (
    K_EF as K_EF,
    K_FF as K_FF,
)

__all__ = [
    "AgentConfig",
    "ControlCommand",
    "Controller",
    "ControllerFactory",
    "DetectionEvent",
    "EngagementState",
    "FMUPlugin",
    "FireCommand",
    "FlightModelPlugin",
    "HitEvent",
    "K_EF",
    "K_FF",
    "KillEvent",
    "LaunchEvent",
    "LockChangeEvent",
    "ManeuverCommand",
    "MapLimits",
    "MissileParameters",
    "NoEscapeZoneCalculator",
    "ObservationHelper",
    "ObservationHook",
    "OrbitConfig",
    "PlatformState",
    "Position",
    "ProtocolAdapter",
    "RewardFunction",
    "SCHEMA_VERSION",
    "ScenarioConfig",
    "ScenarioEndEvent",
    "ScenarioStartEvent",
    "ScriptedController",
    "SensorConfig",
    "SensorPlugin",
    "SensorState",
    "SimMetadata",
    "SimulationConfig",
    "Simulator",
    "TeamConfig",
    "TrackPrioritySystem",
    "TrackState",
    "UnitDestroyedEvent",
    "UnitRegisteredEvent",
    "UnitRemovedEvent",
    "VisualizerPlugin",
    "WeaponConfig",
    "WeaponPlugin",
    "WeaponState",
    "create_orbit_controller",
    "geodetic_bearing_deg",
    "geodetic_distance_km",
    "get_aircraft_class",
    "get_missile_class",
    "migrate_config",
    "normalize_angle",
    "signed_yaw_deg_diff",
    "validate_config",
]
