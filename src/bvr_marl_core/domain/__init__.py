"""Canonical state/command/event domain model for bvr_marl_core.

Architectural role
------------------
The domain model is the **public DTO (Data Transfer Object) layer** between
the core simulation engine and all external consumers (extension packages,
protocol adapters, visualizers, replay recorders, etc.).

It is NOT the internal runtime state of the simulation.  Inside the core the
simulator operates on concrete implementation objects (Aircraft, Missile,
Radar, …).  At the public boundary those objects are **translated** into the
domain DTOs defined here before being handed to any external code.

Consumers must
~~~~~~~~~~~~~~
* Import state, command, and event types exclusively from this module (or
  from ``bvr_marl_core`` directly).
* Never assume that a DTO field maps one-to-one to an internal field name —
  the mapping is an explicit, versioned contract.

Core internals must
~~~~~~~~~~~~~~~~~~~
* Construct and return domain DTOs at every public boundary (simulator step
  outputs, observation hooks, replay recorder calls).
* Never expose raw internal objects across the boundary.

Adding fields to a DTO
~~~~~~~~~~~~~~~~~~~~~~
1. Update the dataclass in state.py / commands.py / events.py.
2. Update the schema_version in schema/version.py if the change is
   externally visible and backwards-incompatible.
3. Update all sites that construct the affected DTO.
"""

from .commands import (
    ControlCommand,
    FireCommand,
    ManeuverCommand,
)
from .events import (
    DetectionEvent,
    HitEvent,
    KillEvent,
    LaunchEvent,
    LockChangeEvent,
    ScenarioEndEvent,
    ScenarioStartEvent,
)
from .state import (
    EngagementState,
    PlatformState,
    SensorState,
    SimMetadata,
    TrackState,
    WeaponState,
)

__all__ = [
    # state
    "PlatformState",
    "SensorState",
    "WeaponState",
    "TrackState",
    "EngagementState",
    "SimMetadata",
    # commands
    "ControlCommand",
    "FireCommand",
    "ManeuverCommand",
    # events
    "DetectionEvent",
    "LaunchEvent",
    "HitEvent",
    "KillEvent",
    "LockChangeEvent",
    "ScenarioStartEvent",
    "ScenarioEndEvent",
]
