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

from bvr_marl_core.domain.commands import ControlCommand, FireCommand, ManeuverCommand
from bvr_marl_core.domain.events import (
    DetectionEvent,
    HitEvent,
    KillEvent,
    LaunchEvent,
    LockChangeEvent,
    ScenarioEndEvent,
    ScenarioStartEvent,
)
from bvr_marl_core.domain.information import (
    ReportId,
    ReportLineage,
    SensorId,
    SensorReport,
    TrackSnapshot,
    WeaponTrack,
)
from bvr_marl_core.domain.state import (
    EngagementState,
    PlatformState,
    SensorState,
    SimMetadata,
    TrackState,
    WeaponState,
)
from bvr_marl_core.domain.tactical_contact import (
    DEFAULT_CONTACT_SLOTS,
    PINNED_ACTION_LOG_STD,
    TacticalContact,
    action_value_for_contact_slot,
    entropy_of_pinned_axes,
)
from bvr_marl_core.domain.track_confidence import (
    CONFIDENCE_BANDS,
    TRACK_CONFIDENCE_COMMIT,
    TRACK_CONFIDENCE_FIRM,
    TRACK_CONFIDENCE_PROBABLE,
    TRACK_CONFIDENCE_SHOOT,
    TRACK_CONFIDENCE_TENTATIVE,
    confidence_band,
)

__all__ = [
    "CONFIDENCE_BANDS",
    "TRACK_CONFIDENCE_COMMIT",
    "TRACK_CONFIDENCE_FIRM",
    "TRACK_CONFIDENCE_PROBABLE",
    "TRACK_CONFIDENCE_SHOOT",
    "TRACK_CONFIDENCE_TENTATIVE",
    "confidence_band",
    # state
    "PlatformState",
    "SensorState",
    "WeaponState",
    "TrackState",
    "EngagementState",
    "SimMetadata",
    "TacticalContact",
    # Target-slot action mapping: callers that pin or initialize the target-selection
    # action must derive the value from the same binning `select` uses.
    "DEFAULT_CONTACT_SLOTS",
    "PINNED_ACTION_LOG_STD",
    "action_value_for_contact_slot",
    "entropy_of_pinned_axes",
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
    # operational information
    "SensorId",
    "ReportId",
    "ReportLineage",
    "SensorReport",
    "TrackSnapshot",
    "WeaponTrack",
]
