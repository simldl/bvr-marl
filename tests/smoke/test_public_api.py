"""API snapshot test: verify the bvr_marl_core public surface is stable.

This test ensures that every symbol documented in bvr_marl_core.__all__ is
importable and that no symbol disappears silently between refactors.  If a
symbol is intentionally removed from the public API, remove it from
EXPECTED_PUBLIC_API below (with a corresponding CHANGES.md entry).
"""

import importlib

import pytest

# Canonical list of public symbols.  Update here when the public API changes.
EXPECTED_PUBLIC_API: set[str] = {
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
    # Domain state
    "PlatformState",
    "SensorState",
    "WeaponState",
    "TrackState",
    "EngagementState",
    "SimMetadata",
    # Domain commands
    "ControlCommand",
    "FireCommand",
    "ManeuverCommand",
    # Domain events
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
}


@pytest.mark.smoke
def test_public_api_complete():
    """Every expected public symbol must be importable from bvr_marl_core."""
    import bvr_marl_core

    missing = []
    for name in sorted(EXPECTED_PUBLIC_API):
        if not hasattr(bvr_marl_core, name):
            missing.append(name)

    assert missing == [], (
        "The following expected public API symbols are missing from bvr_marl_core:\n"
        + "\n".join(f"  {n}" for n in missing)
        + "\n\nIf intentionally removed, also update EXPECTED_PUBLIC_API in this test "
        "and add a CHANGES.md entry."
    )


@pytest.mark.smoke
def test_public_api_no_undeclared_drift():
    """bvr_marl_core.__all__ must match the expected public API set exactly."""
    import bvr_marl_core

    declared = set(bvr_marl_core.__all__)
    extra = declared - EXPECTED_PUBLIC_API
    missing = EXPECTED_PUBLIC_API - declared

    assert not extra and not missing, (
        f"Public API mismatch.\n"
        f"  Added without updating EXPECTED_PUBLIC_API: {sorted(extra)}\n"
        f"  Removed without updating EXPECTED_PUBLIC_API: {sorted(missing)}"
    )


@pytest.mark.smoke
def test_public_api_all_importable():
    """Every symbol in __all__ must actually be an attribute of the package."""
    import bvr_marl_core

    for name in bvr_marl_core.__all__:
        assert hasattr(bvr_marl_core, name), (
            f"bvr_marl_core.__all__ lists '{name}' but it is not accessible as an attribute"
        )
