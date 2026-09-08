"""API snapshot test: verify the bvr_marl_core public surface is stable.

This test ensures that every symbol documented in bvr_marl_core.__all__ is
importable and that no symbol disappears silently between refactors.  If a
symbol is intentionally removed from the public API, remove it from
EXPECTED_PUBLIC_API below (with a corresponding CHANGES.md entry).
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

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
    # Observation-space fixed slot counts
    "K_FF",
    "K_EF",
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


@pytest.mark.smoke
def test_public_subpackage_attribute_access():
    """Public submodules must be reachable as attributes under lazy init.

    Eager init bound these as a side effect of importing the package; lazy init
    must keep ``bvr_marl_core.interfaces`` etc. working without an explicit
    ``import bvr_marl_core.interfaces`` first.
    """
    import bvr_marl_core

    assert bvr_marl_core.interfaces.Controller is bvr_marl_core.Controller
    assert bvr_marl_core.domain.PlatformState is bvr_marl_core.PlatformState
    assert bvr_marl_core.simulator.Simulator is bvr_marl_core.Simulator

    # Unknown attributes still raise AttributeError (not ImportError/KeyError).
    with pytest.raises(AttributeError):
        bvr_marl_core.definitely_not_a_real_attribute


@pytest.mark.smoke
def test_public_api_manifest_modules_importable():
    """Every module in the published allowlist must actually be importable."""
    from bvr_marl_core.public_api import PUBLIC_API_MODULES

    failures = []
    for module in PUBLIC_API_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - failure path
            failures.append(f"{module}: {exc!r}")

    assert failures == [], "Published public API modules failed to import:\n" + "\n".join(
        f"  {f}" for f in failures
    )


@pytest.mark.smoke
def test_public_api_contract_rules():
    """The published import-boundary rules behave as documented."""
    from bvr_marl_core.public_api import (
        PUBLIC_API_MODULES,
        is_public_import,
        violation_for,
    )

    # The contract module and the flat top-level surface are both public.
    assert "bvr_marl_core" in PUBLIC_API_MODULES
    assert "bvr_marl_core.public_api" in PUBLIC_API_MODULES

    # Top-level matches by exact name only — never as a wildcard over internals.
    assert is_public_import("bvr_marl_core")
    assert not is_public_import("bvr_marl_core.rl.environment.gym.bvr_multi_agent_env")

    # Sub-modules of an approved package are allowed.
    assert is_public_import("bvr_marl_core.interfaces.controller")

    # Out-of-namespace imports are never violations; private paths always are.
    assert violation_for("numpy") is None
    assert violation_for("bvr_marl_core._private") is not None


@pytest.mark.smoke
def test_public_api_stub_matches_manifest():
    """The __init__.pyi type stub must re-export exactly the runtime manifest.

    The stub gives type checkers / IDEs static visibility of the flat surface
    that ``__init__.py`` resolves lazily.  This test parses the stub's explicit
    re-exports (``X as X``) and asserts they match ``_PUBLIC_ATTRS`` symbol for
    symbol and module for module, so the two can never drift.
    """
    import bvr_marl_core

    manifest = {name: module for name, (module, _attr) in bvr_marl_core._PUBLIC_ATTRS.items()}

    stub_path = Path(bvr_marl_core.__file__).with_suffix(".pyi")
    assert stub_path.exists(), f"Missing type stub: {stub_path}"

    tree = ast.parse(stub_path.read_text(encoding="utf-8"))

    reexports: dict[str, str] = {}
    non_reexports: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        # Relative import in the stub (level == 1) -> bvr_marl_core.<module>
        module = f"bvr_marl_core.{node.module}" if node.level else node.module
        for alias in node.names:
            # PEP 484 explicit re-export requires `name as name`.
            if alias.asname != alias.name:
                non_reexports.append(f"{module}.{alias.name}")
                continue
            reexports[alias.name] = module

    assert not non_reexports, (
        "Stub imports must use explicit `X as X` re-export form:\n"
        + "\n".join(f"  {n}" for n in non_reexports)
    )

    missing = set(manifest) - set(reexports)
    extra = set(reexports) - set(manifest)
    assert not missing and not extra, (
        "__init__.pyi is out of sync with _PUBLIC_ATTRS.\n"
        f"  Missing from stub: {sorted(missing)}\n"
        f"  Extra in stub:     {sorted(extra)}"
    )

    wrong_module = {
        name: (reexports[name], manifest[name])
        for name in manifest
        if reexports[name] != manifest[name]
    }
    assert not wrong_module, (
        "Stub re-exports a symbol from the wrong module (stub, manifest):\n"
        + "\n".join(f"  {name}: {pair}" for name, pair in sorted(wrong_module.items()))
    )

    # __all__ in the stub must also match the manifest exactly.
    stub_all = next(
        (
            {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
        ),
        None,
    )
    assert stub_all == set(manifest), (
        "__init__.pyi __all__ does not match the manifest.\n"
        f"  Missing: {sorted(set(manifest) - (stub_all or set()))}\n"
        f"  Extra:   {sorted((stub_all or set()) - set(manifest))}"
    )


@pytest.mark.smoke
def test_public_api_contract_import_is_lightweight():
    """Importing the contract module must not import runtime-heavy core modules."""
    code = """
import sys
from bvr_marl_core.public_api import violation_for

assert violation_for("numpy") is None
unexpected = [
    module
    for module in (
        "bvr_marl_core.interfaces",
        "bvr_marl_core.registry",
        "bvr_marl_core.simulator",
        "numpy",
    )
    if module in sys.modules
]
assert unexpected == [], unexpected
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
