"""Architectural contracts for sensor-limited operational layers."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.domain.information_mode import TruthAccessError, audit_sensor_limited_simulator
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.domain.team_information import FriendlyDatalinkReport
from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
from bvr_marl_core.rl.environment.spaces.action_space.automation import (
    MissileAutomation,
    WeaponCooldowns,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors import TriggerProcessor
from bvr_marl_core.rl.environment.spaces.action_space.utils.target_sorting import TargetSorter
from bvr_marl_core.rl.environment.spaces.action_space.weapon_firing import WeaponFiringHandler
from bvr_marl_core.rl.environment.spaces.observation.missile_warning_builder import (
    MissileWarningBuilder,
)

ROOT = Path(__file__).parents[2] / "src" / "bvr_marl_core"
STRICT_MODULES = (
    ROOT / "domain" / "information_mode.py",
    ROOT / "domain" / "tactical_contact.py",
    ROOT / "domain" / "team_information.py",
    ROOT / "radar" / "core" / "friendly_picture.py",
)
PROHIBITED_IMPORTS = (
    "bvr_marl_core.aircraft.aircraft",
    "bvr_marl_core.missiles.missile",
    "bvr_marl_core.simulator.core.units",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _contact(track_id=4):
    return TacticalContact(
        track_id=track_id,
        state=(1000.0, 0.0, 0.0, -200.0, 0.0, 0.0),
        covariance=tuple(tuple(row) for row in np.eye(6)),
        confidence=0.8,
        classification="aircraft",
    )


def test_strict_operational_modules_do_not_import_entity_implementations():
    violations = {
        str(path.relative_to(ROOT)): sorted(
            module for module in _imports(path) if module.startswith(PROHIBITED_IMPORTS)
        )
        for path in STRICT_MODULES
    }
    assert not {path: imports for path, imports in violations.items() if imports}


def test_operational_contact_and_team_reports_have_no_entity_handle_fields():
    prohibited = {"target", "unit", "entity", "missile", "aircraft", "object"}
    for record_type in (TacticalContact, FriendlyDatalinkReport):
        names = {field.name.lower() for field in fields(record_type)}
        assert names.isdisjoint(prohibited)


def test_sensor_limited_selection_and_warning_methods_do_not_scan_roster():
    source = inspect.getsource(TargetSorter.select_contact)
    warning_source = inspect.getsource(
        MissileWarningBuilder._build_from_warning_and_track_estimates
    )
    assert "active_units" not in source
    assert "active_units" not in warning_source


def test_weapon_track_guidance_update_has_no_evaluator_roster_or_target_object_access():
    source = inspect.getsource(GuidanceTargetProvider._update_from_weapon_track)
    assert "active_units" not in source
    assert "missile.target" not in source
    assert "target_obj" not in source


def test_operational_radar_lock_uses_anonymous_contact_ids():
    from bvr_marl_core.radar.radar import Radar
    from bvr_marl_core.radar.units.aircraft import AircraftRadar

    for method in (Radar._update_lock_ctrl, AircraftRadar._update_lock_ctrl):
        source = inspect.getsource(method)
        assert "truth_id" not in source
        assert "getattr(tgt" not in source


def test_contact_saturation_uses_only_operational_simulator_registry():
    calls = []
    simulator = SimpleNamespace(
        count_missiles_at_target=lambda *args: (_ for _ in ()).throw(
            AssertionError("simulator truth registry accessed")
        ),
        count_missiles_at_contact=lambda *args: calls.append(args) or 2,
    )
    automation = MissileAutomation(max_per_target=2)
    cooldowns = WeaponCooldowns(max_missiles_per_target=2)
    handler = WeaponFiringHandler(automation, cooldowns, TriggerProcessor())
    handler.simulator = simulator
    unit = SimpleNamespace(group="blue")
    contact = _contact(17)

    assert handler._target_saturated(unit, contact, 17) is True
    assert calls == [("blue", None, 17, contact.report_lineage)]


def test_runtime_audit_rejects_live_target_retained_in_operational_track():
    live_target = SimpleNamespace(
        id=9,
        position=SimpleNamespace(),
        velocity=SimpleNamespace(),
    )
    radar = SimpleNamespace(cached_tracks=[(1, (), (), live_target)])
    simulator = SimpleNamespace(active_units={1: SimpleNamespace(radar=radar)})

    with pytest.raises(TruthAccessError, match="radar.track"):
        audit_sensor_limited_simulator(simulator)
