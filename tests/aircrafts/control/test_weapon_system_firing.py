"""Deterministic regression tests for missile firing gates and spawn wiring.

These exercise ``AircraftWeaponSystem`` in isolation with a lightweight fake
parent / fake missile class / fake sim — no real simulator run, no randomness —
so they are fast and reproducible. They cover the weapon-handling guarantees
that are not already pinned by the substep / target-provider / kill-resolution
suites:

  * radar-lock requirement for a normal launch,
  * inventory (cooldown) gating and decrement,
  * non-engageable target veto,
  * target assignment onto the spawned missile,
  * launch-context capture (Phase 1) at fire time,
  * the lock-bypassing ``fire_missile_direct`` path.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from bvr_marl_core.aircraft.control.weapon_system import AircraftWeaponSystem
from bvr_marl_core.domain.information import WeaponTrack
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.simulator.core.helpers import Position
from tests.helpers.track_snapshot import track_snapshot


class _FakeMissile:
    """Minimal stand-in for a real Missile — records what the weapon system sets."""

    def __init__(self, firing_time_s, target, source, map_limits, group=None):
        self.firing_time_s = firing_time_s
        self.target = target
        self.source = source
        self.map_limits = map_limits
        self.group = group
        self.id = None
        self.is_missile = True
        self.name = "FakeMissile"
        self.designated_target_id = None
        self.retarget_policy = None
        self.initial_tracked_position_enu = None
        self.initial_tracked_velocity_enu = None
        self.tracker_reference_pos = None
        self.tracking_lock = False
        self.position = source.position.copy()
        self.radar = SimpleNamespace(
            locked_target=None,
            lock_ctrl=SimpleNamespace(set_mode=self._set_mode),
        )
        self.target_provider = SimpleNamespace(current_target_id=None)

    def _set_mode(self, mode, target_id=None):
        self._lock_mode = (mode, target_id)


class _FakeSim:
    """Fake simulator: assigns sequential ids on add_unit, exposes a clock."""

    def __init__(self):
        self.utc_time = 0.0
        self.elapsed_time_s = 5.0
        self.active_units = {}
        self._next_id = 100

    def add_unit(self, unit):
        unit.id = self._next_id
        self._next_id += 1
        self.active_units[unit.id] = unit
        return unit.id


def _make_parent(max_missiles=2, has_lock=True):
    """Build a fake shooter with just the attributes the weapon system reads."""
    sensor = SimpleNamespace(
        has_radar_lock=lambda target: has_lock,
        # dict form: target_id -> track entry (None => no seeded tracker state)
        get_locked_targets=lambda: {2: None},
    )
    parent = SimpleNamespace(
        position=Position(0.0, 0.0, 8000.0),
        yaw_deg=90.0,  # pointing east, toward the target
        pitch_deg=0.0,
        speed=300.0,
        group="blue",
        map_limits=SimpleNamespace(bottom_lat=-5.0, top_lat=5.0, left_lon=-5.0, right_lon=5.0),
        missile_types=[_FakeMissile],
        max_missiles=max_missiles,
        missiles=[],
        sensor=sensor,
        radar=SimpleNamespace(h_fov_deg=120.0, cached_tracks=[]),
        gun_config={},
        # no `wez` and no `remaining_missiles` attrs -> weapon system uses its own
    )
    return parent


def _make_target(engageable=True):
    return SimpleNamespace(
        id=2,
        group="red",
        position=Position(0.0, 0.1, 8000.0),  # ~11 km due east, in FOV
        yaw_deg=270.0,
        speed=250.0,
        is_non_engageable=not engageable,
    )


class TestFireMissileGates:
    def test_requires_radar_lock(self):
        ws = AircraftWeaponSystem(_make_parent(has_lock=False))
        missile, veto, diag = ws.fire_missile(_FakeSim(), _make_target(), _FakeMissile)
        assert missile is None
        assert "no_radar_lock" in veto
        assert diag["has_lock"] is False

    def test_non_engageable_target_vetoed(self):
        ws = AircraftWeaponSystem(_make_parent())
        missile, veto, diag = ws.fire_missile(
            _FakeSim(), _make_target(engageable=False), _FakeMissile
        )
        assert missile is None
        assert "non_engageable" in veto
        assert diag["is_engageable"] is False

    def test_inventory_exhausted_vetoed(self):
        ws = AircraftWeaponSystem(_make_parent(max_missiles=0))
        missile, veto, _ = ws.fire_missile(_FakeSim(), _make_target(), _FakeMissile)
        assert missile is None
        assert "no_inventory" in veto

    def test_successful_fire_assigns_target_and_decrements_inventory(self):
        ws = AircraftWeaponSystem(_make_parent(max_missiles=2))
        sim = _FakeSim()
        target = _make_target()

        missile, veto, diag = ws.fire_missile(sim, target, _FakeMissile)

        assert veto is None and missile is not None
        # Target assignment is threaded onto the spawned missile.
        assert missile.designated_target_id == 2
        assert missile.target_provider.current_target_id == 2
        assert missile.radar.locked_target == 2
        assert missile.tracking_lock is True
        # Spawned at the shooter pose, inherits the shooter group.
        assert missile.group == "blue"
        assert missile.position.alt == pytest.approx(8000.0)
        # Inventory (cooldown) decrements and the missile is registered.
        assert ws.remaining_missiles == 1
        assert missile.id in sim.active_units
        assert missile in ws.parent.missiles

    def test_launch_context_recorded_at_fire_time(self):
        ws = AircraftWeaponSystem(_make_parent())
        missile, _, _ = ws.fire_missile(_FakeSim(), _make_target(), _FakeMissile)

        ctx = missile._launch_context
        assert ctx["launch_time_s"] == pytest.approx(5.0)
        assert ctx["shooter_alt_m"] == pytest.approx(8000.0)
        assert ctx["shooter_speed_mps"] == pytest.approx(300.0)
        assert ctx["target_speed_mps"] == pytest.approx(250.0)
        assert ctx["range_m"] > 10_000.0  # ~11 km due east
        assert ctx["aspect_deg"] is not None

    def test_contact_launch_stores_only_weapon_track_in_operational_path(self):
        parent = _make_parent()
        parent.sensor.sensor_tracks = [
            track_snapshot(
                77,
                state=(1000.0, 0.0, 0.0, 250.0, 0.0, 0.0),
                source_ids=("wing-radar",),
                report_lineage=(("wing-radar", 1),),
            )
        ]
        contact = TacticalContact.from_track_snapshot(
            parent.sensor.sensor_tracks[0],
        )
        ws = AircraftWeaponSystem(parent)

        missile, veto, diagnostics = ws.fire_missile_at_contact(_FakeSim(), contact, _FakeMissile)

        assert veto is None and missile is not None
        assert isinstance(missile.target, WeaponTrack)
        assert missile.target.snapshot.track_id == 77
        assert diagnostics["target_id"] == 77
        assert diagnostics["selected_track_id"] == 77
        assert missile.weapon_track.snapshot.track_id == 77
        assert missile.weapon_track.snapshot.state[:3] == pytest.approx((1000.0, 0.0, 0.0))
        assert missile.weapon_track.snapshot.report_lineage == (("wing-radar", 1),)

    def test_contact_launch_state_is_invariant_to_evaluator_target_mutation(self):
        parent = _make_parent()
        target = _make_target()
        track = track_snapshot(
            77,
            state=(1000.0, 50.0, 10.0, -250.0, 5.0, 0.0),
        )
        parent.sensor.sensor_tracks = [track]
        contact = TacticalContact.from_track_snapshot(track)
        target.position.lon = 5.0
        ws = AircraftWeaponSystem(parent)

        missile, veto, _ = ws.fire_missile_at_contact(_FakeSim(), contact, _FakeMissile)

        assert veto is None
        assert missile.initial_tracked_position_enu == pytest.approx(contact.state[:3])
        assert missile.initial_tracked_velocity_enu == pytest.approx(contact.state[3:])

    def test_stale_contact_launch_does_not_register_missing_truth_target(self):
        parent = _make_parent()
        parent.id = 1
        parent.sensor.sensor_tracks = [
            track_snapshot(77, state=(1000.0, 0.0, 0.0, 250.0, 0.0, 0.0))
        ]
        contact = TacticalContact.from_track_snapshot(parent.sensor.sensor_tracks[0])
        sim = _FakeSim()
        sim.evaluator_truth_id_for_contact = lambda _sensor_id, _track_id: 99
        sim.register_weapon_truth_association = Mock()

        missile, veto, _ = AircraftWeaponSystem(parent).fire_missile_at_contact(
            sim, contact, _FakeMissile
        )

        assert veto is None and missile is not None
        sim.register_weapon_truth_association.assert_not_called()

    def test_contact_launch_keeps_current_truth_vote_provisional(self):
        """A valid contact vote must not be committed as the terminal victim."""
        parent = _make_parent()
        parent.id = 1
        parent.sensor.sensor_tracks = [
            track_snapshot(
                77,
                state=(1000.0, 0.0, 0.0, 250.0, 0.0, 0.0),
                report_lineage=(("wing-radar", 1),),
            )
        ]
        contact = TacticalContact.from_track_snapshot(parent.sensor.sensor_tracks[0])
        sim = _FakeSim()
        sim.active_units[2] = _make_target()
        sim.register_weapon_contact_association = Mock()
        sim.evaluator_truth_id_for_contact = Mock(return_value=2)
        sim.register_weapon_truth_association = Mock()

        missile, veto, _ = AircraftWeaponSystem(parent).fire_missile_at_contact(
            sim, contact, _FakeMissile
        )

        assert veto is None and missile is not None
        sim.register_weapon_contact_association.assert_called_once()
        sim.register_weapon_truth_association.assert_not_called()


class TestFireGunAtContact:
    def test_contact_fov_and_lead_use_estimated_state(self):
        parent = _make_parent()
        parent.velocity = SimpleNamespace(vx=200.0, vy=0.0, vz=0.0)
        ws = AircraftWeaponSystem(parent)
        ws.gun.start_firing = Mock(return_value=[object()])
        contact = TacticalContact(
            track_id=8,
            state=(1000.0, 0.0, 0.0, 100.0, 0.0, 0.0),
            covariance=tuple(tuple(row) for row in np.eye(6)),
            confidence=0.8,
            classification="aircraft",
        )

        result = ws.fire_gun_at_contact(_FakeSim(), contact)

        assert result
        target_tuple = ws.gun.start_firing.call_args.args[1]
        assert target_tuple[0] > parent.position.lon
        assert target_tuple[1] == pytest.approx(parent.position.lat, abs=1e-5)


class TestFireMissileDirect:
    def test_direct_fire_bypasses_lock_and_freezes_weapon_track(self):
        # No radar lock available, but direct fire must still succeed.
        ws = AircraftWeaponSystem(_make_parent(has_lock=False))
        sim = _FakeSim()
        target = _make_target()

        missile, veto, _ = ws.fire_missile_direct(sim, target, _FakeMissile)

        assert veto is None and missile is not None
        assert missile.designated_target_id == 2
        assert missile.retarget_policy == "track_only"
        assert isinstance(missile.target, WeaponTrack)
        assert missile.target.snapshot.track_id == target.id
        assert ws.remaining_missiles == 1
        assert missile._launch_context["range_m"] > 10_000.0

    def test_direct_fire_respects_inventory_and_engageability(self):
        ws_empty = AircraftWeaponSystem(_make_parent(max_missiles=0))
        m, veto, _ = ws_empty.fire_missile_direct(_FakeSim(), _make_target(), _FakeMissile)
        assert m is None and "no_inventory" in veto

        ws = AircraftWeaponSystem(_make_parent())
        m, veto, _ = ws.fire_missile_direct(
            _FakeSim(), _make_target(engageable=False), _FakeMissile
        )
        assert m is None and "non_engageable" in veto
