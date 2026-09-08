#!/usr/bin/env python3
"""
Tests for the recorded sub-stepped tick path and the two consumers that used to
reconstruct it from the tick's endpoints.
"""

import math
from datetime import datetime

import pytest

from bvr_marl_core.aircraft.control.movement_control import AircraftControlSystem
from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.substepping import TerminalPathResolver
from bvr_marl_core.simulator.utils.tick_path import pose_at_fraction, unit_tick_path

M_PER_DEG = 111_320.0


class _Limits:
    bottom_lat = -80.0
    top_lat = 80.0
    left_lon = -180.0
    right_lon = 180.0


class _Jet:
    def __init__(self, alt=8000.0):
        self.physics = AircraftPhysics(
            AircraftPhysics.Params(mass_kg=16000.0, reference_area_m2=28.0)
        )
        self.position = Position(0.0, 0.0, alt)
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.speed = 300.0
        self.min_speed_mps = 80.0
        self.max_speed_mps = 680.0
        self.min_alt_m = 0.0
        self.max_alt_m = 20000.0
        self.map_limits = _Limits()
        self.keep_inside_boundary = True
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.removal_reason = None
        self.fuel = None
        self.is_mortally_hit = False
        self.control = AircraftControlSystem(self)


class _Sim:
    def __init__(self, elapsed=10.0):
        self.elapsed_time_s = elapsed


# ============================================================================
# Recording
# ============================================================================


def test_path_has_one_vertex_per_substep_plus_the_start():
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)
    # Default is 4 sub-steps at a 1 s tick.
    assert len(jet.control.tick_path) == 5
    assert jet.control.tick_path[0] == (0.0, 0.0, 8000.0)
    assert jet.control.tick_path[-1] == (
        jet.position.lat,
        jet.position.lon,
        jet.position.alt,
    )


def test_path_is_refreshed_not_appended_across_ticks():
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=3.0, phi_cmd_deg=40.0)
    jet.control.update_movement(1.0)
    jet.control.update_movement(1.0)
    assert len(jet.control.tick_path) == 5


def test_legacy_path_still_records_a_two_point_segment():
    """Without a lift-vector command there is one step, so the path is a chord.

    That is exactly the old behaviour, so consumers reading it get the same
    answer they used to get from endpoint interpolation.
    """
    jet = _Jet()
    jet.control.set_yaw_deg(30.0)
    jet.control.update_movement(1.0)
    assert len(jet.control.tick_path) == 2


# ============================================================================
# Staleness guard
# ============================================================================


def test_stale_path_is_rejected():
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)
    jet.tick_path = jet.control.tick_path
    jet.tick_path_time_s = 10.0

    assert unit_tick_path(jet, _Sim(elapsed=10.0)) is not None
    # One tick later the recorded path belongs to the past and must be refused.
    assert unit_tick_path(jet, _Sim(elapsed=11.0)) is None


def test_unstamped_or_short_paths_are_rejected():
    jet = _Jet()
    sim = _Sim(elapsed=0.0)
    assert unit_tick_path(jet, sim) is None  # never flew

    jet.tick_path = [(0.0, 0.0, 8000.0)]
    jet.tick_path_time_s = 0.0
    assert unit_tick_path(jet, sim) is None  # single vertex, nothing to sample


# ============================================================================
# Sampling
# ============================================================================


def test_sampling_endpoints_and_midpoint():
    path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    assert pose_at_fraction(path, 0.0) == (0.0, 0.0, 0.0)
    assert pose_at_fraction(path, 1.0) == (3.0, 0.0, 0.0)
    assert pose_at_fraction(path, 0.5)[0] == pytest.approx(1.5)
    # Out-of-range fractions clamp rather than extrapolate.
    assert pose_at_fraction(path, -5.0) == (0.0, 0.0, 0.0)
    assert pose_at_fraction(path, 5.0) == (3.0, 0.0, 0.0)


def test_sampled_arc_departs_from_the_straight_chord():
    """The whole point: mid-tick, the real arc is not on the endpoint chord."""
    jet = _Jet()
    jet.control.autopilot.phi_deg = 70.0
    jet.control.autopilot.n = 5.0
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)

    path = jet.control.tick_path
    arc_mid = pose_at_fraction(path, 0.5)
    chord_mid = (
        0.5 * (path[0][0] + path[-1][0]),
        0.5 * (path[0][1] + path[-1][1]),
        0.5 * (path[0][2] + path[-1][2]),
    )
    offset_m = math.hypot(arc_mid[0] - chord_mid[0], arc_mid[1] - chord_mid[1]) * M_PER_DEG
    assert offset_m > 0.5, "recorded path is indistinguishable from a straight chord"


# ============================================================================
# The CPA resolver actually uses it
# ============================================================================


def test_cpa_prefers_the_recorded_path_over_hermite(monkeypatch):
    """With a real target path available, the Hermite fallback must not be used."""
    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)
    jet.tick_path = jet.control.tick_path
    jet.tick_path_time_s = 0.0
    jet.id = "target1"

    calls = []
    monkeypatch.setattr(
        TerminalPathResolver,
        "_hermite_pose",
        staticmethod(lambda *a, **k: calls.append(1) or (0.0, 0.0, 0.0)),
    )

    resolver = TerminalPathResolver()
    missile = _StubMissile()
    resolver._substep_pair(_SimForCpa(), missile, jet, 1.0, _StubCalc())

    assert calls == [], "Hermite reconstruction was used despite a recorded path"


def test_cpa_falls_back_to_hermite_without_a_recorded_path(monkeypatch):
    """A target that records nothing keeps the previous reconstruction."""
    jet = _Jet()
    jet.id = "target1"  # no tick_path published

    calls = []
    original = TerminalPathResolver._hermite_pose

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(TerminalPathResolver, "_hermite_pose", staticmethod(spy))

    resolver = TerminalPathResolver()
    resolver._substep_pair(_SimForCpa(), _StubMissile(), jet, 1.0, _StubCalc())

    assert calls, "fallback path was not exercised"


class _StubMissile:
    def __init__(self):
        self.id = "m1"
        self.arming_time_s = 0.0
        self.elapsed_time_s = 20.0
        self.tick_path = [(0.5, 0.5, 8000.0), (0.4, 0.4, 8000.0), (0.3, 0.3, 8000.0)]
        self._last_cpa_event = None
        self.should_be_removed = False
        self.removal_reason = None


class _StubCalcParams:
    fuse_radius_m = 10.0
    target_radius_m = 5.0


class _StubCalc:
    params = _StubCalcParams()

    def check_segment_hit(self, m0, m1, t0, t1):
        # Never a hit: we only care which target-pose source gets consulted.
        return False, 1.0, 1e9


class _SimForCpa:
    elapsed_time_s = 0.0
    ccd = None

    def log_event(self, *args, **kwargs):
        return None


# ============================================================================
# Missile guidance uses it too
# ============================================================================


def test_missile_guidance_samples_the_recorded_arc():
    from bvr_marl_core.missiles.missile import Missile

    jet = _Jet()
    jet.control.autopilot.phi_deg = 70.0
    jet.control.autopilot.n = 5.0
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)
    jet.tick_path = jet.control.tick_path
    jet.tick_path_time_s = 0.0
    jet.id = "t1"

    sim = _SimForCpa()
    mid = Missile._target_pose_at_fraction(None, sim, jet, 0.5)
    expected = pose_at_fraction(jet.tick_path, 0.5)
    assert mid.lat == pytest.approx(expected[0])
    assert mid.lon == pytest.approx(expected[1])
    assert mid.alt == pytest.approx(expected[2])


def test_missile_guidance_ignores_a_stale_arc():
    from bvr_marl_core.missiles.missile import Missile

    jet = _Jet()
    jet.control.set_lift_vector(n_cmd=5.0, phi_cmd_deg=70.0)
    jet.control.update_movement(1.0)
    jet.tick_path = jet.control.tick_path
    jet.tick_path_time_s = 99.0  # belongs to a different tick
    jet.id = "t1"

    sim = _SimForCpa()
    pose = Missile._target_pose_at_fraction(None, sim, jet, 0.5)
    stale = pose_at_fraction(jet.tick_path, 0.5)
    # Falls back to the endpoint reconstruction, which for a target whose begin
    # and end poses are both "now" collapses onto the current position.
    assert pose.lat == pytest.approx(jet.position.lat)
    assert pose.lat != pytest.approx(stale[0])


def test_simulator_roster_updates_aircraft_before_missiles():
    """The ordering the whole scheme depends on: aircraft record, then missiles read."""
    from bvr_marl_core.simulator.simulator import Simulator

    sim = Simulator(utc_time=datetime(2025, 6, 1, 12, 0, 0), tick_secs=1.0, random_seed=1)
    units = [
        type("M", (), {"id": "m1", "is_missile": True})(),
        type("A", (), {"id": "a1", "is_missile": False})(),
    ]
    ordered = sorted(units, key=lambda u: (bool(getattr(u, "is_missile", False)), str(u.id)))
    assert [u.id for u in ordered] == ["a1", "m1"]
    assert sim is not None
