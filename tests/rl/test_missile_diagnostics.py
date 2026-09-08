"""Campaign-side answer to "why did the shots we took not kill anything?".

Standalone harness testing put the weapon chain at ~90% kills against exactly the target
an early warmup stage uses -- stationary, 15-100 km, fired through the real tracking chain, with and
without midcourse support. The campaign fires ~1200 missiles for ~12 kills. Nothing
reproducible outside a live episode explains that gap, so the launch conditions the
policy actually chooses have to be measured where they happen.

The headline metric is deliberately NOT the kill rate. ``MissileTerminalEvent`` is
emitted once per PROXIMITY DETONATION, so a missile that flies past without tripping its
fuze -- out of energy, guidance lost, seeker never acquired -- emits nothing at all.
``detonation_rate`` separates "the warhead fired and missed" from "the missile never
arrived", which have different causes and different fixes, and which no existing metric
distinguishes.
"""

from __future__ import annotations

import math

import pytest

from bvr_marl_core.domain.launch_geometry import (
    DETONATION_RATE,
    LAUNCH_ASPECT_DEG,
    LAUNCH_CLOSURE_MPS,
    LAUNCH_RANGE_KM,
    MISSILE_DIAGNOSTIC_KEYS,
    NO_TERMINAL_RATE,
    TERMINAL_MISS_M,
    TERMINAL_PK,
    capture_launch_geometry,
)
from bvr_marl_core.rl.environment.gym.gym_components.missile_diagnostics import (
    MissileDiagnosticsCollector,
)

_M_PER_DEG = 111_320.0


class _Pos:
    def __init__(self, lat, lon, alt):
        self.lat, self.lon, self.alt = lat, lon, alt


class _Vel:
    def __init__(self, vx, vy, vz):
        self.vx, self.vy, self.vz = vx, vy, vz


class _Shooter:
    def __init__(self, yaw_deg=90.0, vel=(300.0, 0.0, 0.0), alt=9000.0):
        self.position = _Pos(0.0, 0.0, alt)
        self.yaw_deg = yaw_deg
        self.velocity = _Vel(*vel)


def _target_east(km, alt=9000.0):
    return _Pos(0.0, km * 1000.0 / _M_PER_DEG, alt)


class _Event:
    """Duck-typed MissileTerminalEvent: the collector matches on class NAME."""

    def __init__(self, record):
        self.record = record


_Event.__name__ = "MissileTerminalEvent"


class _Sim:
    def __init__(self, events=()):
        self.events = list(events)


def test_launch_geometry_measures_range_aspect_and_closure():
    shooter = _Shooter(yaw_deg=90.0, vel=(300.0, 0.0, 0.0))  # nose east, flying east

    geometry = capture_launch_geometry(shooter, _target_east(40.0))

    assert geometry[LAUNCH_RANGE_KM] == pytest.approx(40.0, rel=1e-3)
    assert geometry[LAUNCH_ASPECT_DEG] == pytest.approx(0.0, abs=1.0)  # nose-on
    assert geometry[LAUNCH_CLOSURE_MPS] == pytest.approx(300.0, rel=1e-2)  # closing


def test_a_cold_shot_is_distinguishable_from_a_hot_one():
    """The whole point: the harness fires hot, and a policy may not be."""
    hot = capture_launch_geometry(_Shooter(yaw_deg=90.0, vel=(300.0, 0.0, 0.0)), _target_east(40.0))
    cold = capture_launch_geometry(
        _Shooter(yaw_deg=270.0, vel=(-300.0, 0.0, 0.0)), _target_east(40.0)
    )

    assert hot[LAUNCH_ASPECT_DEG] < 10.0
    assert cold[LAUNCH_ASPECT_DEG] > 170.0
    assert hot[LAUNCH_CLOSURE_MPS] > 0 > cold[LAUNCH_CLOSURE_MPS]


def test_no_target_position_is_survivable():
    assert capture_launch_geometry(_Shooter(), None) is None


def test_detonation_rate_separates_never_arrived_from_missed():
    """Three launches, one detonation: the other two left no trace anywhere else."""
    collector = MissileDiagnosticsCollector()
    for _ in range(3):
        collector.record_launch({LAUNCH_RANGE_KM: 40.0})
    collector.collect_terminal_events(
        _Sim([_Event({"miss_distance_m": 12.0, "pk": 0.9, "killed": True})])
    )

    metrics = collector.episode_metrics()

    assert metrics[DETONATION_RATE] == pytest.approx(1 / 3)
    assert metrics[NO_TERMINAL_RATE] == pytest.approx(2 / 3)
    assert metrics[TERMINAL_MISS_M] == pytest.approx(12.0)
    assert metrics[TERMINAL_PK] == pytest.approx(0.9)


def test_draining_events_is_idempotent_across_steps():
    """Called every step against a cumulative event list; must not double count."""
    collector = MissileDiagnosticsCollector()
    collector.record_launch(None)
    sim = _Sim([_Event({"miss_distance_m": 5.0})])

    for _ in range(10):
        collector.collect_terminal_events(sim)

    assert collector.episode_metrics()[DETONATION_RATE] == pytest.approx(1.0)


def test_an_episode_with_no_shots_reports_no_launch_geometry():
    """Logging 0.0 would read as 'fires at point blank', the opposite of the truth."""
    metrics = MissileDiagnosticsCollector().episode_metrics()

    assert LAUNCH_RANGE_KM not in metrics
    assert DETONATION_RATE not in metrics


def test_reset_clears_everything_between_episodes():
    collector = MissileDiagnosticsCollector()
    collector.record_launch({LAUNCH_RANGE_KM: 40.0})
    collector.collect_terminal_events(_Sim([_Event({"miss_distance_m": 5.0})]))

    collector.reset()

    assert collector.episode_metrics() == {}


def test_every_emitted_key_is_declared_for_the_csv_header():
    """Ray Tune locks progress.csv's header from the first result; see the metrics fix."""
    collector = MissileDiagnosticsCollector()
    collector.record_launch(capture_launch_geometry(_Shooter(), _target_east(40.0)))
    collector.collect_terminal_events(_Sim([_Event({"miss_distance_m": 5.0, "pk": 0.8})]))

    for key in collector.episode_metrics():
        assert key in MISSILE_DIAGNOSTIC_KEYS, key


def test_geometry_is_averaged_over_the_episode_not_just_the_last_shot():
    collector = MissileDiagnosticsCollector()
    collector.record_launch({LAUNCH_RANGE_KM: 20.0})
    collector.record_launch({LAUNCH_RANGE_KM: 60.0})

    assert collector.episode_metrics()[LAUNCH_RANGE_KM] == pytest.approx(40.0)


def test_altitude_delta_is_signed():
    """A shot taken from below is a different failure from one taken from above."""
    from bvr_marl_core.domain.launch_geometry import LAUNCH_ALT_DELTA_M

    high = capture_launch_geometry(_Shooter(alt=9000.0), _target_east(40.0, alt=4000.0))
    low = capture_launch_geometry(_Shooter(alt=4000.0), _target_east(40.0, alt=9000.0))

    assert high[LAUNCH_ALT_DELTA_M] == pytest.approx(-5000.0)
    assert low[LAUNCH_ALT_DELTA_M] == pytest.approx(5000.0)
    assert not math.isnan(high[LAUNCH_ASPECT_DEG])
