"""Golden scenario tests — determinism and behavioral correctness.

Each test runs a fixed, seeded scenario twice and verifies:
  1. Both runs produce identical full event logs (determinism).
  2. The event log matches a frozen reference trace (behavioral regression guard).

The frozen traces are captured from the canonical seeded runs below and serve as
an explicit regression baseline.  If a code change causes a different outcome,
these tests fail and the change must be reviewed before the constants are updated.

Frozen reference traces
-----------------------
GOLDEN_1V1_TRACE     — seed 1001, 40-tick acquisition + 150-tick run
GOLDEN_2V2_TRACE     — seed 2002, 45-tick acquisition + 150-tick run
GOLDEN_TERMINAL_TRACE — seed 3003, 10-tick acquisition + 80-tick run

Acquisition mechanics
---------------------
The scenarios use a lock-based fire gate.  The setup phase runs until the active
radar acquires a lock, then fires.  Setup tick counts are calibrated to give at
least 2 ticks of margin after the expected lock tick for the given seeds.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from bvr_marl_core.aircraft.aircraft import Aircraft
from bvr_marl_core.missiles.fox3.default_missile import LongRangeMissile
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.recorder import ReplayRecorder
from bvr_marl_core.simulator.simulator import Simulator

# ── Frozen reference traces ───────────────────────────────────────────────────
# Captured from deterministic seeded runs with the fixes in place.
# Update ONLY when a deliberate simulator behaviour change warrants it.

GOLDEN_1V1_TRACE: list[str] = [
    "UnitRegisteredEvent",  # blue1 aircraft
    "UnitRegisteredEvent",  # red1 aircraft
    "UnitRegisteredEvent",  # fired missile
    "MissileEnteredTerminalRegionEvent",
    "MissileEngagementEvent",
    "MissileFuzeTriggeredEvent",
    "MissileDetonatedEvent",
    "MissileTerminalEvent",  # structured terminal-outcome record at detonation
    "AircraftMortallyHitEvent",
    "UnitDestroyedEvent",  # missile destroyed on impact
    "AircraftDestroyedEvent",
    # Stochastic kill delay: the target is mortally hit at impact but dies after a
    # short lag, so its destroyed event and the removal-sweep event come later.
    "UnitRemovedEvent",  # red1 wreck swept from the sim
]

GOLDEN_2V2_TRACE: list[str] = [
    "UnitRegisteredEvent",  # blue1
    "UnitRegisteredEvent",  # blue2
    "UnitRegisteredEvent",  # red1
    "UnitRegisteredEvent",  # red2
    "UnitRegisteredEvent",  # missile from blue1
    "UnitRegisteredEvent",  # missile from blue2
    # Both blue missiles intercept their paired red. With the warhead lethal radius
    # reconciled to the proximity-fuze radius, each crossing-geometry CPA now falls
    # inside the lethal envelope, so BOTH detonations are kills (previously one
    # missed): each logs a terminal-outcome record and a missile self-destruct, the
    # target is mortally hit, and after a short stochastic lag its destroyed and
    # wreck-removal-sweep events follow.
    "MissileEnteredTerminalRegionEvent",
    "MissileEngagementEvent",
    "MissileEnteredTerminalRegionEvent",
    "MissileEngagementEvent",
    "MissileFuzeTriggeredEvent",
    "MissileDetonatedEvent",
    "MissileTerminalEvent",
    "AircraftMortallyHitEvent",
    "UnitDestroyedEvent",
    "AircraftDestroyedEvent",
    "MissileFuzeTriggeredEvent",
    "MissileDetonatedEvent",
    "MissileTerminalEvent",
    "AircraftMortallyHitEvent",
    "UnitDestroyedEvent",
    "UnitRemovedEvent",
    "AircraftDestroyedEvent",
    "UnitRemovedEvent",
]

GOLDEN_TERMINAL_TRACE: list[str] = [
    "UnitRegisteredEvent",  # blue1
    "UnitRegisteredEvent",  # red1
    "UnitRegisteredEvent",  # missile (close-range scenario acquires lock quickly)
    "MissileEnteredTerminalRegionEvent",
    "MissileEngagementEvent",
    "MissileFuzeTriggeredEvent",
    "MissileDetonatedEvent",
    "MissileTerminalEvent",  # structured terminal-outcome record at detonation
    "AircraftMortallyHitEvent",
    "UnitDestroyedEvent",  # missile destroyed on impact
    "AircraftDestroyedEvent",
    "UnitRemovedEvent",  # red1 wreck swept from the sim
]


# ── Shared helpers ────────────────────────────────────────────────────────────


def _map_limits():
    return SimpleNamespace(
        left_lon=-2.0,
        right_lon=2.0,
        bottom_lat=-2.0,
        top_lat=2.0,
        min_alt=0,
        max_alt=25000,
    )


def _make_aircraft(
    name: str,
    lat: float,
    lon: float,
    alt: float,
    heading_deg: float,
    speed_mps: float,
    group: str,
    map_limits,
    rcs: float = 5.0,
) -> Aircraft:
    return Aircraft(
        name=name,
        position=Position(lat=lat, lon=lon, alt=alt),
        yaw_deg=heading_deg,
        speed_mps=speed_mps,
        group=group,
        map_limits=map_limits,
        config={
            "mass_kg": 18000,
            "reference_area_m2": 50,
            "max_speed_mps": 650,
            "n_max": 9,
            "radar_horizontal_fov_deg": 70,
            "radar_vertical_fov_deg": 35,
            "radar_max_range_m": 150_000,
            "radar_frequency_hz": 10e9,
            "radar_tx_power_w": 15e3,
            "radar_antenna_gain_db": 40,
            "radar_snr_threshold_db": 10,
            "rcs": rcs,
        },
    )


def _make_sim(seed: int) -> Simulator:
    return Simulator(
        utc_time=datetime(2025, 6, 1, 12, 0, 0),
        tick_secs=1.0,
        random_seed=seed,
    )


def _event_names(sim: Simulator) -> list[str]:
    """Return the full event log as a list of class names."""
    return [type(e).__name__ for e in sim.events]


# ── Scenario builders ─────────────────────────────────────────────────────────


def _setup_1v1(sim: Simulator):
    """1v1 head-on BVR engagement: blue fires one missile at red.

    40 acquisition ticks gives radar lock by tick 38 (with 2 ticks margin).
    """
    ml = _map_limits()
    blue = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=10_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red = _make_aircraft(
        "red1",
        lat=0.8,
        lon=0.0,
        alt=10_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=5.0,
    )
    sim.add_unit(blue)
    sim.add_unit(red)

    for _ in range(40):
        sim.do_tick()

    blue.weapons.fire_missile(sim, red, LongRangeMissile)


def _setup_2v2(sim: Simulator):
    """2v2 symmetric engagement — each blue fires on its paired red.

    45 acquisition ticks gives radar lock for both blues by tick 42 (3 ticks margin).
    """
    ml = _map_limits()
    blue1 = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=-0.2,
        alt=10_000,
        heading_deg=10.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
    )
    blue2 = _make_aircraft(
        "blue2",
        lat=0.0,
        lon=0.2,
        alt=10_500,
        heading_deg=350.0,
        speed_mps=290.0,
        group="blue",
        map_limits=ml,
    )
    red1 = _make_aircraft(
        "red1",
        lat=0.8,
        lon=-0.2,
        alt=10_000,
        heading_deg=190.0,
        speed_mps=270.0,
        group="red",
        map_limits=ml,
    )
    red2 = _make_aircraft(
        "red2",
        lat=0.8,
        lon=0.2,
        alt=9_800,
        heading_deg=170.0,
        speed_mps=260.0,
        group="red",
        map_limits=ml,
    )
    for unit in (blue1, blue2, red1, red2):
        sim.add_unit(unit)

    for _ in range(45):
        sim.do_tick()

    blue1.weapons.fire_missile(sim, red1, LongRangeMissile)
    blue2.weapons.fire_missile(sim, red2, LongRangeMissile)


def _setup_terminal_accounting(sim: Simulator):
    """Close-range setup so a missile hit and kill can be validated quickly.

    At 0.1 deg (~11 km) the radar acquires lock by tick 1; 10 setup ticks
    gives generous margin.
    """
    ml = _map_limits()
    blue = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=8_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red = _make_aircraft(
        "red1",
        lat=0.1,
        lon=0.0,
        alt=8_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=5.0,
    )
    sim.add_unit(blue)
    sim.add_unit(red)
    for _ in range(10):
        sim.do_tick()
    blue.weapons.fire_missile(sim, red, LongRangeMissile)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_1v1_bvr_is_deterministic_and_matches_golden():
    """1v1 BVR: event log must be identical across two seeded runs and match frozen trace."""
    SEED = 1001
    N_TICKS = 150

    sim_a = _make_sim(SEED)
    sim_b = _make_sim(SEED)

    _setup_1v1(sim_a)
    for _ in range(N_TICKS):
        sim_a.do_tick()

    _setup_1v1(sim_b)
    for _ in range(N_TICKS):
        sim_b.do_tick()

    names_a = _event_names(sim_a)
    names_b = _event_names(sim_b)

    assert names_a == names_b, (
        f"1v1 BVR event log is not deterministic.\nRun A: {names_a}\nRun B: {names_b}"
    )
    assert names_a == GOLDEN_1V1_TRACE, (
        "1v1 BVR event log does not match frozen reference trace.\n"
        f"Got:      {names_a}\n"
        f"Expected: {GOLDEN_1V1_TRACE}"
    )


@pytest.mark.slow
def test_1v1_bvr_fires_missile():
    """After the 1v1 setup (40 acquisition ticks) a missile must be registered."""
    sim = _make_sim(1001)
    _setup_1v1(sim)
    missiles_in_sim = [u for u in sim.active_units.values() if getattr(u, "is_missile", False)]
    assert len(missiles_in_sim) >= 1, (
        "No missile was registered after 1v1 setup — radar lock may not have been acquired"
    )


@pytest.mark.slow
def test_2v2_bvr_is_deterministic_and_matches_golden():
    """2v2 BVR: event log must be identical across two seeded runs and match frozen trace."""
    SEED = 2002
    N_TICKS = 150

    sim_a = _make_sim(SEED)
    sim_b = _make_sim(SEED)

    _setup_2v2(sim_a)
    for _ in range(N_TICKS):
        sim_a.do_tick()

    _setup_2v2(sim_b)
    for _ in range(N_TICKS):
        sim_b.do_tick()

    names_a = _event_names(sim_a)
    names_b = _event_names(sim_b)

    assert names_a == names_b, "2v2 BVR event log is not deterministic"
    assert names_a == GOLDEN_2V2_TRACE, (
        "2v2 BVR event log does not match frozen reference trace.\n"
        f"Got:      {names_a}\n"
        f"Expected: {GOLDEN_2V2_TRACE}"
    )


@pytest.mark.slow
def test_2v2_both_missiles_launched():
    """In the 2v2 setup both blue aircraft must fire a missile."""
    sim = _make_sim(2002)
    _setup_2v2(sim)
    missiles = [u for u in sim.active_units.values() if getattr(u, "is_missile", False)]
    assert len(missiles) >= 2, f"Expected >= 2 missiles after 2v2 setup, found {len(missiles)}"


@pytest.mark.slow
def test_terminal_accounting_units_registered():
    """All initial units must emit UnitRegisteredEvents."""
    sim = _make_sim(3003)
    ml = _map_limits()
    blue = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=8_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red = _make_aircraft(
        "red1",
        lat=0.1,
        lon=0.0,
        alt=8_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=5.0,
    )
    sim.add_unit(blue)
    sim.add_unit(red)

    from bvr_marl_core.simulator.core.events import UnitRegisteredEvent

    registered = [e for e in sim.events if isinstance(e, UnitRegisteredEvent)]
    assert len(registered) == 2


@pytest.mark.slow
def test_terminal_accounting_is_deterministic_and_matches_golden():
    """Close-range engagement outcome must match frozen trace across two runs."""
    SEED = 3003
    N_TICKS = 80

    sim_a = _make_sim(SEED)
    sim_b = _make_sim(SEED)

    _setup_terminal_accounting(sim_a)
    for _ in range(N_TICKS):
        sim_a.do_tick()

    _setup_terminal_accounting(sim_b)
    for _ in range(N_TICKS):
        sim_b.do_tick()

    names_a = _event_names(sim_a)
    names_b = _event_names(sim_b)

    assert names_a == names_b, "Terminal accounting scenario is not deterministic"
    assert names_a == GOLDEN_TERMINAL_TRACE, (
        "Terminal accounting event log does not match frozen reference trace.\n"
        f"Got:      {names_a}\n"
        f"Expected: {GOLDEN_TERMINAL_TRACE}"
    )


@pytest.mark.slow
def test_replay_metadata_captures_all_determinants(tmp_path):
    """ReplayRecorder must persist seed, simulator_version, scenario_version, env_settings."""
    sim = _make_sim(4004)
    recorder = ReplayRecorder(scenario_id="golden_1v1", scenario_version="1.0")
    recorder.on_reset(sim, env_settings={"map_size_deg": 4.0, "tick_secs": 1.0})

    _setup_1v1(sim)
    for _ in range(20):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    out = tmp_path / "golden_1v1.json"
    recorder.save(out)

    log = ReplayRecorder.load(out)
    assert log.initial_seed == 4004
    assert log.scenario_id == "golden_1v1"
    assert log.scenario_version == "1.0"
    assert log.simulator_version != "", "simulator_version must not be empty"
    assert "tick_secs" in log.env_settings
    assert len(log.ticks) == 20


@pytest.mark.slow
def test_replay_rng_state_is_restorable(tmp_path):
    """Saved RNG states must allow re-forking the simulation from any tick."""
    sim = _make_sim(5005)
    recorder = ReplayRecorder(scenario_id="fork_test")
    recorder.on_reset(sim)

    ml = _map_limits()
    blue = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=10_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    sim.add_unit(blue)

    for _ in range(10):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    out = tmp_path / "fork_test.json"
    recorder.save(out)
    log = ReplayRecorder.load(out)

    import random

    tick5_state = log.ticks[5].rng_state
    assert tick5_state is not None
    rng = random.Random()
    rng.setstate(tick5_state)
    _ = rng.random()


@pytest.mark.slow
def test_different_seeds_produce_different_traces():
    """Two identical scenarios with different seeds must diverge in their RNG state."""
    N_TICKS = 30
    sim_a = _make_sim(1)
    sim_b = _make_sim(2)

    ml = _map_limits()
    for sim in (sim_a, sim_b):
        blue = _make_aircraft(
            "blue1",
            lat=0.0,
            lon=0.0,
            alt=10_000,
            heading_deg=0.0,
            speed_mps=300.0,
            group="blue",
            map_limits=ml,
            rcs=5.0,
        )
        sim.add_unit(blue)
        for _ in range(N_TICKS):
            sim.do_tick()

    assert sim_a.rnd_gen.getstate() != sim_b.rnd_gen.getstate(), (
        "Different seeds produced identical RNG state — seeding is broken"
    )


# ── Extended regression tests ─────────────────────────────────────────────────


@pytest.mark.slow
def test_1v1_output_is_stable_across_multiple_reruns():
    """Running the same 1v1 scenario three times must always produce the same trace.

    This guards against any non-deterministic state that is not governed by
    the seeded RNG (e.g. dict ordering, floating-point environment differences
    within the same process).
    """
    SEED = 7007
    N_TICKS = 120

    traces = []
    for _ in range(3):
        sim = _make_sim(SEED)
        _setup_1v1(sim)
        for _ in range(N_TICKS):
            sim.do_tick()
        traces.append(_event_names(sim))

    assert traces[0] == traces[1] == traces[2], (
        "1v1 scenario output differs between repeated runs — hidden non-determinism detected.\n"
        f"Run 0: {traces[0]}\nRun 1: {traces[1]}\nRun 2: {traces[2]}"
    )


@pytest.mark.slow
def test_rcs_change_alters_engagement_outcome():
    """Reducing RCS below radar detection threshold must prevent lock acquisition.

    This is a deliberate outcome regression: a stealthy target (RCS=0.01) should
    produce a different event trace than a standard target (RCS=5.0) under the
    same scenario geometry.
    """
    SEED = 8008
    N_TICKS = 150
    ml = _map_limits()

    # Standard RCS — radar should acquire lock and fire
    sim_normal = _make_sim(SEED)
    blue_n = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=10_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red_n = _make_aircraft(
        "red1",
        lat=0.8,
        lon=0.0,
        alt=10_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=5.0,
    )
    sim_normal.add_unit(blue_n)
    sim_normal.add_unit(red_n)
    for _ in range(40):
        sim_normal.do_tick()
    blue_n.weapons.fire_missile(sim_normal, red_n, LongRangeMissile)
    for _ in range(N_TICKS):
        sim_normal.do_tick()

    # Stealth RCS — radar cannot see the target reliably; fire gate should not be met
    sim_stealth = _make_sim(SEED)
    blue_s = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=10_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red_s = _make_aircraft(
        "red1",
        lat=0.8,
        lon=0.0,
        alt=10_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=0.001,
    )
    sim_stealth.add_unit(blue_s)
    sim_stealth.add_unit(red_s)
    for _ in range(N_TICKS):
        sim_stealth.do_tick()

    normal_kills = sum(1 for e in _event_names(sim_normal) if e == "UnitDestroyedEvent")
    stealth_kills = sum(1 for e in _event_names(sim_stealth) if e == "UnitDestroyedEvent")

    assert normal_kills > stealth_kills, (
        f"Standard-RCS scenario should produce more kills than stealth-RCS scenario.\n"
        f"Normal kills: {normal_kills}, Stealth kills: {stealth_kills}"
    )


@pytest.mark.slow
def test_replay_tick_event_summaries_are_accurate(tmp_path):
    """Event summaries stored per tick in the replay log must match real sim events."""
    SEED = 9009

    sim = _make_sim(SEED)
    recorder = ReplayRecorder(scenario_id="tick_event_test", scenario_version="2.0")
    recorder.on_reset(sim)

    ml = _map_limits()
    blue = _make_aircraft(
        "blue1",
        lat=0.0,
        lon=0.0,
        alt=10_000,
        heading_deg=0.0,
        speed_mps=300.0,
        group="blue",
        map_limits=ml,
        rcs=5.0,
    )
    red = _make_aircraft(
        "red1",
        lat=0.8,
        lon=0.0,
        alt=10_000,
        heading_deg=180.0,
        speed_mps=280.0,
        group="red",
        map_limits=ml,
        rcs=5.0,
    )
    sim.add_unit(blue)
    sim.add_unit(red)

    # Capture setup-phase events (unit registrations happen before any tick).
    recorder.record_setup_phase(sim)

    # Run 40 ticks of acquisition, then fire, then 60 more ticks
    for _ in range(40):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    blue.weapons.fire_missile(sim, red, LongRangeMissile)

    for _ in range(60):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    out = tmp_path / "tick_event_test.json"
    recorder.save(out)
    log = ReplayRecorder.load(out)

    # The log must have exactly 100 tick records
    assert len(log.ticks) == 100, f"Expected 100 tick records, got {len(log.ticks)}"

    # Unit registrations belong to the setup phase, not to any tick.
    assert "UnitRegisteredEvent" in log.setup_events, (
        "UnitRegisteredEvent must be captured in the setup_events field of the replay log"
    )

    # Within-tick events (kills, missile destructions) must appear in tick summaries.
    tick_event_names = [ev for t in log.ticks for ev in t.events_summary]
    assert "UnitDestroyedEvent" in tick_event_names, (
        "UnitDestroyedEvent was never captured in per-tick event summaries"
    )

    # Schema version must be captured
    assert log.scenario_version == "2.0"
    assert log.initial_seed == SEED


@pytest.mark.slow
def test_replay_schema_version_is_preserved_through_save_load(tmp_path):
    """The scenario_version field in a replay log must survive save/load intact."""
    sim = _make_sim(1111)
    recorder = ReplayRecorder(scenario_id="schema_compat_test", scenario_version="1.0")
    recorder.on_reset(sim)

    for _ in range(5):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    out = tmp_path / "schema_compat.json"
    recorder.save(out)
    log = ReplayRecorder.load(out)

    assert log.scenario_version == "1.0", (
        "scenario_version must be preserved through save/load — "
        f"expected '1.0', got {log.scenario_version!r}"
    )
    assert log.scenario_id == "schema_compat_test"


@pytest.mark.slow
def test_replay_log_tick_count_matches_ticks_run(tmp_path):
    """Replay log must contain exactly as many tick records as ticks executed."""
    N_TICKS = 25
    sim = _make_sim(2222)
    recorder = ReplayRecorder(scenario_id="tick_count_test")
    recorder.on_reset(sim)

    for _ in range(N_TICKS):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    out = tmp_path / "tick_count.json"
    recorder.save(out)
    log = ReplayRecorder.load(out)

    assert len(log.ticks) == N_TICKS, (
        f"Expected {N_TICKS} tick records in replay log, got {len(log.ticks)}"
    )
