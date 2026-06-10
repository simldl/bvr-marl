"""Golden-scenario test: verify that re-running from the same seed produces identical events.

This is a minimal smoke test. Full golden scenarios with frozen reference traces
should be added alongside each release.
"""

import random
from datetime import datetime

import pytest

from bvr_marl_core.simulator.core.recorder import ReplayRecorder
from bvr_marl_core.simulator.simulator import Simulator


def _make_simulator(seed: int) -> Simulator:
    return Simulator(
        utc_time=datetime(2025, 1, 1, 0, 0, 0),
        tick_secs=1,
        random_seed=seed,
    )


def _run_empty_sim(sim: Simulator, n_ticks: int) -> list[str]:
    """Run n_ticks on an empty simulator; return event-name sequence."""
    event_names = []
    for _ in range(n_ticks):
        events = sim.do_tick()
        event_names.extend(type(e).__name__ for e in events)
    return event_names


def test_empty_sim_is_deterministic():
    """Two simulators with the same seed must emit identical event sequences."""
    seed = 42
    sim_a = _make_simulator(seed)
    sim_b = _make_simulator(seed)
    assert _run_empty_sim(sim_a, 10) == _run_empty_sim(sim_b, 10)


def test_replay_recorder_captures_ticks(tmp_path):
    """ReplayRecorder must capture a tick record for every do_tick() call."""
    sim = _make_simulator(99)
    recorder = ReplayRecorder(scenario_id="smoke")
    recorder.on_reset(sim)
    for _ in range(5):
        recorder.before_tick(sim)
        events = sim.do_tick()
        recorder.after_tick(sim, events)

    assert len(recorder._log.ticks) == 5

    out = tmp_path / "replay.json"
    recorder.save(out)
    assert out.exists()

    loaded = ReplayRecorder.load(out)
    assert len(loaded.ticks) == 5
    assert loaded.initial_seed == 99


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different RNG state after ticks."""
    sim_a = _make_simulator(1)
    sim_b = _make_simulator(2)
    # After 5 ticks the internal rnd state should differ
    for _ in range(5):
        sim_a.do_tick()
        sim_b.do_tick()
    assert sim_a.rnd_gen.getstate() != sim_b.rnd_gen.getstate()
