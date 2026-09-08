"""EMCON: the radar on/off action (index 9) and its effect on emissions."""

import numpy as np

from bvr_marl_core.rl.environment.spaces.action_space import ActionProcessor, ActionSpaceManager
from tests.rl.environment.spaces.test_action_space import DummySim


def test_action_space_has_radar_toggle_dim():
    mgr = ActionSpaceManager(agent_ids=["A1"], shape=10)
    assert mgr.get("A1").shape == (10,)


def _apply(radar_channel):
    sim = DummySim()
    proc = ActionProcessor(sim)
    # 9 base action entries + the EMCON radar toggle at index 9.
    action = np.array(
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, radar_channel], dtype=np.float32
    )
    proc.apply("A1", action)
    return sim.active_units["A1"]


def test_radar_stays_on_at_neutral_fills():
    for neutral in (0.0, 0.5, 0.7):
        assert _apply(neutral).radar_emitting is True, neutral


def test_radar_goes_silent_when_channel_high():
    assert _apply(0.9).radar_emitting is False


def test_base_dim_action_leaves_radar_unset():
    sim = DummySim()
    proc = ActionProcessor(sim)
    action = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    proc.apply("A1", action)  # no index 9 -> processor must not touch radar state
    assert not hasattr(sim.active_units["A1"], "radar_emitting") or (
        sim.active_units["A1"].radar_emitting is True
    )
