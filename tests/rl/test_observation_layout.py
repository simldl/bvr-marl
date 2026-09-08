"""The observation layout must be derivable from config, and must match reality.

The observation width is slot counts times per-token widths, and the slot counts come
from ``config.get(..., <default>)`` -- values that live only in code. So a checkpoint and
a config can disagree with nothing recording it, and the disagreement surfaces as::

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x175 and 197x512)

That is what a checkpoint trained under a different slot count does. Identifying it meant
reconstructing the arithmetic by hand: ``num_fm`` had been 6 against a current default of
4, and 2 extra friendly-missile slots x d_FM 11 is exactly the missing 22. Nothing in the
saved run config recorded ``num_fm`` at all.

``observation_layout`` mirrors the resolution order in ``gym_components/config.py``. That
duplication is the risk it introduces, so the last test here pins the mirror against a
real environment: if the two ever drift, this fails rather than a checkpoint silently
becoming unloadable.
"""

from __future__ import annotations

import pytest

from bvr_marl_core.rl.environment.spaces.observation.constants import d_EF, d_FM, d_OWN
from bvr_marl_core.rl.environment.spaces.observation.layout import (
    describe_layout_mismatch,
    observation_layout,
    observation_width,
)


def test_defaults_resolve_to_the_measured_width():
    """175 is what a stage-1 config actually produces; pin it."""
    layout = observation_layout({"emcon_action_enabled": True})

    assert layout["total"] == 175
    assert layout["block_friendly_missiles"] == 4 * d_FM
    assert layout["block_enemy_fighters"] == 2 * d_EF


def test_num_fm_is_the_knob_that_explains_the_v18_checkpoint():
    """The incident, encoded: 4 -> 6 is exactly the 22 features that were missing."""
    base = {"emcon_action_enabled": True}

    assert observation_width(base) == 175
    assert observation_width({**base, "num_fm": 6}) == 197
    assert observation_width({**base, "num_fm": 6}) - observation_width(base) == 2 * d_FM


def test_emcon_widens_only_the_ownship_block():
    on = observation_layout({"emcon_action_enabled": True})
    off = observation_layout({"emcon_action_enabled": False})

    assert off["own_state_dim"] == d_OWN
    assert on["total"] - off["total"] == on["own_state_dim"] - off["own_state_dim"]


def test_opponent_awacs_silently_widens_the_enemy_block():
    """The trap: a scenario flag that invalidates every earlier checkpoint.

    Toggling this is not obviously an observation-space change, which is exactly why it
    needs to be visible in a recorded layout.
    """
    without = observation_layout({"scenario_config": {"awacs_config": {"opponent_awacs": False}}})
    with_awacs = observation_layout(
        {
            "scenario_config": {
                "awacs_config": {"opponent_awacs": True, "awacs_non_engageable": True}
            }
        }
    )

    assert with_awacs["num_ef"] == without["num_ef"] + 1
    assert with_awacs["total"] - without["total"] == d_EF


def test_an_engageable_opponent_awacs_does_not_bump_the_slot():
    """The bump exists to reserve a slot for a NON-engageable AWACS specifically."""
    layout = observation_layout(
        {
            "scenario_config": {
                "awacs_config": {"opponent_awacs": True, "awacs_non_engageable": False}
            }
        }
    )

    assert layout["num_ef"] == 2


def test_passive_radar_slots_follow_the_enemy_count():
    """Second side effect: num_pr defaults to num_enemy."""
    layout = observation_layout({"num_enemy": 3})

    assert layout["num_pr"] == 3
    assert layout["num_ef"] == 3


def test_mismatch_names_the_responsible_knob():
    """'expected num_fm=6, got 4' is actionable; a matrix shape error is not."""
    expected = observation_layout({"emcon_action_enabled": True, "num_fm": 6})
    actual = observation_layout({"emcon_action_enabled": True})

    message = describe_layout_mismatch(expected, actual)

    assert "num_fm: expected 6, got 4" in message
    assert "total: expected 197, got 175" in message


def test_identical_layouts_report_no_mismatch():
    layout = observation_layout({"emcon_action_enabled": True})

    assert describe_layout_mismatch(layout, layout) == ""


def test_the_layout_matches_a_real_environment():
    """Guards the duplication: layout.py mirrors gym_components/config.py by hand."""
    import numpy as np

    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from bvr_marl_core.simulator.simulator import Simulator

    config = {
        "simulator": Simulator(tick_secs=0.5),
        "num_agents_per_team": 1,
        "map_size": 400.0,
        "max_steps": 8,
        "emcon_action_enabled": True,
    }
    env = BVRMultiAgentEnv(dict(config))

    # observation_space is a plain dict of per-agent Dict spaces.
    agent = next(iter(env.observation_space))
    measured = sum(int(np.prod(s.shape)) for s in env.observation_space[agent].spaces.values())

    assert env.observation_layout["total"] == measured, describe_layout_mismatch(
        env.observation_layout, {"total": measured}
    )
