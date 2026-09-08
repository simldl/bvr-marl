"""The shooting decision must be judged only when a shot was actually available.

`valid_shot_rate` and `trigger_precision_rate` both divide by trigger PRESSES:

    valid_shot_rate    = launches / (launches + vetoed)      # all presses, incl. spam
    trigger_precision_rate = launches / (launches + wasted)      # presses allowed to try

Neither can separate "never had a shot" from "had shots and declined them" -- an agent
that never presses reads 0.0, and one press in perfect geometry reads 1.0 while ignoring
every other opening. `shot_opportunity_this_step` counts viable solutions independently
of the trigger, giving a denominator of OPPORTUNITIES.
"""

from __future__ import annotations

import pytest

from bvr_marl_core.rl.environment.spaces.action_space.utils.debug_info import (
    DebugInfoCollector,
)


def test_signal_is_initialised_and_reset_each_step():
    collector = DebugInfoCollector()
    state = collector.init_training_signals()

    assert state["shot_opportunity_this_step"] == 0

    state["shot_opportunity_this_step"] = 1
    collector.reset_step_counters(state)

    assert state["shot_opportunity_this_step"] == 0, "must not leak across steps"


def test_signal_is_exposed_to_the_env():
    collector = DebugInfoCollector()
    state = collector.init_training_signals()
    state["shot_opportunity_this_step"] = 1

    assert collector.get_training_signals(state)["shot_opportunity"] == 1


def test_state_tracker_accumulates_opportunities_per_episode():
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker

    tracker = StateTracker()
    for opportunity in (1, 0, 1, 1):
        tracker.update_diagnostic_metrics(
            "agent_0",
            valid_shots=0,
            vetoed_shots=0,
            lock_ok=True,
            fov_ok=True,
            shot_opportunity=opportunity,
        )

    assert tracker.episode_shot_opportunity_count["agent_0"] == 3
    assert tracker.episode_steps_count["agent_0"] == 4


def test_opportunities_are_cleared_between_episodes():
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker

    tracker = StateTracker()
    tracker.update_diagnostic_metrics(
        "agent_0", valid_shots=0, vetoed_shots=0, lock_ok=True, fov_ok=True, shot_opportunity=1
    )
    tracker.reset()

    assert tracker.episode_shot_opportunity_count == {}


# -- the distinction the metric exists to make -----------------------------


def _press_based_rates(launches: int, vetoed: int, wasted: int) -> tuple[float, float]:
    """The two shipped rates, as tactical_metrics computes them."""
    valid_shot_rate = launches / max(launches + vetoed, 1)
    trigger_precision_rate = launches / max(launches + wasted, 1)
    return valid_shot_rate, trigger_precision_rate


def _conversion(launches: int, opportunities: int) -> float | None:
    return min(launches, opportunities) / opportunities if opportunities > 0 else None


def test_press_based_rates_cannot_see_a_declined_opportunity():
    # A policy holding a firing solution for 50 steps that never pulls the trigger.
    valid_rate, possible_rate = _press_based_rates(launches=0, vetoed=0, wasted=0)

    # Both read 0.0 -- indistinguishable from a policy that never had a shot at all.
    assert valid_rate == 0.0
    assert possible_rate == 0.0
    # The opportunity metric says what actually happened: 50 chances, none taken.
    assert _conversion(launches=0, opportunities=50) == 0.0


def test_press_based_rates_flatter_a_single_lucky_press():
    # One press, in perfect geometry, while ignoring 49 other openings.
    _, possible_rate = _press_based_rates(launches=1, vetoed=0, wasted=0)

    assert possible_rate == 1.0, "looks like flawless discipline"
    assert _conversion(launches=1, opportunities=50) == pytest.approx(0.02)


def test_conversion_is_undefined_rather_than_zero_without_an_opportunity():
    # No viable solution ever existed: there is no shooting decision to score. Reporting
    # 0.0 would read as "declined every shot" and drag the campaign average down.
    assert _conversion(launches=0, opportunities=0) is None


def test_conversion_is_bounded_to_one():
    assert _conversion(launches=5, opportunities=3) == 1.0


# --- P(fire | can_fire) -----------------------------------------------------------
#
# `shot_opportunities` alone gives the denominator. Without a matching numerator the
# shooting DECISION is still unreadable: the raw fire rate is ~2% of steps, but the
# infeasible-step pin makes the trigger expressible only on can_fire steps, so 2% is
# equally consistent with "fire head collapsed" and "fires whenever allowed". Measured
# in one run, missiles_fired / shot_opportunities was 0.71%, and nothing in the metric set
# could say which of those two it meant.


def test_attempt_signal_is_initialised_and_reset_each_step():
    collector = DebugInfoCollector()
    state = collector.init_training_signals()
    assert state["fire_attempt_on_opportunity_this_step"] == 0

    state["fire_attempt_on_opportunity_this_step"] = 1
    collector.reset_step_counters(state)
    assert state["fire_attempt_on_opportunity_this_step"] == 0, "must not leak across steps"


def test_attempt_signal_is_exposed_to_the_env():
    collector = DebugInfoCollector()
    state = collector.init_training_signals()
    state["fire_attempt_on_opportunity_this_step"] = 1
    assert collector.get_training_signals(state)["fire_attempt_on_opportunity"] == 1


def test_state_tracker_accumulates_attempts_per_episode():
    from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker

    tracker = StateTracker()
    for attempted in (1, 0, 1, 1):
        tracker.update_diagnostic_metrics(
            "agent_0",
            valid_shots=0,
            vetoed_shots=0,
            lock_ok=True,
            fov_ok=True,
            shot_opportunity=1,
            fire_attempt_on_opportunity=attempted,
        )

    assert tracker.episode_shot_opportunity_count["agent_0"] == 4
    assert tracker.episode_fire_attempt_on_opportunity_count["agent_0"] == 3


@pytest.mark.parametrize(
    ("opportunities", "attempts", "expected"),
    [
        (200, 100, 0.5),
        (200, 0, 0.0),  # had 200 shots and declined every one -- a COLLAPSED fire head
        (200, 200, 1.0),  # fires whenever allowed -- the opposite reading
        (0, 0, None),  # never had a shot: NOT the same statement as declining them
    ],
)
def test_conditional_fire_rate_semantics(opportunities, attempts, expected):
    """0.0 and None must stay distinguishable.

    Collapsing "no opportunities" to 0.0 would make a policy that never got a firing
    solution indistinguishable from one that got hundreds and refused them all -- which
    is the entire distinction this metric exists to draw.
    """
    rate = attempts / opportunities if opportunities > 0 else None
    assert rate == expected
