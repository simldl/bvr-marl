"""EMCON ownship observation + emission tracker."""

from bvr_marl_core.aircraft.systems.emission_tracker import EmissionTracker
from bvr_marl_core.rl.environment.spaces.observation.constants import (
    d_OWN,
    d_OWN_EMCON,
    own_state_dim,
)


def test_own_state_dim_helper():
    assert own_state_dim(False) == d_OWN == 16
    assert own_state_dim(True) == d_OWN_EMCON == 19


def test_emission_tracker_duty_cycle_and_toggles():
    t = EmissionTracker()
    assert t.duty_cycle == 1.0  # before any step
    for emitting in [True, True, False, False, True]:
        t.record(emitting)
    assert t.steps == 5
    assert t.emitting_steps == 3
    assert t.duty_cycle == 3 / 5
    assert t.transitions == 2  # T->F and F->T
    # last two records: F,T -> toggle at step 5 -> steps_since_toggle resets to 0.
    assert t.steps_since_toggle == 0
    summary = t.summary()
    assert summary["emission_duty_cycle"] == 3 / 5
    assert summary["emission_transitions"] == 2.0


def test_emission_tracker_all_on_is_full_duty():
    t = EmissionTracker()
    for _ in range(10):
        t.record(True)
    assert t.duty_cycle == 1.0
    assert t.transitions == 0
    assert t.steps_since_toggle == 10


def test_emission_tracker_reset():
    t = EmissionTracker()
    t.record(False)
    t.reset()
    assert t.steps == 0 and t.duty_cycle == 1.0
