from __future__ import annotations

from dataclasses import dataclass

from bvr_marl_core.rl.environment.gym.gym_components.state_tracker import StateTracker
from bvr_marl_core.rl.environment.gym.gym_components.step_processor import StepProcessor
from bvr_marl_core.simulator import UnitDestroyedEvent


@dataclass
class _Unit:
    id: int
    source: object | None = None


def _processor() -> StepProcessor:
    return StepProcessor(
        simulator=object(),
        obs_builder=object(),
        action_processor=object(),
        reward_calculator=object(),
        config=object(),
    )


def test_duplicate_destroyed_events_credit_only_one_kill() -> None:
    tracker = StateTracker()
    tracker.initialize_agent("A0")
    tracker.initialize_agent("B0")
    agent_to_unit_id = {"A0": 1, "B0": 2}
    event = UnitDestroyedEvent("sim", _Unit(1), _Unit(2))
    kills_this_step: dict[str, int] = {}

    proc = _processor()
    proc._process_kill_event(event, agent_to_unit_id, tracker, kills_this_step)
    proc._process_kill_event(event, agent_to_unit_id, tracker, kills_this_step)

    assert tracker.episode_kills["A0"] == 1
    assert kills_this_step["A0"] == 1


def test_boundary_destroyed_event_does_not_credit_kill() -> None:
    tracker = StateTracker()
    tracker.initialize_agent("A0")
    tracker.initialize_agent("B0")
    tracker.record_boundary_violation("B0")
    agent_to_unit_id = {"A0": 1, "B0": 2}
    event = UnitDestroyedEvent("sim", _Unit(1), _Unit(2))
    kills_this_step: dict[str, int] = {}

    _processor()._process_kill_event(event, agent_to_unit_id, tracker, kills_this_step)

    assert tracker.episode_kills["A0"] == 0
    assert kills_this_step == {}
