from types import SimpleNamespace

import numpy as np
import pytest

from bvr_marl_core.rl.environment.gym.gym_components.helpers import AgentHelpers
from bvr_marl_core.rl.environment.rewards import RewardCalculator
from bvr_marl_core.rl.environment.rewards.information import (
    RewardInformationClass,
    ensure_reward_information_allowed,
)
from bvr_marl_core.simulator.core.helpers import Position
from tests.helpers.track_snapshot import track_snapshot


def test_reward_defaults_to_observation_only_with_separate_terminal_terms():
    calculator = RewardCalculator()

    assert calculator.reward_information_mode is RewardInformationClass.OBSERVATION_ONLY
    assert {item["information_class"] for item in calculator.metadata()["reward_components"]} == {
        "evaluator_terminal_only"
    }
    reward, *_ = calculator.compute_total_reward(None, None, enemy_kill_count=1)
    assert reward == 1.0


def test_forbidden_reward_information_fails_immediately():
    with pytest.raises(ValueError, match="forbidden"):
        ensure_reward_information_allowed(
            RewardInformationClass.OBSERVATION_ONLY,
            "truth_geometry_shaping",
            RewardInformationClass.PRIVILEGED_TRAINING,
        )


def test_reward_information_mode_is_validated():
    with pytest.raises(ValueError, match="reward_information_mode"):
        RewardCalculator(reward_information_mode="sometimes_truth")


def _track(track_id, classification):
    return track_snapshot(
        track_id,
        state=(10_000.0, 20_000.0, 1_000.0, 100.0, 200.0, 10.0),
        classification=classification,
        confidence=0.8,
    )


def test_observation_only_reward_context_uses_estimated_contacts():
    sensor = SimpleNamespace(
        sensor_tracks=[_track("fighter-track", "fighter"), _track("missile-track", "missile")],
        get_locked_targets=lambda: {"fighter-track"},
    )
    ownship = SimpleNamespace(id=1, position=Position(50.0, 8.0, 5_000.0), sensor=sensor)
    simulator = SimpleNamespace(active_units={1: ownship})
    helpers = AgentHelpers(simulator, ["A0"], ["B0"], {"A0": 1})

    enemies = helpers.get_estimated_enemies_for_agent("A0")
    targets = helpers.get_estimated_targets_for_agent("A0")
    missiles = helpers.get_estimated_incoming_missiles_for_agent("A0")

    assert [contact.id for contact in enemies] == ["fighter-track"]
    assert [contact.id for contact in targets] == ["fighter-track"]
    assert [contact.id for contact in missiles] == ["missile-track"]
    assert enemies[0].information_class is RewardInformationClass.OBSERVATION_ONLY
    assert enemies[0].operational_contact.state[:3] == (10_000.0, 20_000.0, 1_000.0)
    assert enemies[0].speed == pytest.approx(np.linalg.norm([100.0, 200.0, 10.0]))
    assert not hasattr(enemies[0], "remaining_missiles")


def test_estimated_reward_context_is_limited_to_observation_capacity():
    tracks = [_track(f"fighter-{index}", "fighter") for index in range(5)]
    tracks += [_track(f"missile-{index}", "missile") for index in range(5)]
    sensor = SimpleNamespace(sensor_tracks=tracks, get_locked_targets=lambda: set())
    ownship = SimpleNamespace(id=1, position=Position(50.0, 8.0, 5_000.0), sensor=sensor)
    helpers = AgentHelpers(SimpleNamespace(active_units={1: ownship}), ["A0"], [], {"A0": 1})

    enemies, targets, missiles = helpers.get_estimated_reward_context(
        "A0", fighter_limit=2, missile_limit=3
    )

    assert len(enemies) == 2
    assert targets == []
    assert len(missiles) == 3
