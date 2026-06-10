"""Tests for kill reward calculation in the minimal reward calculator."""

from bvr_marl_core.rl.environment.rewards import RewardCalculator


def test_kill_reward_calculation():
    """Test that kill rewards are correctly calculated."""
    calc = RewardCalculator(kill_reward=200.0, destruction_penalty=-200.0)

    # No kills, destroyed
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(reward - (-200.0)) < 0.01

    # One kill, destroyed
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=1,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(reward - 0.0) < 0.01  # kill + destruction = 0

    # Two kills, destroyed
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=2,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(reward - 200.0) < 0.01

    # Kill without dying (alive unit)
    class _MockUnit:
        position = type("P", (), {"lon": 0, "lat": 0, "alt": 5000})()

    reward, _, _, _ = calc.compute_total_reward(
        unit=_MockUnit(),
        map_limits=None,
        enemy_kill_count=1,
        destroyed=False,
    )
    assert abs(reward - 200.0) < 0.01


def test_kill_vs_boundary_rewards():
    """Boundary death and combat death carry the same magnitude penalty."""
    calc = RewardCalculator(
        kill_reward=200.0,
        destruction_penalty=-200.0,
        boundary_violation_penalty=-200.0,
    )

    combat_death, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=False,
    )

    boundary_death, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=True,
    )

    assert abs(combat_death - boundary_death) < 0.01


def test_reward_attributes():
    """RewardCalculator exposes the configured values as attributes."""
    calc = RewardCalculator(kill_reward=5.0, destruction_penalty=-3.0)
    assert calc.kill_reward == 5.0
    assert calc.destruction_penalty == -3.0
