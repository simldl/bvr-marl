"""Integration tests for the minimal reward calculator."""

from bvr_marl_core.rl.environment.rewards import RewardCalculator


class _MockUnit:
    position = type("P", (), {"lon": 0, "lat": 0, "alt": 5000})()
    locked_targets = []


def test_complete_reward_flow():
    """Terminal-only scenarios produce correct reward values."""
    calc = RewardCalculator(
        kill_reward=200.0,
        destruction_penalty=-200.0,
        boundary_violation_penalty=-200.0,
        last_team_reward=40.0,
    )

    # 2 kills + last team
    reward, _, _, _ = calc.compute_total_reward(
        unit=_MockUnit(),
        map_limits=None,
        enemy_kill_count=2,
        destroyed=False,
        is_last_team=True,
    )
    assert abs(reward - (400.0 + 40.0)) < 0.01

    # Trade (1 kill + 1 death)
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=1,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(reward - 0.0) < 0.01

    # Boundary death
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=True,
    )
    assert abs(reward - (-200.0)) < 0.01

    # 3 kills + last team
    reward, _, _, _ = calc.compute_total_reward(
        unit=_MockUnit(),
        map_limits=None,
        enemy_kill_count=3,
        destroyed=False,
        is_last_team=True,
    )
    assert abs(reward - (600.0 + 40.0)) < 0.01

    # Pure death
    reward, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(reward - (-200.0)) < 0.01


def test_reward_consistency():
    """Same inputs produce same outputs across repeated calls."""
    calc = RewardCalculator(
        kill_reward=200.0,
        destruction_penalty=-200.0,
        boundary_violation_penalty=-200.0,
    )

    # Boundary == combat death (same magnitude)
    combat, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    boundary, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=0,
        destroyed=True,
        died_from_boundary_violation=True,
    )
    assert abs(combat - boundary) < 0.01

    # Kill/death symmetry
    assert abs(calc.kill_reward) == abs(calc.destruction_penalty)

    # Even trade = zero-sum
    trade, _, _, _ = calc.compute_total_reward(
        unit=None,
        map_limits=None,
        enemy_kill_count=1,
        destroyed=True,
        died_from_boundary_violation=False,
    )
    assert abs(trade) < 0.01


def test_reward_breakdown():
    """get_reward_breakdown returns a dict with a 'total' key."""
    calc = RewardCalculator(kill_reward=10.0)
    breakdown = calc.get_reward_breakdown(
        unit=_MockUnit(), map_limits=None, enemy_kill_count=1, destroyed=False
    )
    assert isinstance(breakdown, dict)
    assert "total" in breakdown
    assert breakdown["total"] == 10.0
