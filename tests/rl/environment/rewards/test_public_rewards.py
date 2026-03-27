"""
Tests for the public baseline reward calculator.

This tests the simple terminal reward system provided in the public release.
"""

import pytest

from air_to_air_rl.rl.environment.rewards.calculator import RewardCalculator


class TestPublicRewardCalculator:
    """Test the public baseline reward calculator."""

    def test_initialization(self):
        """Test reward calculator initializes with correct defaults."""
        calc = RewardCalculator()

        assert calc.kill_reward == 1.0
        assert calc.destruction_penalty == -1.0
        assert calc.boundary_violation_penalty == -1.0
        assert calc.last_team_reward == 0.5

    def test_custom_initialization(self):
        """Test reward calculator with custom parameters."""
        calc = RewardCalculator(
            kill_reward=2.0,
            destruction_penalty=-0.5,
            boundary_violation_penalty=-2.0,
            last_team_reward=1.0,
        )

        assert calc.kill_reward == 2.0
        assert calc.destruction_penalty == -0.5
        assert calc.boundary_violation_penalty == -2.0
        assert calc.last_team_reward == 1.0

    def test_no_events(self):
        """Test reward when nothing happens."""
        calc = RewardCalculator()

        reward, sqi, tactical, energy = calc.compute_total_reward(unit=None, map_limits=None)

        assert reward == 0.0
        assert sqi == 0.0
        assert tactical == 0.0
        assert energy == 0.0

    def test_enemy_kill(self):
        """Test reward for killing enemies."""
        calc = RewardCalculator(kill_reward=2.0)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, enemy_kill_count=1
        )

        assert reward == 2.0
        assert sqi == 0.0
        assert tactical == 0.0
        assert energy == 0.0

    def test_multiple_kills(self):
        """Test reward for multiple kills."""
        calc = RewardCalculator(kill_reward=1.5)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, enemy_kill_count=3
        )

        assert reward == 4.5  # 1.5 * 3

    def test_destruction_penalty(self):
        """Test penalty for being destroyed."""
        calc = RewardCalculator(destruction_penalty=-2.0)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, destroyed=True
        )

        assert reward == -2.0

    def test_boundary_violation(self):
        """Test penalty for boundary violation death."""
        calc = RewardCalculator(boundary_violation_penalty=-1.5)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, destroyed=True, died_from_boundary_violation=True
        )

        assert reward == -1.5

    def test_last_team_bonus(self):
        """Test bonus for being last team standing."""
        calc = RewardCalculator(last_team_reward=1.0)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, is_last_team=True
        )

        assert reward == 1.0

    def test_combined_events(self):
        """Test reward with multiple simultaneous events."""
        calc = RewardCalculator(kill_reward=1.0, destruction_penalty=-1.0, last_team_reward=0.5)

        reward, sqi, tactical, energy = calc.compute_total_reward(
            unit=None, map_limits=None, enemy_kill_count=2, destroyed=True, is_last_team=True
        )

        # 2 kills (2.0) + destruction (-1.0) + last team (0.5) = 1.5
        assert reward == 1.5

    def test_reward_breakdown(self):
        """Test reward breakdown returns total."""
        calc = RewardCalculator()

        breakdown = calc.get_reward_breakdown(unit=None, map_limits=None, enemy_kill_count=1)

        assert "total" in breakdown
        assert breakdown["total"] == 1.0

    def test_kwargs_ignored(self):
        """Test that unknown kwargs are silently ignored."""
        calc = RewardCalculator(unknown_param=5.0, another_unknown=True)

        # Should still work with defaults
        assert calc.kill_reward == 1.0

        # Should work with unknown kwargs in compute_total_reward
        reward, _, _, _ = calc.compute_total_reward(
            unit=None, map_limits=None, unknown_kwarg=42, another_unknown="test"
        )

        assert reward == 0.0
