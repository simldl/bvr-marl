"""
Reward calculator configuration helper.
"""

from air_to_air_rl.rl.environment.rewards.calculator import RewardCalculator


def create_reward_calculator(config: dict) -> RewardCalculator:
    """Create and configure the reward calculator from config dict."""
    return RewardCalculator()
