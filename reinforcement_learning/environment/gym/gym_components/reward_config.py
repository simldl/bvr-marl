"""
Reward calculator configuration helper.
"""

from reinforcement_learning.environment.rewards.calculator import RewardCalculator


def create_reward_calculator(config: dict) -> RewardCalculator:
    """
    Create and configure the reward calculator from config dict.

    Uses terminal/sparse rewards only:
    - Kill rewards
    - Destruction penalties
    - Boundary violation penalties
    - Team survival bonuses
    """
    # Get reward magnitudes from config, with defaults
    reward_magnitudes = config.get("reward_magnitudes", {})

    # Get rewards config if it exists (new format from train_config.yaml)
    rewards_config = config.get("rewards", {})

    return RewardCalculator(
        # Terminal rewards (use rewards section if available, otherwise reward_magnitudes)
        kill_reward=rewards_config.get("kill_reward",
                                      reward_magnitudes.get("kill_reward", 200.0)),
        destruction_penalty=rewards_config.get("destruction_penalty",
                                               reward_magnitudes.get("destruction_penalty", -200.0)),
        boundary_violation_penalty=rewards_config.get("boundary_violation_penalty",
                                                      reward_magnitudes.get("boundary_violation_penalty", -200.0)),
        last_team_reward=rewards_config.get("last_team_reward",
                                           reward_magnitudes.get("last_team_reward", 40.0)),
    )
