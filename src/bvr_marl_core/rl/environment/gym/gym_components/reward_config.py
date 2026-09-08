"""
Reward calculator factory for the core minimal reward system.

Creates a :class:`~bvr_marl_core.rl.environment.rewards.RewardCalculator`
from an environment config dict. Only the four terminal-reward magnitudes
are forwarded; all other reward parameters are accepted and ignored so that
configs written for an extended system still load.

For the full modular factory (tactical, energy, control, defensive) use an
extension package's ``rl.environment.gym.reward_config.create_reward_calculator``.
"""

from bvr_marl_core.rl.environment.rewards.calculator import RewardCalculator


def create_reward_calculator(config: dict) -> RewardCalculator:
    """Create the minimal terminal-only reward calculator from config."""
    magnitudes = config.get("reward_magnitudes", {})
    return RewardCalculator(
        kill_reward=magnitudes.get("kill_reward", 1.0),
        destruction_penalty=magnitudes.get("destruction_penalty", -1.0),
        boundary_violation_penalty=magnitudes.get("boundary_violation_penalty", -1.0),
        last_team_reward=magnitudes.get("last_team_reward", 0.5),
        reward_information_mode=config.get("reward_information_mode", "observation_only"),
    )
