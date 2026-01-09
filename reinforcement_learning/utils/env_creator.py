"""Environment creator factory for BVR multi-agent training.

Creates and configures environment instances with proper wrapping and setup.
"""

from typing import Callable
from omegaconf import OmegaConf
from simulator.simulator import Simulator
from .type_maps import resolve_aircraft_config
from .reward_wrapper import RewardNormalizationWrapper


def create_env_creator(cfg) -> Callable[[dict], object]:
    """
    Returns a callable env_creator(env_config) -> BVRMultiAgentEnv.

    This factory function creates an environment creator that:
    - Merges Hydra config (cfg.env) with per-instance overrides (env_config)
    - Ensures a Simulator is present (caller may still override)
    - Wraps environment with reward normalization for training stability

    Args:
        cfg: Hydra configuration object with env settings

    Returns:
        Callable that creates configured environment instances
    """
    from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

    def env_creator(env_config: dict):
        """
        Create a configured BVR multi-agent environment instance.

        Args:
            env_config: Per-instance environment configuration overrides

        Returns:
            Wrapped BVRMultiAgentEnv instance with reward normalization
        """
        # Merge Hydra config with instance overrides
        merged = OmegaConf.to_container(cfg.env, resolve=True) if cfg and hasattr(cfg, "env") else {}
        merged.update(env_config or {})

        # Resolve aircraft configuration from strings to classes
        aircraft_config = resolve_aircraft_config(merged)
        merged.update(aircraft_config)

        # Create simulator if not provided
        if "simulator" not in merged:
            weapon_config = merged.get("weapon_config", {})
            merged["simulator"] = Simulator(weapon_config=weapon_config)

        # Create base environment
        base_env = BVRMultiAgentEnv(merged)

        # Wrap with reward normalization for training stability
        # This helps prevent extreme loss values by keeping rewards in a consistent scale
        wrapped_env = RewardNormalizationWrapper(base_env, clip_reward=10.0)

        return wrapped_env

    return env_creator
