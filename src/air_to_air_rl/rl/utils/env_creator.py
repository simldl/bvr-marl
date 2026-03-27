"""Environment creator factory for BVR multi-agent training.

Creates and configures environment instances with proper wrapping and setup.
"""

from collections.abc import Callable

from air_to_air_rl.simulator.simulator import Simulator

from .reward_wrapper import RewardNormalizationWrapper
from .type_maps import resolve_aircraft_config


def _set_gymnasium_spec(env, env_id: str) -> None:
    """Set env.spec to a minimal EnvSpec so gymnasium.make()'s assertion passes.

    The new RLlib API stack wraps envs in SyncVectorMultiAgentEnv which calls
    gymnasium.make() internally. gymnasium.make() asserts `env.spec is not None`
    after creating the env via a callable — but callables don't auto-set spec.
    """
    if getattr(env, "spec", None) is not None:
        return
    try:
        from gymnasium.envs.registration import EnvSpec

        env.spec = EnvSpec(id=env_id)
    except Exception:
        # Fallback: minimal duck-typed spec object
        env.spec = type("_EnvSpec", (), {"id": env_id, "max_episode_steps": None})()


def _extract_env_cfg(cfg) -> dict:
    """Extract env sub-config from either a plain dict or OmegaConf DictConfig."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        env = cfg.get("env", {})
        return dict(env) if env else {}
    # OmegaConf DictConfig
    if hasattr(cfg, "env"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg.env, resolve=True)
    return {}


def create_simplified_env_creator(cfg) -> Callable[[dict], object]:
    """
    Returns a callable env_creator(env_config) -> SimplifiedMultiAgentEnv.

    Reads from cfg["env"] (plain dict) or cfg.env (OmegaConf).
    """
    from air_to_air_rl.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv

    def env_creator(env_config: dict):
        merged = _extract_env_cfg(cfg)
        merged.update(env_config or {})

        # Resolve aircraft type strings -> classes for the BVR spawning layer
        agent_str = merged.pop("agent_aircraft_type", None) or "Eurofighter"
        opp_str = merged.pop("opponent_aircraft_type", None) or "Eurofighter"
        merged["aircraft_config"] = {"agent_type": agent_str, "opponent_type": opp_str}
        aircraft_config = resolve_aircraft_config(merged)
        merged.update(aircraft_config)

        if "simulator" not in merged:
            merged["simulator"] = Simulator(weapon_config=merged.get("weapon_config", {}))

        base_env = SimplifiedMultiAgentEnv(merged)
        wrapped = RewardNormalizationWrapper(base_env, clip_reward=10.0)
        _set_gymnasium_spec(wrapped, "SimplifiedMultiAgentEnv-v0")
        return wrapped

    return env_creator


def create_env_creator(cfg) -> Callable[[dict], object]:
    """
    Returns a callable env_creator(env_config) -> BVRMultiAgentEnv.

    Reads from cfg["env"] (plain dict) or cfg.env (OmegaConf).
    Wraps the environment with reward normalization for training stability.
    """
    from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

    def env_creator(env_config: dict):
        merged = _extract_env_cfg(cfg)
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
        _set_gymnasium_spec(wrapped_env, "BVRMultiAgentEnv-v0")
        return wrapped_env

    return env_creator
