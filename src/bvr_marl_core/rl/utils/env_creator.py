"""Environment creator factory for BVR multi-agent training.

Creates and configures environment instances with proper wrapping and setup.
"""

from collections.abc import Callable

from bvr_marl_core.simulator import Simulator

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


def _apply_reward_normalization(base_env, env_cfg: dict):
    """Wrap an env only when reward normalization is explicitly enabled."""
    norm_cfg = env_cfg.get("reward_normalization", {})
    if bool(norm_cfg.get("enabled", False)):
        return RewardNormalizationWrapper(
            base_env,
            clip_reward=float(norm_cfg.get("clip_reward", 10.0)),
        )
    return base_env


def create_simplified_env_creator(cfg) -> Callable[[dict], object]:
    """
    Returns a callable env_creator(env_config) -> SimplifiedMultiAgentEnv.

    Reads from cfg["env"] (plain dict) or cfg.env (OmegaConf).
    """
    from bvr_marl_core.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv

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
        env = _apply_reward_normalization(base_env, merged)
        _set_gymnasium_spec(env, "SimplifiedMultiAgentEnv-v0")
        return env

    return env_creator


def create_env_creator(cfg) -> Callable[[dict], object]:
    """
    Returns a callable env_creator(env_config) -> BVRMultiAgentEnv.

    Reads from cfg["env"] (plain dict) or cfg.env (OmegaConf).
    Wraps the environment with reward normalization for training stability.
    """
    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv

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

        base_env = BVRMultiAgentEnv(merged)

        env = _apply_reward_normalization(base_env, merged)
        _set_gymnasium_spec(env, "BVRMultiAgentEnv-v0")
        return env

    return env_creator
