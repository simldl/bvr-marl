"""Training config validator for the public core baseline.

The core repository validates environment/config compatibility without
assuming any private RLModule implementation.  Extension-specific validators can
perform deeper model dry-runs in their own package.
"""

from __future__ import annotations

import traceback


def validate_training_config(config_file: str, training_mode: str = "bvr") -> dict:
    """Dry-run the environment side of a training config.

    Args:
        config_file: Config filename or path, for example ``"basic.yaml"``.
        training_mode: ``"bvr"`` or ``"simplified"``.

    Returns:
        A dict with ``success``, ``summary``, ``details``, and ``warnings`` keys.
    """
    result = {"success": False, "summary": "", "details": [], "warnings": []}

    def log(msg: str) -> None:
        result["details"].append(msg)

    def warn(msg: str) -> None:
        result["warnings"].append(msg)

    try:
        from gymnasium.spaces.utils import flatdim as space_flatdim

        from bvr_marl_core.utils import load_config
        from bvr_marl_core.utils.config_loader import resolve_training_config

        config_path = resolve_training_config(config_file)
        if not config_path.exists():
            result["summary"] = f"Config file not found: {config_path}"
            return result

        cfg = load_config(config_path).to_dict()
        log(f"Loaded config: {config_path}")

        if training_mode == "simplified":
            from bvr_marl_core.rl.utils import create_simplified_env_creator

            env_creator = create_simplified_env_creator(cfg)
            env_label = "SimplifiedMultiAgentEnv"
        else:
            from bvr_marl_core.rl.utils import create_env_creator

            env_creator = create_env_creator(cfg)
            env_label = "BVRMultiAgentEnv"

        probe_env = env_creator({})
        try:
            probe_env.reset()
            obs_spaces = probe_env.observation_space
            act_spaces = probe_env.action_space
        finally:
            close = getattr(probe_env, "close", None)
            if callable(close):
                close()

        log(f"Environment created: {env_label}")

        aid = next(iter(obs_spaces.keys()))
        obs_space = obs_spaces[aid]
        act_space = act_spaces[aid]
        obs_flatdim = space_flatdim(obs_space)
        act_dim = int(act_space.shape[0])
        log(f"Sample agent: {aid}")
        log(f"Observation flatdim: {obs_flatdim}")
        log(f"Action dim: {act_dim}")

        model_cfg = cfg.get("model", {}).get("model_config", {})
        config_act_dim = model_cfg.get("action_dim")
        if config_act_dim is not None and int(config_act_dim) != act_dim:
            warn(
                f"action_dim in model config ({config_act_dim}) differs from "
                f"env action space ({act_dim})"
            )

        result["success"] = True
        result["summary"] = f"obs={obs_flatdim}d  act={act_dim}d  default RLlib PPO"

    except Exception as e:
        result["summary"] = f"{type(e).__name__}: {e}"
        result["details"].append(traceback.format_exc())

    return result


def render_validation_result(result: dict) -> None:
    """Render a validation result into Streamlit."""
    import streamlit as st

    if result["success"]:
        st.success(f"Config OK: {result['summary']}")
    else:
        st.error(f"Validation failed: {result['summary']}")

    for warning in result.get("warnings", []):
        st.warning(warning)

    with st.expander("Validation details", expanded=not result["success"]):
        for line in result.get("details", []):
            st.text(line)
