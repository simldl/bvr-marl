"""
Config Architecture Validator

Performs a dry-run check that a training config loads successfully
and the environment can be instantiated before training is launched.
"""

import traceback


def validate_training_config(config_file: str, training_mode: str = "bvr") -> dict:
    """
    Dry-run check for a training config.

    Args:
        config_file: Config filename or path (e.g. "standard.yaml")
        training_mode: "bvr" or "simplified"

    Returns:
        dict with keys:
          success  (bool)
          summary  (str)   — one-line result
          details  (list[str]) — step-by-step log
          warnings (list[str]) — non-fatal issues
    """
    result = {"success": False, "summary": "", "details": [], "warnings": []}

    def log(msg):
        result["details"].append(msg)

    try:
        # ── 1. Load config ───────────────────────────────────────────────────
        from air_to_air_rl.utils.config_loader import resolve_training_config
        from air_to_air_rl.utils.simple_config import load_config

        config_path = resolve_training_config(config_file)
        if not config_path.exists():
            result["summary"] = f"Config file not found: {config_path}"
            return result

        cfg = load_config(config_path).to_dict()
        log(f"Loaded config: {config_file}")

        # ── 2. Create probe environment ──────────────────────────────────────
        if training_mode == "simplified":
            from air_to_air_rl.rl.utils import create_simplified_env_creator

            env_creator = create_simplified_env_creator(cfg)
            env_label = "SimplifiedMultiAgentEnv"
        else:
            from air_to_air_rl.rl.utils import create_env_creator

            env_creator = create_env_creator(cfg)
            env_label = "BVRMultiAgentEnv"

        probe_env = env_creator({})
        probe_env.reset()
        obs_spaces = probe_env.observation_space
        act_spaces = probe_env.action_space
        log(f"Environment created: {env_label}")

        # ── 3. Compute space dimensions ──────────────────────────────────────
        aid = next(iter(obs_spaces.keys()))
        obs_spaces[aid]
        act_space = act_spaces[aid]
        act_dim = act_space.shape[0]
        log(f"Action dim: {act_dim}")

        # ── 4. Cross-check model config ──────────────────────────────────────
        model_cfg = cfg.get("model", {}).get("model_config", {})
        config_act_dim = model_cfg.get("action_dim", None)

        if config_act_dim is not None and config_act_dim != act_dim:
            result["warnings"].append(
                f"action_dim in model config ({config_act_dim}) differs from env action space ({act_dim})"
            )

        log("Config validation complete (using Ray RLlib default network)")

        result["success"] = True
        result["summary"] = f"act={act_dim}d  env={env_label}  network=Ray default PPO"

    except Exception as e:
        result["summary"] = f"{type(e).__name__}: {e}"
        result["details"].append(traceback.format_exc())

    return result


def render_validation_result(result: dict):
    """Render validation result into Streamlit."""
    import streamlit as st

    if result["success"]:
        st.success(f"Architecture OK — {result['summary']}")
    else:
        st.error(f"Validation failed — {result['summary']}")

    for w in result.get("warnings", []):
        st.warning(f"Warning: {w}")

    with st.expander("Validation details", expanded=not result["success"]):
        for line in result.get("details", []):
            st.text(line)
