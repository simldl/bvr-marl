"""
Training Configuration Builder

Interface for creating, editing and tracking training configurations.
Saves configurations to configs/training/ at the project root.
"""

import json
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

_LEGACY_REWARD_CATEGORY_MAP = {
    "enable_terminal": "terminal",
    "enable_tactical": "tactical",
    "enable_energy": "energy",
    "enable_control": "control",
    "enable_defensive": "defensive",
}

_REWARD_CATEGORY_DEFAULTS = {
    "terminal": True,
    "tactical": True,
    "energy": False,
    "control": False,
    "defensive": False,
    "line_objective": True,
}

_REWARD_MAGNITUDE_DEFAULTS = {
    "kill_reward": 200.0,
    "destruction_penalty": -200.0,
    "boundary_violation_penalty": -200.0,
    "last_team_reward": 40.0,
    "tracking_reward_scale": 0.5,
    "nez_positioning_reward_scale": 1.5,
    "sqi_shot_bonus_scale": 0.0,
    "optimal_zone_reward_scale": 0.8,
    "sqi_bonus_threshold": 0.55,
    "sqi_delta_shaping_scale": 0.0,
    "energy_reward_scale": 0.3,
    "low_altitude_penalty_scale": -2.0,
    "low_altitude_threshold_m": 5000.0,
    "altitude_loss_penalty_scale": -0.1,
    "lift_balance_penalty_scale": -0.8,
    "heading_alignment_reward_scale": 0.2,
    "passivity_penalty_scale": -1.0,
    "evasion_reward_scale": 0.8,
    "boundary_progressive_penalty_scale": -10.0,
}

_LINE_OBJECTIVE_DEFAULTS = {
    "enabled": True,
    "attacker_team": "A",
    "defender_team": "B",
    "penetration_line_north_m": 100000.0,
    "penetration_axis_rad": 0.0,
    "crossing_buffer_m": 2000.0,
    "count_crossing_once": True,
    "attacker_crossing_bonus": 80.0,
    "attacker_team_crossing_bonus": 120.0,
    "attacker_progress_scale": 20.0,
    "attacker_failure_penalty": -80.0,
    "attacker_stagnation_penalty": -0.01,
    "defender_hold_reward": 0.05,
    "defender_penetration_penalty": -120.0,
    "defender_terminal_success_reward": 80.0,
    "attacker_success_requires_crossing": True,
    "defender_destroyed_counts_as_attacker_success": False,
    "defender_no_cross_enabled": True,
    "defender_crossing_penalty": -20.0,
    "defender_allowed_buffer_m": 5000.0,
}


class ConfigManager:
    """Manages configuration creation, editing and storage."""

    def __init__(self):
        from bvr_marl_core.utils.paths import core_project_root, rl_configs_root

        self.config_dir = core_project_root() / "configs" / "training"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._package_config_dir = rl_configs_root()

    def save_config(self, config: dict[str, Any], name: str) -> str:
        """Save configuration to configs/training/ directory."""
        filename = f"{name}.yaml"
        filepath = self.config_dir / filename
        config = deepcopy(config)
        migrate_reward_schema(config)

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return str(filepath)

    def load_config(self, filepath: str) -> dict[str, Any]:
        """Load configuration from file."""
        with open(filepath) as f:
            return yaml.safe_load(f)

    def get_existing_configs(self) -> list[str]:
        """Get list of existing configuration files (user configs first, then package defaults)."""
        seen: set[str] = set()
        configs: list[str] = []
        for d in [self.config_dir, self._package_config_dir]:
            if d.exists():
                for config_file in sorted(d.glob("*.yaml")):
                    if config_file.name not in seen:
                        seen.add(config_file.name)
                        configs.append(str(config_file))
        return configs


def validate_config_for_save(
    config: dict[str, Any],
    name: str,
    existing_files: list[str],
) -> tuple[list[str], list[str]]:
    """Validate a config before saving.  Returns ``(errors, warnings)``.

    Errors block saving; warnings are shown but do not block.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Name ---
    if not name:
        errors.append("Configuration name cannot be empty.")
    elif not re.match(r"^[A-Za-z0-9_\-]+$", name):
        errors.append(
            f"Invalid config name **{name!r}** - use only letters, digits, hyphens, and underscores."
        )
    else:
        target = f"{name}.yaml"
        if any(Path(f).name == target for f in existing_files):
            warnings.append(
                f"A file named **{target}** already exists and will be **overwritten**."
            )

    # --- training_mode marker ---
    training_mode = config.get("training_mode")
    if training_mode not in ("simplified", "bvr"):
        warnings.append(
            "Missing or unrecognised `training_mode`. "
            "Set it to `simplified` or `bvr` in **General Settings -> Environment Type** "
            "so the dashboard shows this config in the correct training tab."
        )

    # --- Simplified/normal model consistency ---
    is_simplified = training_mode == "simplified"
    use_wrapper = config.get("model", {}).get("use_neural_wrapper", False)
    if is_simplified and use_wrapper:
        errors.append(
            "**Simplified** configs must have `use_neural_wrapper: false` "
            "(the simplified environment has a fixed 4D action space). "
            "Go to **Networks -> Action Space** to fix this."
        )

    # --- Missile lists ---
    missile_cfg = config.get("env", {}).get("missile_config", {})
    if not missile_cfg.get("agent_missiles"):
        errors.append("Agent missile list is empty - add at least one missile type.")
    if not missile_cfg.get("opponent_missiles"):
        errors.append("Opponent missile list is empty - add at least one missile type.")

    # --- Duplicate rollout_fragment_length keys ---
    top_rfl = config.get("rollout_fragment_length")
    nested_rfl = config.get("training", {}).get("rollout_fragment_length")
    if top_rfl is not None and nested_rfl is not None:
        warnings.append(
            "`rollout_fragment_length` is set both at the top level "
            f"(`{top_rfl}`) and inside `training:` (`{nested_rfl}`). "
            "The top-level value takes effect; remove the nested one to avoid confusion."
        )

    # --- rollout_fragment_length vs batch-size consistency ---
    rfl = top_rfl if top_rfl is not None else nested_rfl
    if rfl is not None and rfl != "auto" and isinstance(rfl, int):
        num_runners = config.get("num_env_runners", 1)
        batch_size = config.get("training", {}).get("train_batch_size", 0)
        collected = num_runners * rfl
        if batch_size and collected != batch_size:
            errors.append(
                f"`rollout_fragment_length` ({rfl}) * `num_env_runners` ({num_runners}) "
                f"= **{collected}** but `train_batch_size` = **{batch_size}**. "
                f"These must match. Set `rollout_fragment_length` to **auto** "
                f"(recommended) or to **{batch_size // max(num_runners, 1)}**."
            )

    return errors, warnings


def migrate_reward_schema(config: dict[str, Any], *, warn: bool = False) -> bool:
    """Migrate old ``training.reward_config`` into canonical env reward schema.

    Returns True when both old and new schemas were present.
    """
    env = config.setdefault("env", {})
    training = config.setdefault("training", {})
    legacy = training.get("reward_config")
    categories = env.setdefault("reward_categories", {})
    magnitudes = env.setdefault("reward_magnitudes", {})
    had_canonical = bool(categories or magnitudes)
    had_legacy = isinstance(legacy, dict)
    had_both = bool(had_legacy and had_canonical)

    if had_legacy:
        for old_key, new_key in _LEGACY_REWARD_CATEGORY_MAP.items():
            if old_key in legacy and new_key not in categories:
                categories[new_key] = bool(legacy[old_key])
        for key, value in legacy.items():
            if key not in _LEGACY_REWARD_CATEGORY_MAP:
                magnitudes.setdefault(key, value)
        training.pop("reward_config", None)

    for key, value in _REWARD_CATEGORY_DEFAULTS.items():
        categories.setdefault(key, value)
    for key, value in _REWARD_MAGNITUDE_DEFAULTS.items():
        magnitudes.setdefault(key, value)

    if warn and had_both:
        st.warning(
            "This config contained both legacy `training.reward_config` and canonical "
            "`env.reward_categories` / `env.reward_magnitudes`. Canonical values were kept."
        )

    return had_both


def reward_view_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the legacy-shaped view used by the existing Streamlit controls."""
    migrate_reward_schema(config)
    env = config.setdefault("env", {})
    categories = env.setdefault("reward_categories", {})
    magnitudes = env.setdefault("reward_magnitudes", {})
    view = {
        "enable_terminal": bool(categories.get("terminal", True)),
        "enable_tactical": bool(categories.get("tactical", True)),
        "enable_energy": bool(categories.get("energy", False)),
        "enable_control": bool(categories.get("control", False)),
        "enable_defensive": bool(categories.get("defensive", False)),
        "enable_line_objective": bool(categories.get("line_objective", True)),
    }
    view.update(magnitudes)
    return view


def write_reward_view_to_config(config: dict[str, Any], reward_config: dict[str, Any]) -> None:
    """Write the Streamlit reward view back to canonical env keys."""
    env = config.setdefault("env", {})
    training = config.setdefault("training", {})
    categories = env.setdefault("reward_categories", {})
    magnitudes = env.setdefault("reward_magnitudes", {})
    categories.update(
        {
            "terminal": bool(reward_config.get("enable_terminal", True)),
            "tactical": bool(reward_config.get("enable_tactical", True)),
            "energy": bool(reward_config.get("enable_energy", False)),
            "control": bool(reward_config.get("enable_control", False)),
            "defensive": bool(reward_config.get("enable_defensive", False)),
            "line_objective": bool(reward_config.get("enable_line_objective", True)),
        }
    )
    for key in _REWARD_MAGNITUDE_DEFAULTS:
        if key in reward_config:
            magnitudes[key] = reward_config[key]
    training.pop("reward_config", None)


def config_builder():
    """Main training configuration builder interface."""
    st.header("Training Configuration Builder")
    st.caption(
        "Create, validate, compare, and save reinforcement-learning training configurations."
    )

    st.info("""
    **Training Configuration Builder**
    Create and edit training configurations for the reinforcement learning system.
    Configurations are saved to: `configs/training/`
    """)

    # Initialize config manager
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigManager()

    # Initialize current config in session state
    if "current_config" not in st.session_state:
        st.session_state.current_config = create_default_config()

    # Sidebar for config management
    with st.sidebar:
        st.subheader("Configuration Management")

        # Load existing config
        existing_configs = st.session_state.config_manager.get_existing_configs()
        config_names = ["Create New"] + [Path(f).name for f in existing_configs]

        selected_config = st.selectbox(
            "Load Configuration:",
            config_names,
            help="Load an existing configuration or create a new one",
        )

        if selected_config != "Create New":
            config_path = next(f for f in existing_configs if Path(f).name == selected_config)
            if st.button("Load Selected Config"):
                st.session_state.current_config = st.session_state.config_manager.load_config(
                    config_path
                )
                st.success(f"Loaded: {selected_config}")
                st.rerun()

        # Save current config
        st.markdown("---")
        config_name = st.text_input(
            "Configuration Name:",
            value="",
            placeholder="my_training_config",
            help="Name for this configuration (letters, digits, hyphens, underscores only)",
        )

        save_errors, save_warnings = validate_config_for_save(
            st.session_state.current_config,
            config_name,
            existing_configs,
        )

        has_overwrite_warning = any("overwritten" in w for w in save_warnings)

        for err in save_errors:
            st.error(err)
        for warn in save_warnings:
            st.warning(warn)

        # Overwrite confirmation checkbox (only shown when the file already exists)
        confirmed_overwrite = True
        if has_overwrite_warning and not save_errors:
            confirmed_overwrite = st.checkbox(
                "Confirm overwrite of existing file",
                value=False,
                key="confirm_overwrite",
            )

        save_disabled = bool(save_errors) or (has_overwrite_warning and not confirmed_overwrite)

        if st.button("Save Configuration", width="stretch", disabled=save_disabled):
            filepath = st.session_state.config_manager.save_config(
                st.session_state.current_config,
                config_name,
            )
            st.success(f"Saved: {filepath}")

    # Main configuration tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["General Settings", "Training", "Networks", "Rewards", "Compare Configs"]
    )

    with tab1:
        render_general_settings()

    with tab2:
        render_training_settings()

    with tab3:
        render_network_settings()

    with tab4:
        render_rewards_settings()

    with tab5:
        render_config_comparison()


def create_default_config() -> dict[str, Any]:
    """Create a default configuration template."""
    return {
        "env": {
            "num_agents_per_side": 2,
            "map_size": 200,
            "max_steps": 1024,
            "debug": False,
            "simulation_config": {
                "tick_secs": 1.0,
                "max_real_time_s": None,
                "early_termination": {
                    "all_enemies_dead": True,
                    "mission_complete": True,
                    "no_missiles_remaining": False,
                },
            },
            "aircraft_config": {"agent_type": "Eurofighter", "opponent_type": "Eurofighter"},
            "missile_config": {
                "agent_missiles": ["AIM120_AMRAAM"],
                "opponent_missiles": ["AIM120_AMRAAM"],
            },
            "datalink_mode": "full",
            "scenario_config": {
                "allowed_regimes": [],
                "randomize_regime": True,
                "awacs_config": {
                    "agent_awacs": True,
                    "opponent_awacs": True,
                    "orbit_altitude_m": 10000.0,
                    "awacs_non_engageable": True,
                },
                "line_objective": deepcopy(_LINE_OBJECTIVE_DEFAULTS),
            },
            "reward_categories": deepcopy(_REWARD_CATEGORY_DEFAULTS),
            "reward_magnitudes": deepcopy(_REWARD_MAGNITUDE_DEFAULTS),
            "reward_normalization": {
                "enabled": False,
                "clip_reward": 10.0,
            },
        },
        "framework": "torch",
        "num_gpus": 1,
        # Top-level keys read directly by build_ppo_config
        "num_env_runners": 4,
        "rollout_fragment_length": "auto",
        "training": {
            "steps": 512,
            "learning_rate": 0.0003,
            "train_batch_size": 4096,
            "sgd_minibatch_size": 128,
            "num_epochs": 3,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "vf_loss_coeff": 0.5,
            "kl_coeff": 0.2,
            "gamma": 0.995,
            "use_gae": True,
            "lambda": 0.95,
            "grad_clip": 1.0,
            "multi_agent": {
                "policy_mode": "shared",
                "shared_policy_id": "shared_policy",
                "attacker_policy_id": "attacker_policy",
                "defender_policy_id": "defender_policy",
                "train_attacker": True,
                "train_defender": True,
            },
        },
        "model": {
            "use_neural_wrapper": True,
            "model_config": {
                "action_dim": 4,
                "wrapped_action_dim": 4,
                "full_action_dim": 10,
                "active_indices": [0, 1, 2, 3],
                "hidden_dim": 256,
                "num_hidden_layers": 2,
                "activation": "tanh",
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
                "free_log_std": False,
            },
        },
        "logging": {
            "log_dir": "logs/training",
            "save_dir": "models/training",
            "model_name": "training_model",
            "use_tensorboard": True,
        },
    }


def _ensure_env_defaults(config: dict[str, Any]):
    """Ensure all expected nested env keys exist so the UI never KeyErrors on a loaded config."""
    env = config.setdefault("env", {})

    # simulation_config may be a string in older YAML files; normalise to dict
    sim = env.get("simulation_config", {})
    if isinstance(sim, str):
        import ast

        try:
            sim = ast.literal_eval(sim)
        except Exception:
            sim = {}
    env["simulation_config"] = sim
    sim.setdefault("tick_secs", 1.0)
    sim.setdefault("max_real_time_s", None)
    sim.setdefault(
        "early_termination",
        {
            "all_enemies_dead": True,
            "mission_complete": True,
            "no_missiles_remaining": False,
        },
    )

    env.setdefault("aircraft_config", {"agent_type": "Eurofighter", "opponent_type": "Eurofighter"})
    if isinstance(env["aircraft_config"], str):
        import ast

        try:
            env["aircraft_config"] = ast.literal_eval(env["aircraft_config"])
        except Exception:
            env["aircraft_config"] = {"agent_type": "Eurofighter", "opponent_type": "Eurofighter"}

    env.setdefault(
        "missile_config",
        {"agent_missiles": ["AIM120_AMRAAM"], "opponent_missiles": ["AIM120_AMRAAM"]},
    )
    if isinstance(env["missile_config"], str):
        import ast

        try:
            env["missile_config"] = ast.literal_eval(env["missile_config"])
        except Exception:
            env["missile_config"] = {
                "agent_missiles": ["AIM120_AMRAAM"],
                "opponent_missiles": ["AIM120_AMRAAM"],
            }

    env.setdefault("num_fm", 4)

    env.setdefault("scenario_config", {})
    env["scenario_config"].setdefault(
        "awacs_config",
        {
            "agent_awacs": True,
            "opponent_awacs": True,
            "orbit_altitude_m": 10000.0,
            "awacs_non_engageable": True,
        },
    )
    line_cfg = env["scenario_config"].setdefault(
        "line_objective", deepcopy(_LINE_OBJECTIVE_DEFAULTS)
    )
    if isinstance(line_cfg, dict):
        for key, value in _LINE_OBJECTIVE_DEFAULTS.items():
            line_cfg.setdefault(key, value)
    env.setdefault("reward_normalization", {"enabled": False, "clip_reward": 10.0})

    env.setdefault("num_agents_per_side", env.get("num_agents_per_team", 2))
    env.setdefault("map_size", env.get("map_size_km", 200))
    env.setdefault("max_steps", 1024)


def render_general_settings():
    """Render general environment settings page."""
    st.subheader("General Environment Settings")

    config = st.session_state.current_config
    _ensure_env_defaults(config)

    # Environment Type Selection
    st.markdown("### Environment Type")
    environment_types = ["Standard BVR", "Simplified"]
    env_type = st.radio(
        "Select Environment:",
        environment_types,
        index=environment_types.index(
            "Simplified" if config.get("training_mode") == "simplified" else "Standard BVR"
        ),
        help="Standard: Full BVR with radar, AWACS, etc. Simplified: Truth observations, reduced action space.",
    )
    # Persist so render_network_settings can read it
    st.session_state["training_env_type"] = env_type
    config["training_mode"] = "simplified" if env_type == "Simplified" else "bvr"

    # Eagerly fix model config when Simplified is selected so that saving
    # without ever visiting the Networks tab still produces a valid config.
    if env_type == "Simplified":
        config.setdefault("model", {}).setdefault("model_config", {})
        config["model"]["use_neural_wrapper"] = False
        config["model"]["model_config"]["action_dim"] = 4
        config["model"]["model_config"]["wrapped_action_dim"] = 4
        config["model"]["model_config"]["full_action_dim"] = 4
        config["model"]["model_config"]["active_indices"] = [0, 1, 2, 3]

    # Team Configuration
    st.markdown("### Team Configuration")
    col1, col2 = st.columns(2)

    with col1:
        config["env"]["num_agents_per_side"] = st.number_input(
            "Agents per Side:",
            min_value=1,
            max_value=8,
            value=config["env"].get("num_agents_per_side", 2),
            help="Number of agents on each team",
        )

        config["env"]["map_size"] = st.number_input(
            "Map Size (km):",
            min_value=50,
            max_value=500,
            value=config["env"].get("map_size", 200),
            help="Size of the battle area",
        )

    with col2:
        config["env"]["max_steps"] = st.number_input(
            "Max Episode Steps:",
            min_value=100,
            max_value=5000,
            value=config["env"].get("max_steps", 1024),
            help="Maximum simulation steps per episode",
        )

        config["env"]["simulation_config"]["tick_secs"] = st.number_input(
            "Simulation Tick (seconds):",
            min_value=0.1,
            max_value=10.0,
            value=config["env"]["simulation_config"].get("tick_secs", 1.0),
            help="Time per simulation step",
        )

    # Aircraft Configuration
    st.markdown("### Aircraft Configuration")
    aircraft_types = ["Eurofighter", "F22", "F35", "Su57", "AWACS", "DebugPlane"]

    col1, col2 = st.columns(2)
    with col1:
        config["env"]["aircraft_config"]["agent_type"] = st.selectbox(
            "Agent Aircraft:",
            aircraft_types,
            index=aircraft_types.index(
                config["env"]["aircraft_config"].get("agent_type", "Eurofighter")
            ),
        )

    with col2:
        config["env"]["aircraft_config"]["opponent_type"] = st.selectbox(
            "Opponent Aircraft:",
            aircraft_types,
            index=aircraft_types.index(
                config["env"]["aircraft_config"].get("opponent_type", "Eurofighter")
            ),
        )

    # Missile Configuration
    st.markdown("### Missile Configuration")
    missile_types = ["AIM120_AMRAAM", "Meteor", "K77M", "DefaultMissile"]

    config["env"]["missile_config"]["agent_missiles"] = st.multiselect(
        "Agent Missiles:",
        missile_types,
        default=config["env"]["missile_config"].get("agent_missiles", ["AIM120_AMRAAM"]),
    )

    config["env"]["missile_config"]["opponent_missiles"] = st.multiselect(
        "Opponent Missiles:",
        missile_types,
        default=config["env"]["missile_config"].get("opponent_missiles", ["AIM120_AMRAAM"]),
    )

    config["env"]["num_fm"] = st.number_input(
        "Missiles per Agent (num_fm):",
        min_value=1,
        max_value=12,
        value=int(config["env"].get("num_fm", 4)),
        step=1,
        help="Number of missiles each agent carries at the start of the episode.",
    )

    # AWACS Configuration (BVR only; Simplified env has no AWACS)
    if env_type != "Simplified":
        st.markdown("### AWACS Configuration")
        col1, col2 = st.columns(2)

        with col1:
            config["env"]["scenario_config"]["awacs_config"]["agent_awacs"] = st.checkbox(
                "Blue Team AWACS",
                value=config["env"]["scenario_config"]["awacs_config"].get("agent_awacs", True),
            )

            config["env"]["scenario_config"]["awacs_config"]["opponent_awacs"] = st.checkbox(
                "Red Team AWACS",
                value=config["env"]["scenario_config"]["awacs_config"].get("opponent_awacs", True),
            )

        with col2:
            config["env"]["scenario_config"]["awacs_config"]["orbit_altitude_m"] = st.number_input(
                "AWACS Altitude (m):",
                min_value=5000,
                max_value=20000,
                value=int(
                    config["env"]["scenario_config"]["awacs_config"].get("orbit_altitude_m", 10000)
                ),
            )

            config["env"]["scenario_config"]["awacs_config"]["awacs_non_engageable"] = st.checkbox(
                "AWACS Non-Engageable",
                value=config["env"]["scenario_config"]["awacs_config"].get(
                    "awacs_non_engageable", True
                ),
                help="AWACS cannot be targeted or destroyed",
            )

    # Boundary Configuration
    st.markdown("### Boundary & Termination")

    # Add a boundary section
    if "boundary_config" not in config["env"]:
        config["env"]["boundary_config"] = {
            "enable_boundary": True,
            "boundary_size_km": config["env"].get("map_size", 200),
            "violation_penalty": -50.0,
        }

    config["env"]["boundary_config"]["enable_boundary"] = st.checkbox(
        "Enable Map Boundary",
        value=config["env"]["boundary_config"].get("enable_boundary", True),
        help="Enforce map boundaries with penalties",
    )

    if config["env"]["boundary_config"]["enable_boundary"]:
        config["env"]["boundary_config"]["boundary_size_km"] = st.number_input(
            "Boundary Size (km):",
            min_value=50,
            max_value=500,
            value=config["env"]["boundary_config"].get(
                "boundary_size_km", config["env"].get("map_size", 200)
            ),
            help="Boundary enforcement area",
        )


def render_training_settings():
    """Render training hyperparameters page."""
    st.subheader("Training Hyperparameters")

    config = st.session_state.current_config

    # Migrate old key names from configs saved before the rename
    training = config["training"]
    if "batch_size" in training and "train_batch_size" not in training:
        training["train_batch_size"] = training.pop("batch_size")
    if "n_envs" in training:
        config.setdefault("num_env_runners", training.pop("n_envs"))
    if "rollout_fragment_length" in training and "rollout_fragment_length" not in config:
        config["rollout_fragment_length"] = training.pop("rollout_fragment_length")

    # Training Steps and Resources
    st.markdown("### Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        config["training"]["steps"] = st.number_input(
            "Training Iterations:",
            min_value=1,
            max_value=1000000,
            value=config["training"].get("steps", 512),
            help="Number of PPO training iterations (each collects one batch of experience)",
        )

        config["num_env_runners"] = st.number_input(
            "Parallel Env Runners:",
            min_value=1,
            max_value=128,
            value=config.get("num_env_runners", 4),
            help="Number of parallel Ray env runners collecting experience (`num_env_runners` in RLlib)",
        )

        config["num_gpus"] = st.number_input(
            "Number of GPUs:",
            min_value=0,
            max_value=8,
            value=config.get("num_gpus", 1),
            help="Number of GPUs for the learner",
        )

    with col2:
        config["training"]["train_batch_size"] = st.number_input(
            "Train Batch Size:",
            min_value=256,
            max_value=65536,
            value=config["training"].get("train_batch_size", 4096),
            help=(
                "Total timesteps collected per training iteration "
                "(`train_batch_size_per_learner` in RLlib PPO). "
                "Should be >> SGD Minibatch Size."
            ),
        )

        _rfl_auto = st.checkbox(
            "Auto rollout fragment length (recommended)",
            value=config.get("rollout_fragment_length", "auto") == "auto",
            help="Let Ray compute rollout_fragment_length automatically to match train_batch_size.",
        )
        if _rfl_auto:
            config["rollout_fragment_length"] = "auto"
        else:
            _rfl_val = config.get("rollout_fragment_length", 128)
            config["rollout_fragment_length"] = st.number_input(
                "Rollout Fragment Length:",
                min_value=8,
                max_value=2048,
                value=_rfl_val if isinstance(_rfl_val, int) else 128,
                help=(
                    "Timesteps each env runner collects before sending to the learner. "
                    "Total batch is approximately num_env_runners * rollout_fragment_length."
                ),
            )

        config["training"]["sgd_minibatch_size"] = st.number_input(
            "SGD Minibatch Size:",
            min_value=8,
            max_value=1024,
            value=config["training"].get("sgd_minibatch_size", 128),
            help=(
                "Mini-batch size within each SGD epoch. "
                "For stable PPO updates this should stay below the train batch size."
            ),
        )

    # Learning Parameters
    st.markdown("### Learning Parameters")
    col1, col2 = st.columns(2)

    with col1:
        config["training"]["learning_rate"] = st.number_input(
            "Learning Rate:",
            min_value=1e-6,
            max_value=1e-1,
            value=config["training"].get("learning_rate", 0.0003),
            format="%.6f",
            help="Adam learning rate",
        )

        config["training"]["clip_param"] = st.number_input(
            "PPO Clip Parameter:",
            min_value=0.01,
            max_value=1.0,
            value=config["training"].get("clip_param", 0.2),
            help="PPO surrogate clipping parameter",
        )

        config["training"]["entropy_coef"] = st.number_input(
            "Entropy Coefficient:",
            min_value=0.0,
            max_value=0.1,
            value=config["training"].get("entropy_coef", 0.01),
            help="Entropy bonus coefficient (encourages exploration)",
        )

        config["training"]["kl_coeff"] = st.number_input(
            "KL Coefficient:",
            min_value=0.0,
            max_value=2.0,
            value=config["training"].get("kl_coeff", 0.2),
            help="Adaptive KL penalty coefficient",
        )

    with col2:
        config["training"]["vf_loss_coeff"] = st.number_input(
            "Value Function Loss Coefficient:",
            min_value=0.1,
            max_value=2.0,
            value=config["training"].get("vf_loss_coeff", 0.5),
            help="Scales the value function loss relative to the policy loss",
        )

        config["training"]["num_epochs"] = st.number_input(
            "SGD Epochs per Iteration:",
            min_value=1,
            max_value=20,
            value=config["training"].get("num_epochs", 3),
            help="How many passes over the collected batch per training iteration",
        )

        config["training"]["gamma"] = st.number_input(
            "Discount Factor:",
            min_value=0.9,
            max_value=1.0,
            value=config["training"].get("gamma", 0.995),
            format="%.4f",
            help="Reward discount factor",
        )

        config["training"]["grad_clip"] = st.number_input(
            "Gradient Clip:",
            min_value=0.0,
            max_value=10.0,
            value=config["training"].get("grad_clip", 1.0)
            if config["training"].get("grad_clip")
            else 1.0,
            help="Global-norm gradient clipping threshold (0 = disabled)",
        )

        if config["training"]["grad_clip"] == 0.0:
            config["training"]["grad_clip"] = None

    # GAE Configuration
    st.markdown("### Generalized Advantage Estimation (GAE)")
    col1, col2 = st.columns(2)

    with col1:
        config["training"]["use_gae"] = st.checkbox(
            "Use GAE",
            value=config["training"].get("use_gae", True),
            help="Enable Generalized Advantage Estimation",
        )

    with col2:
        if config["training"]["use_gae"]:
            config["training"]["lambda"] = st.number_input(
                "GAE Lambda:",
                min_value=0.0,
                max_value=1.0,
                value=config["training"].get("lambda", 0.95),
                help="GAE lambda parameter",
            )

    st.markdown("### Multi-Agent Policies")
    ma_cfg = config["training"].setdefault(
        "multi_agent",
        {
            "policy_mode": "shared",
            "shared_policy_id": "shared_policy",
            "attacker_policy_id": "attacker_policy",
            "defender_policy_id": "defender_policy",
            "train_attacker": True,
            "train_defender": True,
        },
    )
    policy_modes = ["shared", "team_separate"]
    current_mode = ma_cfg.get("policy_mode", "shared")
    if current_mode not in policy_modes:
        current_mode = "shared"
    ma_cfg["policy_mode"] = st.selectbox(
        "Policy Mode:",
        policy_modes,
        index=policy_modes.index(current_mode),
        help="Use one shared policy or separate attacker/defender policies.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        ma_cfg["shared_policy_id"] = st.text_input(
            "Shared Policy ID:",
            value=ma_cfg.get("shared_policy_id", "shared_policy"),
        )
    with col2:
        ma_cfg["attacker_policy_id"] = st.text_input(
            "Attacker Policy ID:",
            value=ma_cfg.get("attacker_policy_id", "attacker_policy"),
        )
    with col3:
        ma_cfg["defender_policy_id"] = st.text_input(
            "Defender Policy ID:",
            value=ma_cfg.get("defender_policy_id", "defender_policy"),
        )
    col1, col2 = st.columns(2)
    with col1:
        ma_cfg["train_attacker"] = st.checkbox(
            "Train Attacker Policy",
            value=bool(ma_cfg.get("train_attacker", True)),
        )
    with col2:
        ma_cfg["train_defender"] = st.checkbox(
            "Train Defender Policy",
            value=bool(ma_cfg.get("train_defender", True)),
        )


def render_network_settings():
    """Render public-core model and action-space settings."""
    st.subheader("Model Configuration")

    config = st.session_state.current_config
    model = config.setdefault("model", {})
    model_cfg = model.setdefault("model_config", {})

    st.markdown("### Action Space")
    is_simplified = st.session_state.get("training_env_type", "Standard BVR") == "Simplified"

    if is_simplified:
        st.info(
            "Simplified environments use a fixed 4D action space: "
            "energy, lift angle, lift load, and missile trigger."
        )
        model["use_neural_wrapper"] = False
        model_cfg["action_dim"] = 4
        model_cfg["wrapped_action_dim"] = 4
        model_cfg["full_action_dim"] = 4
        model_cfg["active_indices"] = [0, 1, 2, 3]
    else:
        action_options = {
            4: "4D - flight controls and missile trigger",
            5: "5D - add target selection",
            6: "6D - add gun trigger",
            10: "10D - full action vector",
        }
        dim_values = list(action_options.keys())
        current_dim = model_cfg.get("action_dim", 10)
        if current_dim not in dim_values:
            current_dim = 10

        selected_dim = st.selectbox(
            "Action Dimensions:",
            dim_values,
            index=dim_values.index(current_dim),
            format_func=lambda value: action_options[value],
            help="Choose how much of the public environment action vector the policy controls.",
        )
        model_cfg["action_dim"] = selected_dim
        model_cfg["wrapped_action_dim"] = selected_dim
        model_cfg["full_action_dim"] = 10
        model_cfg["active_indices"] = list(range(selected_dim))
        model["use_neural_wrapper"] = selected_dim < 10

        if selected_dim < 10:
            st.info(
                "Reduced action vectors rely on the environment action processor for the "
                "remaining public control channels."
            )

    st.markdown("### Default PPO Model")
    st.caption(
        "Core training uses RLlib's default PPO model. These fields document the "
        "intended public baseline and are safe to override from custom extensions."
    )

    fcnet_hiddens = model_cfg.get("fcnet_hiddens")
    if not isinstance(fcnet_hiddens, list) or not fcnet_hiddens:
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        num_layers = int(model_cfg.get("num_hidden_layers", 2))
        fcnet_hiddens = [hidden_dim] * num_layers

    col1, col2 = st.columns(2)
    with col1:
        hidden_dim = st.number_input(
            "Hidden Dimension:",
            min_value=64,
            max_value=1024,
            value=int(fcnet_hiddens[0]),
            help="Width of each fully connected hidden layer.",
        )
        num_layers = st.number_input(
            "Hidden Layers:",
            min_value=1,
            max_value=8,
            value=len(fcnet_hiddens),
            help="Number of fully connected hidden layers.",
        )

    with col2:
        activations = ["relu", "tanh"]
        current_activation = model_cfg.get("fcnet_activation", model_cfg.get("activation", "tanh"))
        if current_activation not in activations:
            current_activation = "tanh"
        activation = st.selectbox(
            "Activation Function:",
            activations,
            index=activations.index(current_activation),
        )
        free_log_std = st.checkbox(
            "Free Log Std",
            value=bool(model_cfg.get("free_log_std", False)),
            help="Use RLlib's standard free log standard deviation parameterization.",
        )

    model_cfg["fcnet_hiddens"] = [int(hidden_dim)] * int(num_layers)
    model_cfg["fcnet_activation"] = activation
    model_cfg["free_log_std"] = free_log_std
    model_cfg["hidden_dim"] = int(hidden_dim)
    model_cfg["num_hidden_layers"] = int(num_layers)
    model_cfg["activation"] = activation

    public_model_keys = {
        "action_dim",
        "wrapped_action_dim",
        "full_action_dim",
        "active_indices",
        "hidden_dim",
        "num_hidden_layers",
        "activation",
        "fcnet_hiddens",
        "fcnet_activation",
        "free_log_std",
    }
    for key in list(model_cfg):
        if key not in public_model_keys:
            model_cfg.pop(key, None)


def render_rewards_settings():
    """Render rewards configuration page with explanations."""
    st.subheader("Reward System Configuration")

    config = st.session_state.current_config
    config.setdefault("training", {})
    migrate_reward_schema(config, warn=True)
    reward_config = reward_view_from_config(config)

    # Reward Categories
    st.markdown("### Reward Categories")
    st.info("""
    The reward system is divided into categories that can be enabled/disabled:
    - **Terminal**: Episode-ending events (kills, destruction, boundary violations)
    - **Tactical**: Combat positioning and engagement rewards
    - **Energy**: Energy management and altitude maintenance
    - **Control**: Flight control and maneuvering
    - **Defensive**: Evasion and defensive maneuvers
    """)

    col1, col2 = st.columns(2)

    with col1:
        reward_config["enable_terminal"] = st.checkbox(
            "Enable Terminal Rewards",
            value=reward_config.get("enable_terminal", True),
            help="Rewards for kills, destruction, mission completion",
        )

        reward_config["enable_tactical"] = st.checkbox(
            "Enable Tactical Rewards",
            value=reward_config.get("enable_tactical", True),
            help="Rewards for combat positioning, tracking, engagement",
        )

        reward_config["enable_energy"] = st.checkbox(
            "Enable Energy Rewards",
            value=reward_config.get("enable_energy", True),
            help="Rewards for energy management and altitude",
        )

    with col2:
        reward_config["enable_control"] = st.checkbox(
            "Enable Control Rewards",
            value=reward_config.get("enable_control", True),
            help="Rewards for flight control and maneuvering",
        )

        reward_config["enable_defensive"] = st.checkbox(
            "Enable Defensive Rewards",
            value=reward_config.get("enable_defensive", True),
            help="Rewards for evasion and defensive actions",
        )

        reward_config["enable_line_objective"] = st.checkbox(
            "Enable Line Objective",
            value=reward_config.get("enable_line_objective", True),
            help="Rewards attackers for penetration and defenders for denial",
        )

    write_reward_view_to_config(config, reward_config)

    # Line Objective Rewards
    if reward_config["enable_line_objective"]:
        st.markdown("### Line Objective / Penetration Scenario")
        env = config.setdefault("env", {})
        scenario_cfg = env.setdefault("scenario_config", {})
        line_cfg = scenario_cfg.setdefault("line_objective", deepcopy(_LINE_OBJECTIVE_DEFAULTS))

        col1, col2 = st.columns(2)
        with col1:
            line_cfg["enabled"] = st.checkbox(
                "Line Objective Active",
                value=bool(line_cfg.get("enabled", True)),
            )
            line_cfg["attacker_team"] = st.selectbox(
                "Attacker Team:",
                ["A", "B"],
                index=0 if line_cfg.get("attacker_team", "A") == "A" else 1,
            )
            defender_default = "B" if line_cfg["attacker_team"] == "A" else "A"
            line_cfg["defender_team"] = st.selectbox(
                "Defender Team:",
                ["A", "B"],
                index=0 if line_cfg.get("defender_team", defender_default) == "A" else 1,
            )
            line_cfg["penetration_line_north_m"] = st.number_input(
                "Penetration Line North (m):",
                min_value=-500000.0,
                max_value=500000.0,
                value=float(line_cfg.get("penetration_line_north_m", 100000.0)),
            )
            axis_deg = st.number_input(
                "Penetration Axis (deg):",
                min_value=-180.0,
                max_value=180.0,
                value=float(line_cfg.get("penetration_axis_rad", 0.0)) * 180.0 / 3.141592653589793,
            )
            line_cfg["penetration_axis_rad"] = axis_deg * 3.141592653589793 / 180.0
        with col2:
            line_cfg["attacker_crossing_bonus"] = st.number_input(
                "Attacker Crossing Bonus:",
                min_value=0.0,
                max_value=1000.0,
                value=float(line_cfg.get("attacker_crossing_bonus", 80.0)),
            )
            line_cfg["attacker_team_crossing_bonus"] = st.number_input(
                "Attacker Team Crossing Bonus:",
                min_value=0.0,
                max_value=1000.0,
                value=float(line_cfg.get("attacker_team_crossing_bonus", 120.0)),
            )
            line_cfg["attacker_progress_scale"] = st.number_input(
                "Attacker Progress Scale:",
                min_value=0.0,
                max_value=1000.0,
                value=float(line_cfg.get("attacker_progress_scale", 20.0)),
            )
            line_cfg["attacker_failure_penalty"] = st.number_input(
                "Attacker Failure Penalty:",
                min_value=-1000.0,
                max_value=0.0,
                value=float(line_cfg.get("attacker_failure_penalty", -80.0)),
            )
            line_cfg["attacker_stagnation_penalty"] = st.number_input(
                "Attacker Stagnation Penalty:",
                min_value=-1000.0,
                max_value=0.0,
                value=float(line_cfg.get("attacker_stagnation_penalty", -0.01)),
            )
            line_cfg["defender_hold_reward"] = st.number_input(
                "Defender Hold Reward:",
                min_value=0.0,
                max_value=10.0,
                value=float(line_cfg.get("defender_hold_reward", 0.05)),
            )
            line_cfg["defender_penetration_penalty"] = st.number_input(
                "Defender Penetration Penalty:",
                min_value=-1000.0,
                max_value=0.0,
                value=float(line_cfg.get("defender_penetration_penalty", -120.0)),
            )
            line_cfg["defender_terminal_success_reward"] = st.number_input(
                "Defender Terminal Success Reward:",
                min_value=0.0,
                max_value=1000.0,
                value=float(line_cfg.get("defender_terminal_success_reward", 80.0)),
            )
            line_cfg["crossing_buffer_m"] = st.number_input(
                "Crossing Buffer (m):",
                min_value=0.0,
                max_value=50000.0,
                value=float(line_cfg.get("crossing_buffer_m", 2000.0)),
            )
            line_cfg["count_crossing_once"] = st.checkbox(
                "Count Crossing Once",
                value=bool(line_cfg.get("count_crossing_once", True)),
            )
            line_cfg["attacker_success_requires_crossing"] = st.checkbox(
                "Attacker Success Requires Crossing",
                value=bool(line_cfg.get("attacker_success_requires_crossing", True)),
            )
            line_cfg["defender_destroyed_counts_as_attacker_success"] = st.checkbox(
                "Defender Destroyed Counts as Attacker Success",
                value=bool(line_cfg.get("defender_destroyed_counts_as_attacker_success", False)),
            )
            line_cfg["defender_no_cross_enabled"] = st.checkbox(
                "Defender No-Cross Penalty Enabled",
                value=bool(line_cfg.get("defender_no_cross_enabled", True)),
            )
            line_cfg["defender_crossing_penalty"] = st.number_input(
                "Defender Crossing Penalty:",
                min_value=-1000.0,
                max_value=0.0,
                value=float(line_cfg.get("defender_crossing_penalty", -20.0)),
            )
            line_cfg["defender_allowed_buffer_m"] = st.number_input(
                "Defender Allowed Buffer (m):",
                min_value=0.0,
                max_value=100000.0,
                value=float(line_cfg.get("defender_allowed_buffer_m", 5000.0)),
            )

    # Terminal Rewards
    if reward_config["enable_terminal"]:
        st.markdown("### Terminal Rewards")

        with st.expander("1i  Terminal Rewards Explanation"):
            st.markdown("""
            **Terminal rewards** are given for episode-ending events:
            - **Kill Reward**: Positive reward for destroying an enemy
            - **Destruction Penalty**: Negative reward for being destroyed
            - **Boundary Violation**: Penalty for leaving the engagement area
            - **Last Team Standing**: Bonus for team survival
            """)

        col1, col2 = st.columns(2)

        with col1:
            reward_config["kill_reward"] = st.number_input(
                "Kill Reward:",
                min_value=0.0,
                max_value=1000.0,
                value=reward_config.get("kill_reward", 200.0),
                help="Reward for destroying an enemy aircraft",
            )

            reward_config["destruction_penalty"] = st.number_input(
                "Destruction Penalty:",
                min_value=-1000.0,
                max_value=0.0,
                value=reward_config.get("destruction_penalty", -200.0),
                help="Penalty for being destroyed",
            )

        with col2:
            reward_config["boundary_violation_penalty"] = st.number_input(
                "Boundary Violation Penalty:",
                min_value=-500.0,
                max_value=0.0,
                value=reward_config.get("boundary_violation_penalty", -200.0),
                help="Penalty for leaving the engagement area",
            )

            reward_config["last_team_reward"] = st.number_input(
                "Last Team Standing Reward:",
                min_value=0.0,
                max_value=200.0,
                value=reward_config.get("last_team_reward", 40.0),
                help="Bonus for team survival",
            )

    # Tactical Rewards
    if reward_config["enable_tactical"]:
        st.markdown("### Tactical Rewards")

        with st.expander("1i  Tactical Rewards Explanation"):
            st.markdown("""
            **Tactical rewards** encourage good combat positioning:
            - **Tracking Reward**: For maintaining radar lock on enemies
            - **NEZ Positioning**: For positioning within No Escape Zone
            - **Optimal Zone**: For maintaining optimal engagement range
            - **SQI Shot Bonus**: For high Shot Quality Index missile launches
            - **Early Shot Penalty**: Penalty for premature missile launches
            """)

        col1, col2 = st.columns(2)

        with col1:
            reward_config["tracking_reward_scale"] = st.number_input(
                "Tracking Reward Scale:",
                min_value=0.0,
                max_value=5.0,
                value=reward_config.get("tracking_reward_scale", 0.5),
                help="Reward scale for tracking enemies",
            )

            reward_config["nez_positioning_reward_scale"] = st.number_input(
                "NEZ Positioning Scale:",
                min_value=0.0,
                max_value=5.0,
                value=reward_config.get("nez_positioning_reward_scale", 1.5),
                help="Reward for positioning within No Escape Zone",
            )

            reward_config["sqi_shot_bonus_scale"] = st.number_input(
                "SQI Shot Bonus:",
                min_value=0.0,
                max_value=100.0,
                value=reward_config.get("sqi_shot_bonus_scale", 0.0),
                help="Disabled for training; SQI is used for metrics/evaluation only",
            )

        with col2:
            reward_config["optimal_zone_reward_scale"] = st.number_input(
                "Optimal Zone Scale:",
                min_value=0.0,
                max_value=5.0,
                value=reward_config.get("optimal_zone_reward_scale", 0.8),
                help="Reward for optimal engagement range",
            )

            reward_config["early_shot_penalty_scale"] = st.number_input(
                "Early Shot Penalty:",
                min_value=-10.0,
                max_value=0.0,
                value=reward_config.get("early_shot_penalty_scale", 0.0),
                help="Disabled; shot quality is learned from hit and kill outcomes",
            )

            reward_config["sqi_bonus_threshold"] = st.number_input(
                "SQI Bonus Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=reward_config.get("sqi_bonus_threshold", 0.55),
                help="SQI threshold for shot bonus",
            )

    # Energy Rewards
    if reward_config["enable_energy"]:
        st.markdown("### Energy Management Rewards")

        with st.expander("1i  Energy Rewards Explanation"):
            st.markdown("""
            **Energy rewards** encourage proper energy management:
            - **Energy Reward**: For maintaining high energy state
            - **Low Altitude Penalty**: For flying too low
            - **Altitude Loss Penalty**: For excessive altitude loss
            """)

        col1, col2 = st.columns(2)

        with col1:
            reward_config["energy_reward_scale"] = st.number_input(
                "Energy Reward Scale:",
                min_value=0.0,
                max_value=5.0,
                value=reward_config.get("energy_reward_scale", 0.3),
                help="Reward scale for energy state",
            )

            reward_config["low_altitude_penalty_scale"] = st.number_input(
                "Low Altitude Penalty:",
                min_value=-10.0,
                max_value=0.0,
                value=reward_config.get("low_altitude_penalty_scale", -2.0),
                help="Penalty for flying below minimum altitude",
            )

        with col2:
            reward_config["low_altitude_threshold_m"] = st.number_input(
                "Low Altitude Threshold (m):",
                min_value=1000.0,
                max_value=10000.0,
                value=reward_config.get("low_altitude_threshold_m", 5000.0),
                help="Altitude below which penalty applies",
            )

            reward_config["altitude_loss_penalty_scale"] = st.number_input(
                "Altitude Loss Penalty:",
                min_value=-1.0,
                max_value=0.0,
                value=reward_config.get("altitude_loss_penalty_scale", -0.1),
                help="Penalty per meter of altitude lost",
            )

    # Control Rewards
    if reward_config["enable_control"]:
        st.markdown("### Control Rewards")

        with st.expander("1i  Control Rewards Explanation"):
            st.markdown("""
            **Control rewards** encourage smooth flight and proper maneuvering:
            - **Lift Balance Penalty**: For excessive load factor
            - **Heading Alignment**: For proper heading towards targets
            - **Passivity Penalty**: Discourages passive behavior
            """)

        col1, col2 = st.columns(2)

        with col1:
            reward_config["lift_balance_penalty_scale"] = st.number_input(
                "Lift Balance Penalty:",
                min_value=-5.0,
                max_value=0.0,
                value=reward_config.get("lift_balance_penalty_scale", -0.8),
                help="Penalty for excessive load factor",
            )

            reward_config["heading_alignment_reward_scale"] = st.number_input(
                "Heading Alignment Scale:",
                min_value=0.0,
                max_value=2.0,
                value=reward_config.get("heading_alignment_reward_scale", 0.2),
                help="Reward for proper heading alignment",
            )

        with col2:
            reward_config["passivity_penalty_scale"] = st.number_input(
                "Passivity Penalty:",
                min_value=-5.0,
                max_value=0.0,
                value=reward_config.get("passivity_penalty_scale", -1.0),
                help="Penalty for passive behavior",
            )

    # Defensive Rewards
    if reward_config["enable_defensive"]:
        st.markdown("### Defensive Rewards")

        with st.expander("1i  Defensive Rewards Explanation"):
            st.markdown("""
            **Defensive rewards** encourage survival and evasion:
            - **Evasion Reward**: For successful defensive maneuvers
            - **Boundary Penalties**: Progressive penalties near boundaries
            """)

        col1, col2 = st.columns(2)

        with col1:
            reward_config["evasion_reward_scale"] = st.number_input(
                "Evasion Reward Scale:",
                min_value=0.0,
                max_value=5.0,
                value=reward_config.get("evasion_reward_scale", 0.8),
                help="Reward for successful evasion",
            )

        with col2:
            reward_config["boundary_progressive_penalty_scale"] = st.number_input(
                "Progressive Boundary Penalty:",
                min_value=-20.0,
                max_value=0.0,
                value=reward_config.get("boundary_progressive_penalty_scale", -10.0),
                help="Progressive penalty near boundaries",
            )

    st.markdown("### Reward Normalization")
    norm_cfg = config.setdefault("env", {}).setdefault(
        "reward_normalization", {"enabled": False, "clip_reward": 10.0}
    )
    col1, col2 = st.columns(2)
    with col1:
        norm_cfg["enabled"] = st.checkbox(
            "Enable Reward Normalization",
            value=bool(norm_cfg.get("enabled", False)),
        )
    with col2:
        norm_cfg["clip_reward"] = st.number_input(
            "Reward Clip Value:",
            min_value=0.1,
            max_value=1000.0,
            value=float(norm_cfg.get("clip_reward", 10.0)),
        )

    write_reward_view_to_config(config, reward_config)


def render_config_comparison():
    """Render configuration comparison interface."""
    st.subheader("Configuration Comparison")

    # Load configurations for comparison
    existing_configs = st.session_state.config_manager.get_existing_configs()
    config_names = [Path(f).name for f in existing_configs]

    if len(config_names) < 2:
        st.warning("You need at least 2 saved configurations to compare.")
        return

    # Select configurations to compare
    col1, col2 = st.columns(2)

    with col1:
        config1_name = st.selectbox("Configuration 1:", config_names, key="comp_config1")

    with col2:
        config2_name = st.selectbox(
            "Configuration 2:",
            [name for name in config_names if name != config1_name],
            key="comp_config2",
        )

    if st.button("Compare Configurations"):
        # Load both configurations
        config1_path = next(f for f in existing_configs if Path(f).name == config1_name)
        config2_path = next(f for f in existing_configs if Path(f).name == config2_name)

        config1 = st.session_state.config_manager.load_config(config1_path)
        config2 = st.session_state.config_manager.load_config(config2_path)

        # Compare configurations
        st.markdown("### Configuration Differences")

        differences = find_config_differences(config1, config2)

        if not differences:
            st.success("Configurations are identical.")
        else:
            for section, diffs in differences.items():
                st.markdown(f"#### {section.title()}")

                for key, values in diffs.items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{config1_name}**")
                        st.code(f"{key}: {values['config1']}")
                    with col2:
                        st.markdown(f"**{config2_name}**")
                        st.code(f"{key}: {values['config2']}")

                st.markdown("---")


def find_config_differences(
    config1: dict[str, Any], config2: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Find differences between two configurations."""
    differences = {}

    def compare_section(section_name: str, section1: Any, section2: Any, path: str = ""):
        if section_name not in differences:
            differences[section_name] = {}

        if isinstance(section1, dict) and isinstance(section2, dict):
            # Compare dictionaries recursively
            all_keys = set(section1.keys()) | set(section2.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                val1 = section1.get(key, "NOT_SET")
                val2 = section2.get(key, "NOT_SET")

                if isinstance(val1, dict) and isinstance(val2, dict):
                    compare_section(section_name, val1, val2, key_path)
                elif val1 != val2:
                    differences[section_name][key_path] = {"config1": val1, "config2": val2}
        elif section1 != section2:
            differences[section_name][path or section_name] = {
                "config1": section1,
                "config2": section2,
            }

    # Compare main sections
    for section in ["env", "training", "model", "logging"]:
        if section in config1 or section in config2:
            compare_section(section, config1.get(section, {}), config2.get(section, {}))

    # Remove empty sections
    differences = {k: v for k, v in differences.items() if v}

    return differences


if __name__ == "__main__":
    config_builder()
