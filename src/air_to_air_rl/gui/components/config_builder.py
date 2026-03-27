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


class ConfigManager:
    """Manages configuration creation, editing and storage."""

    def __init__(self):
        from air_to_air_rl.core.paths import project_root, rl_configs_root

        self.config_dir = project_root() / "configs" / "training"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._package_config_dir = rl_configs_root()

    def save_config(self, config: dict[str, Any], name: str) -> str:
        """Save configuration to configs/training/ directory."""
        filename = f"{name}.yaml"
        filepath = self.config_dir / filename

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
            f"Invalid config name **{name!r}** — use only letters, digits, hyphens, and underscores."
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
            "Set it to `simplified` or `bvr` in **General Settings → Environment Type** "
            "so the dashboard shows this config in the correct training tab."
        )

    # --- Missile lists ---
    missile_cfg = config.get("env", {}).get("missile_config", {})
    if not missile_cfg.get("agent_missiles"):
        errors.append("Agent missile list is empty — add at least one missile type.")
    if not missile_cfg.get("opponent_missiles"):
        errors.append("Opponent missile list is empty — add at least one missile type.")

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
                f"`rollout_fragment_length` ({rfl}) × `num_env_runners` ({num_runners}) "
                f"= **{collected}** but `train_batch_size` = **{batch_size}**. "
                f"These must match. Set `rollout_fragment_length` to **auto** "
                f"(recommended) or to **{batch_size // max(num_runners, 1)}**."
            )

    return errors, warnings


def config_builder():
    """Main training configuration builder interface."""
    st.header("⚙️ Training Configuration Builder")

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

        if st.button("💾 Save Configuration", use_container_width=True, disabled=save_disabled):
            filepath = st.session_state.config_manager.save_config(
                st.session_state.current_config,
                config_name,
            )
            st.success(f"Saved: {filepath}")

    # Main configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 General Settings", "🚀 Training", "🧠 Networks", "📊 Compare Configs"]
    )

    with tab1:
        render_general_settings()

    with tab2:
        render_training_settings()

    with tab3:
        render_network_settings()

    with tab4:
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
        },
        "model": {
            "model_config": {
                "action_dim": 4,
                "action_mask_key": "action_mask",
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

    # simulation_config may be a string in older YAML files — normalise to dict
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

    # AWACS Configuration (BVR only — Simplified env has no AWACS)
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
                    "Total batch ≈ num_env_runners × rollout_fragment_length."
                ),
            )

        config["training"]["sgd_minibatch_size"] = st.number_input(
            "SGD Minibatch Size:",
            min_value=8,
            max_value=1024,
            value=config["training"].get("sgd_minibatch_size", 128),
            help="Mini-batch size within each SGD epoch.",
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
            "PPO Clip Parameter (ε):",
            min_value=0.01,
            max_value=1.0,
            value=config["training"].get("clip_param", 0.2),
            help="PPO surrogate clipping parameter ε",
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
            "Discount Factor (γ):",
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


def render_network_settings():
    """Render network architecture configuration page."""
    st.subheader("Network Architecture Configuration")

    config = st.session_state.current_config

    # Action Space / Neural Wrapper Configuration
    st.markdown("### Action Space Configuration")

    is_simplified = st.session_state.get("training_env_type", "Standard BVR") == "Simplified"

    if is_simplified:
        st.info(
            "🚀 **Simplified Environment**: Fixed 4D action space "
            "(energy, lift_phi, lift_N, missile_trigger). Neural wrapper not applicable."
        )
        config["model"]["model_config"]["action_dim"] = 4
        config["model"]["model_config"]["wrapped_action_dim"] = 4
        config["model"]["model_config"]["full_action_dim"] = 4
        config["model"]["model_config"]["active_indices"] = [0, 1, 2, 3]
    else:
        # BVR: agent can control 4, 5, 6 or all 10 dimensions
        _ACTION_DIM_OPTIONS = {
            4: "4D — Flight + Missile Trigger  (target selection, gun & CM automated)",
            5: "5D — + Target Selection  (gun & CM automated)",
            6: "6D — + Gun Trigger  (CM automated)",
            10: "10D — + Countermeasures  (full control, no neural wrapper)",
        }
        _dim_values = list(_ACTION_DIM_OPTIONS.keys())
        _dim_labels = list(_ACTION_DIM_OPTIONS.values())

        current_dim = config["model"]["model_config"].get("action_dim", 4)
        if current_dim not in _dim_values:
            current_dim = 4

        selected_label = st.selectbox(
            "Action Dimensions (Neural Wrapper Level):",
            _dim_labels,
            index=_dim_values.index(current_dim),
            help=(
                "How many actions the agent controls directly. "
                "The remaining actions are handled automatically by the neural wrapper.\n\n"
                "4D: most automated   |   10D: full agent control"
            ),
        )
        selected_dim = _dim_values[_dim_labels.index(selected_label)]

        use_wrapper = selected_dim < 10
        config["model"]["model_config"]["action_dim"] = selected_dim
        config["model"]["model_config"]["wrapped_action_dim"] = selected_dim
        config["model"]["model_config"]["full_action_dim"] = 10

        if use_wrapper:
            st.info(
                f"🤖 **Neural Wrapper Active ({selected_dim}D)**: {_ACTION_DIM_OPTIONS[selected_dim]}"
            )
            automation_levels = ["defensive", "balanced", "aggressive"]
            st.selectbox(
                "Automation Level:",
                automation_levels,
                index=automation_levels.index(config["model"].get("automation_level", "balanced")),
                help="Behaviour of the automated components (target selection, gun, countermeasures)",
            )
        else:
            st.warning(
                "⚠️ **Full Control Mode (10D)**: Network outputs all 10 actions. "
                "No automation — most complex, maximum agent control."
            )

    st.info(
        "Ray RLlib's default PPO network is used. No custom architecture configuration is required."
    )


def render_config_comparison():
    """Render configuration comparison interface."""
    st.subheader("Configuration Comparison")

    # Load configurations for comparison
    existing_configs = st.session_state.config_manager.get_existing_configs()
    config_names = [Path(f).name for f in existing_configs]

    if len(config_names) < 2:
        st.warning("⚠️ You need at least 2 saved configurations to compare.")
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

    if st.button("🔍 Compare Configurations"):
        # Load both configurations
        config1_path = next(f for f in existing_configs if Path(f).name == config1_name)
        config2_path = next(f for f in existing_configs if Path(f).name == config2_name)

        config1 = st.session_state.config_manager.load_config(config1_path)
        config2 = st.session_state.config_manager.load_config(config2_path)

        # Compare configurations
        st.markdown("### Configuration Differences")

        differences = find_config_differences(config1, config2)

        if not differences:
            st.success("✅ Configurations are identical!")
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
