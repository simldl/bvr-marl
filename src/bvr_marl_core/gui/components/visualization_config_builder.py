"""
Visualization Configuration Builder

Interface for creating, editing and tracking visualization configurations.
Saves configurations to visualization/configs/ directory.
"""

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from bvr_marl_core.visualization.scenario_overlays import (
    get_visualization_scenario,
    list_visualization_scenarios,
)


class VizConfigManager:
    """Manages visualization configuration creation, editing and storage."""

    def __init__(self):
        from bvr_marl_core.utils.paths import (
            core_project_root,
            core_root,
            rl_configs_root,
        )

        self.config_dir = core_project_root() / "configs" / "visualization"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._package_viz_dir = core_root() / "visualization" / "configs"
        self.project_training_configs_dir = core_project_root() / "configs" / "training"
        self.training_configs_dir = rl_configs_root()

    def save_config(self, config: dict[str, Any], name: str) -> str:
        """Save configuration to visualization/configs directory."""
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
        """Get list of existing visualization configuration files."""
        configs = []

        # Add visualization configs from visualization/configs
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                configs.append(str(config_file))

        return sorted(configs)

    def get_training_configs(self) -> list[str]:
        """Get list of available training configurations."""
        configs = []
        seen = set()

        for config_dir in [self.project_training_configs_dir, self.training_configs_dir]:
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    if config_file.name not in seen:
                        seen.add(config_file.name)
                        configs.append(str(config_file))

        return sorted(configs)

    def get_checkpoint_files(self) -> list[str]:
        """Get list of available model checkpoint files."""
        from bvr_marl_core.services.visualization import find_checkpoint_files

        return find_checkpoint_files()


def visualization_config_builder():
    """Main visualization configuration builder interface."""
    st.header("Visualization Configuration Builder")
    st.caption("Create, compare, and save visualization presets for the live 2D tooling.")

    st.info("""
    **Visualization Configuration Builder**
    Create and edit visualization configurations for the live 2D visualization system.
    Configurations are saved to: `visualization/configs/`
    """)

    # Initialize config manager
    if "viz_config_manager" not in st.session_state:
        st.session_state.viz_config_manager = VizConfigManager()

    # Initialize current config in session state
    if "current_viz_config" not in st.session_state:
        st.session_state.current_viz_config = create_default_viz_config()

    # Sidebar for config management
    with st.sidebar:
        st.subheader("Configuration Management")

        # Load existing config
        existing_configs = st.session_state.viz_config_manager.get_existing_configs()
        config_names = ["Create New"] + [Path(f).name for f in existing_configs]

        selected_config = st.selectbox(
            "Load Configuration:",
            config_names,
            help="Load an existing visualization configuration or create a new one",
        )

        if selected_config != "Create New":
            config_path = next(f for f in existing_configs if Path(f).name == selected_config)
            if st.button("Load Selected Config"):
                st.session_state.current_viz_config = (
                    st.session_state.viz_config_manager.load_config(config_path)
                )
                st.success(f"Loaded: {selected_config}")
                st.rerun()

        # Save current config
        st.markdown("---")
        config_name = st.text_input(
            "Configuration Name:",
            value="",
            placeholder="my_viz_config",
            help="Name for this visualization configuration",
        )

        if st.button("Save Configuration", width="stretch"):
            if config_name:
                filepath = st.session_state.viz_config_manager.save_config(
                    st.session_state.current_viz_config, config_name
                )
                st.success(f"Saved: {filepath}")
            else:
                st.error("Please enter a configuration name")

    # Main configuration tabs
    tab1, tab2, tab3 = st.tabs(
        ["Model & Training Config", "Visualization Settings", "Compare Configs"]
    )

    with tab1:
        render_model_and_training_settings()

    with tab2:
        render_visualization_settings()

    with tab3:
        render_viz_config_comparison()


def create_default_viz_config() -> dict[str, Any]:
    """Create a default visualization configuration template."""
    return {
        "checkpoint_path": None,
        "model_config_path": None,
        "train_config_path": "configs/training/basic.yaml",
        "env_type": "bvr",
        "scenario": "default",
        "overlay": {
            "show_line": False,
            "line_east_km": 0.0,
            "map_extents_mode": "auto",
        },
        "visualization": {
            "frames": 100,
            "interval": 100,
            "real_time_speed": False,
            "save_video": False,
            "save_rewards": False,
            "symbol_mode": "flying_objects",
            "dpi": 200,
            "symbol_scale": 2.0,
        },
    }


def render_model_and_training_settings():
    """Render model and training configuration selection."""
    st.subheader("Model & Training Configuration")

    config = st.session_state.current_viz_config

    # Model Selection
    st.markdown("### Model Selection")

    use_trained_model = st.checkbox(
        "Use Trained Model",
        value=config.get("checkpoint_path") is not None,
        help="Use a trained model checkpoint. If disabled, random actions will be used.",
    )

    if use_trained_model:
        # Checkpoint selection
        checkpoint_files = st.session_state.viz_config_manager.get_checkpoint_files()

        if checkpoint_files:
            current_checkpoint = config.get("checkpoint_path", "")

            # Try to find current checkpoint in list
            checkpoint_index = 0
            if current_checkpoint and current_checkpoint in checkpoint_files:
                checkpoint_index = checkpoint_files.index(current_checkpoint)

            selected_checkpoint = st.selectbox(
                "Model Checkpoint:",
                ["None"] + checkpoint_files,
                index=checkpoint_index + 1 if current_checkpoint else 0,
                help="Select a trained model checkpoint file",
            )

            if selected_checkpoint != "None":
                config["checkpoint_path"] = selected_checkpoint
                st.success(f"✓ Using checkpoint: {selected_checkpoint}")
            else:
                config["checkpoint_path"] = None
        else:
            st.warning(
                "No checkpoint files found. Please train a model first or place checkpoint files in models/, outputs/, or checkpoints/ directories."
            )
            config["checkpoint_path"] = None

        # Model config override (optional)
        st.markdown("#### Model Config Override (Optional)")
        custom_model_config = st.text_input(
            "Model Config Path:",
            value=config.get("model_config_path", "") or "",
            placeholder="Leave empty to use default model config",
            help="Optional path to override model configuration",
        )

        config["model_config_path"] = custom_model_config if custom_model_config else None
    else:
        config["checkpoint_path"] = None
        config["model_config_path"] = None
        st.info(
            "**Random Actions Mode**: The visualization will use random actions instead of a trained model."
        )

    # Training Configuration
    st.markdown("### Training Configuration")
    st.info(
        "Select the training configuration that matches your model (or desired environment setup for random actions)."
    )

    training_configs = st.session_state.viz_config_manager.get_training_configs()
    training_config_names = [Path(f).name for f in training_configs]

    if training_config_names:
        current_train_config = config.get("train_config_path", "configs/training/basic.yaml")
        current_name = Path(current_train_config).name if current_train_config else ""

        # Try to find current config in list
        train_config_index = 0
        if current_name and current_name in training_config_names:
            train_config_index = training_config_names.index(current_name)

        selected_train_config_name = st.selectbox(
            "Training Configuration:",
            training_config_names,
            index=train_config_index,
            help="Training configuration that matches your model",
        )

        # Find the full path
        selected_train_config_path = next(
            f for f in training_configs if Path(f).name == selected_train_config_name
        )
        config["train_config_path"] = selected_train_config_path

        st.success(f"✓ Using training config: {selected_train_config_name}")

        # Display config info
        with st.expander("Training Config Summary"):
            try:
                train_cfg = st.session_state.viz_config_manager.load_config(
                    selected_train_config_path
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Environment Settings:**")
                    env_cfg = train_cfg.get("env", {})
                    st.write(f"- Agents per side: {env_cfg.get('num_agents_per_side', 'N/A')}")
                    st.write(f"- Map size: {env_cfg.get('map_size', 'N/A')} km")
                    st.write(f"- Max steps: {env_cfg.get('max_steps', 'N/A')}")

                with col2:
                    st.markdown("**Model Settings:**")
                    model_cfg = train_cfg.get("model", {}).get("model_config", {})
                    st.write(
                        f"- Neural wrapper: {train_cfg.get('model', {}).get('use_neural_wrapper', 'N/A')}"
                    )
                    st.write(f"- Hidden dim: {model_cfg.get('hidden_dim', 'N/A')}")
                    st.write(f"- Hidden layers: {model_cfg.get('num_hidden_layers', 'N/A')}")

            except Exception as e:
                st.error(f"Error reading training config: {e}")
    else:
        st.warning("No training configurations found in `configs/training/`.")
        custom_train_config = st.text_input(
            "Training Config Path:",
            value=config.get("train_config_path", ""),
            placeholder="configs/training/basic.yaml",
        )
        config["train_config_path"] = custom_train_config

    # Environment Type
    st.markdown("### Environment Type")
    env_types = ["simplified", "bvr"]
    current_env_type = config.get("env_type", "simplified")

    config["env_type"] = st.selectbox(
        "Environment Type:",
        env_types,
        index=env_types.index(current_env_type) if current_env_type in env_types else 0,
        help="Type of environment to visualize. Must match your training configuration!",
    )

    if config["env_type"] == "simplified":
        st.info(
            "**Simplified Environment**: Truth position observations, 4D action space, and no radar simulation."
        )
    else:
        st.info(
            "**BVR Environment**: Full radar/AWACS simulation with a complex observation space."
        )


def render_visualization_settings():
    """Render visualization display and recording settings."""
    st.subheader("Visualization Settings")

    config = st.session_state.current_viz_config
    viz_config = config["visualization"]
    overlay_config = config.setdefault(
        "overlay",
        {
            "show_line": False,
            "line_east_km": 0.0,
            "map_extents_mode": "auto",
        },
    )

    # Scenario overlay
    st.markdown("### Scenario Overlay")
    scenario_options = list_visualization_scenarios()
    scenario_labels = [scenario.label for scenario in scenario_options]
    scenario_label_by_key = {scenario.key: scenario.label for scenario in scenario_options}
    current_scenario = get_visualization_scenario(config.get("scenario"))
    selected_scenario_label = st.selectbox(
        "Scenario:",
        options=scenario_labels,
        index=scenario_labels.index(
            scenario_label_by_key.get(current_scenario.key, scenario_labels[0])
        ),
        help="Choose extra scenario annotations for the 2D map.",
    )
    selected_scenario = get_visualization_scenario(selected_scenario_label)
    config["scenario"] = selected_scenario.key
    st.info(selected_scenario.description)

    overlay_config["show_line"] = st.checkbox(
        "Show line of engagement on any scenario",
        value=overlay_config.get("show_line", False),
        help="Adds a movable LOE to scenarios that do not define one themselves.",
    )
    line_overlay_available = overlay_config["show_line"] or selected_scenario.has_line_of_engagement
    overlay_config["line_east_km"] = st.slider(
        "Line X Offset (km):",
        min_value=-300,
        max_value=300,
        value=int(round(float(overlay_config.get("line_east_km", 0.0)))),
        step=5,
        disabled=not line_overlay_available,
        help="Move the LOE west/east along the x-axis. Negative is west, positive is east.",
    )
    overlay_modes = ["auto", "combat", "full"]
    overlay_labels = {
        "auto": "Auto",
        "combat": "Combat Zone",
        "full": "Full Scenario",
    }
    current_map_mode = str(overlay_config.get("map_extents_mode", "auto")).lower()
    if current_map_mode not in overlay_modes:
        current_map_mode = "auto"
    selected_map_mode_label = st.selectbox(
        "Displayed Map Area:",
        options=[overlay_labels[m] for m in overlay_modes],
        index=overlay_modes.index(current_map_mode),
        help=(
            "Auto follows the scenario default. Full Scenario shows the full "
            "display bounds, including side zones such as AWACS areas."
        ),
    )
    overlay_config["map_extents_mode"] = next(
        mode for mode in overlay_modes if overlay_labels[mode] == selected_map_mode_label
    )

    # Simulation Settings
    st.markdown("### Simulation Settings")
    col1, col2 = st.columns(2)

    with col1:
        viz_config["frames"] = st.number_input(
            "Number of Frames:",
            min_value=10,
            max_value=2000,
            value=viz_config.get("frames", 100),
            help="Total number of simulation frames to run",
        )

        viz_config["interval"] = st.number_input(
            "Frame Interval (ms):",
            min_value=10,
            max_value=2000,
            value=viz_config.get("interval", 100),
            help="Milliseconds between frames (lower = faster playback)",
        )

    with col2:
        viz_config["real_time_speed"] = st.checkbox(
            "Real-time Playback",
            value=viz_config.get("real_time_speed", False),
            help="Sync playback speed to simulation tick rate for realistic timing",
        )

        if viz_config["real_time_speed"]:
            st.info("🕐 Real-time mode: Playback speed matches simulation timing")
        else:
            st.info(f"⚡ Fixed interval: {viz_config['interval']}ms between frames")

    # Visual Settings
    st.markdown("### Visual Settings")
    col1, col2 = st.columns(2)

    with col1:
        symbol_modes = ["flying_objects", "nato", "procedural"]
        viz_config["symbol_mode"] = st.selectbox(
            "Symbol Mode:",
            symbol_modes,
            index=symbol_modes.index(viz_config.get("symbol_mode", "flying_objects")),
            help="Visual representation of aircraft and missiles",
        )

        if viz_config["symbol_mode"] == "flying_objects":
            st.info("**Flying Objects**: Aircraft-specific silhouettes with blue/red coloring.")
        elif viz_config["symbol_mode"] == "nato":
            st.info("🎖️ **NATO Symbols**: MIL-STD-2525C military symbology")
        else:
            st.info("🎨 **Procedural**: Simple geometric shapes")

        viz_config["dpi"] = st.selectbox(
            "Render Quality (DPI):",
            [200, 300, 400],
            index=[200, 300, 400].index(viz_config.get("dpi", 200)),
            help="Rendering resolution - higher values produce better quality but slower rendering",
        )

    with col2:
        viz_config["symbol_scale"] = st.slider(
            "Symbol Scale:",
            min_value=0.5,
            max_value=5.0,
            value=viz_config.get("symbol_scale", 2.0),
            step=0.1,
            help="Size multiplier for aircraft/missile symbols",
        )

        st.markdown("**Quality vs Speed:**")
        if viz_config["dpi"] == 200:
            st.success("Fast rendering with good quality.")
        elif viz_config["dpi"] == 300:
            st.warning("⚖️ Balanced rendering speed/quality")
        else:
            st.error("🐌 Slow rendering, best quality")

    # Recording Settings
    st.markdown("### Recording & Export Settings")
    col1, col2 = st.columns(2)

    with col1:
        viz_config["save_video"] = st.checkbox(
            "Save Video",
            value=viz_config.get("save_video", False),
            help="Save the visualization as a video file",
        )

        if viz_config["save_video"]:
            st.info("Video will be saved to the visualization output directory.")

    with col2:
        viz_config["save_rewards"] = st.checkbox(
            "Save Reward Logs",
            value=viz_config.get("save_rewards", False),
            help="Export detailed reward information as CSV and JSON files",
        )

        if viz_config["save_rewards"]:
            st.info("Reward data will be saved for analysis.")

    # Preview Settings
    st.markdown("### Configuration Preview")
    with st.expander("Current Configuration"):
        st.json(config)


def render_viz_config_comparison():
    """Render visualization configuration comparison interface."""
    st.subheader("Configuration Comparison")

    # Load configurations for comparison
    existing_configs = st.session_state.viz_config_manager.get_existing_configs()
    config_names = [Path(f).name for f in existing_configs]

    if len(config_names) < 2:
        st.warning("You need at least 2 saved visualization configurations to compare.")
        return

    # Select configurations to compare
    col1, col2 = st.columns(2)

    with col1:
        config1_name = st.selectbox("Configuration 1:", config_names, key="viz_comp_config1")

    with col2:
        config2_name = st.selectbox(
            "Configuration 2:",
            [name for name in config_names if name != config1_name],
            key="viz_comp_config2",
        )

    if st.button("Compare Configurations"):
        # Load both configurations
        config1_path = next(f for f in existing_configs if Path(f).name == config1_name)
        config2_path = next(f for f in existing_configs if Path(f).name == config2_name)

        config1 = st.session_state.viz_config_manager.load_config(config1_path)
        config2 = st.session_state.viz_config_manager.load_config(config2_path)

        # Compare configurations
        st.markdown("### Configuration Differences")

        differences = find_viz_config_differences(config1, config2)

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


def find_viz_config_differences(
    config1: dict[str, Any], config2: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Find differences between two visualization configurations."""
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
    for key in config1.keys() | config2.keys():
        if key == "visualization":
            compare_section("visualization", config1.get(key, {}), config2.get(key, {}))
        else:
            compare_section("general", config1.get(key), config2.get(key), key)

    # Remove empty sections
    differences = {k: v for k, v in differences.items() if v}

    return differences


if __name__ == "__main__":
    visualization_config_builder()
