"""
Training Dashboard

Interface for launching and monitoring training jobs.
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml

from bvr_marl_core.gui.components.checkpoint_picker import (
    checkpoint_picker,
    find_train_config_for_checkpoint,
)
from bvr_marl_core.gui.components.config_validator import (
    render_validation_result,
    validate_training_config,
)
from bvr_marl_core.gui.components.output_paths import (
    display_compact_output_paths,
    display_recent_outputs,
)
from bvr_marl_core.gui.components.training_monitor import TrainingMonitor, render_training_monitor
from bvr_marl_core.services.training import (
    build_simple_training_cmd,
    build_training_cmd,
    launch_background_process,
    list_training_configs,
)
from bvr_marl_core.utils.config_loader import load_gui_config
from bvr_marl_core.utils.paths import core_project_root as project_root


def find_config_files():
    """Find available training configuration files (all configs)."""
    return list_training_configs()


def _read_training_mode(filename: str) -> str | None:
    """Return the training_mode value from a config file, or None on failure."""
    from bvr_marl_core.utils.paths import rl_configs_root

    raw_path = Path(filename)
    if raw_path.is_absolute():
        candidates = [raw_path]
    elif "/" in filename or "\\" in filename:
        candidates = [project_root() / raw_path]
    else:
        candidates = [
            project_root() / "configs" / "training" / filename,
            rl_configs_root() / filename,
        ]
    for cfg_path in candidates:
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            return data.get("training_mode")
        except (OSError, yaml.YAMLError):
            pass
    return None


def find_bvr_config_files() -> list[str]:
    """Return config filenames that are NOT tagged as simplified.

    A config is considered simplified when:
    - its filename contains "simplified" or "simple", OR
    - its ``training_mode`` key equals ``"simplified"``.

    Everything else (including untagged configs) is treated as BVR.
    """
    all_configs = list_training_configs()
    bvr = []
    for filename in all_configs:
        name_lower = filename.lower()
        if "simplified" in name_lower or "simple" in name_lower:
            continue
        mode = _read_training_mode(filename)
        if mode == "simplified":
            continue
        bvr.append(filename)
    return bvr


def find_checkpoint_files():
    """Find available checkpoint files under the project root."""
    from bvr_marl_core.services.visualization import find_checkpoint_files as _svc_find

    return _svc_find()


def training_dashboard():
    """Render the training dashboard."""
    gui_cfg = load_gui_config()
    default_config = gui_cfg.get("training", {}).get("default_config", "basic")
    default_config_name = (
        default_config if str(default_config).endswith(".yaml") else f"{default_config}.yaml"
    )

    st.header("Training Dashboard")
    st.caption("Launch, monitor, and manage reinforcement-learning training workflows.")

    _render_training_config_info()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Training Monitor", "Standard Training", "Simplified Training", "Batch Operations"]
    )

    with tab1:
        render_training_monitor()

    with tab2:
        _render_standard_training_tab(default_config_name)

    with tab3:
        _render_simplified_training_tab()

    with tab4:
        _render_batch_operations_tab(default_config_name)

    _render_training_output_info()


def _render_training_config_info() -> None:
    st.info("""
    **Training configuration files**
    Configs are loaded from `configs/training/`, standard campaign stages, or package defaults.
    This ensures consistent paths and prevents config file confusion.
    """)


def _render_standard_training_tab(default_config_name: str) -> None:
    st.subheader("Standard BVR Training")

    col_config, col_actions = st.columns([2, 1])
    with col_config:
        selected_config = _render_standard_config_selector(default_config_name)
        custom_config = _render_custom_config_path()
        seed, steps, model_name = _render_standard_training_parameters()
        resume_checkpoint = _render_resume_checkpoint_picker()

    with col_actions:
        final_config = custom_config if custom_config else selected_config
        _render_standard_training_controls(
            final_config=final_config,
            seed=seed,
            steps=steps,
            model_name=model_name,
            resume_checkpoint=resume_checkpoint,
        )


def _render_standard_config_selector(default_config_name: str) -> str:
    st.markdown("#### Training Configuration")
    st.info(
        "Configs are loaded from: `configs/training/`, standard campaign stages, "
        "or package defaults"
    )

    config_files = find_bvr_config_files()
    if not config_files:
        st.warning("No BVR configuration files found in `configs/training/`.")
        st.markdown(
            "**Tip:** Create a config in the **Config Builder** with "
            "Environment Type set to **Standard BVR**."
        )
        return ""

    config_options = [""] + config_files
    default_index = (
        config_options.index(default_config_name) if default_config_name in config_options else 0
    )
    selected_config = st.selectbox(
        "Select training configuration:",
        options=config_options,
        index=default_index,
        help=(
            "Only Standard BVR configs are shown here. "
            "Simplified configs are available under the Simplified Training tab."
        ),
    )
    if selected_config:
        st.success(f"✓ Using: `{selected_config}`")
    return selected_config


def _render_custom_config_path() -> str:
    with st.expander("Advanced: Custom Config Path"):
        custom_config = st.text_input(
            "Custom config file path:",
            placeholder="Leave empty to use selection above",
            help="Only use this if you have configs outside the standard directory",
        )
        if custom_config:
            st.warning(
                "Using custom config path - ensure the file exists and is properly formatted"
            )
    return custom_config


def _render_standard_training_parameters() -> tuple[int, int, str]:
    st.markdown("#### Training Parameters")

    col_seed, col_steps = st.columns(2)
    with col_seed:
        seed = st.number_input("Random seed", value=42, min_value=0)
    with col_steps:
        steps = st.number_input("Training steps", value=1000, min_value=1)

    model_name = st.text_input(
        "Model name:", value="gui_training_model", help="Name for the trained model"
    )
    return int(seed), int(steps), model_name


def _render_resume_checkpoint_picker() -> str | None:
    st.markdown("#### Resume Training (Optional)")
    resume_checkpoint = checkpoint_picker(
        key_prefix="train_resume",
        allow_none=True,
        none_label="Start fresh",
        label="Resume from checkpoint",
    )

    if resume_checkpoint:
        resume_cfg = find_train_config_for_checkpoint(resume_checkpoint)
        if resume_cfg:
            st.info(
                f"This checkpoint was trained with `{resume_cfg}`. "
                "Consider selecting the same config above to keep a consistent scenario."
            )
    return resume_checkpoint


def _render_standard_training_controls(
    *,
    final_config: str,
    seed: int,
    steps: int,
    model_name: str,
    resume_checkpoint: str | None,
) -> None:
    st.markdown("#### Launch Training")

    if st.button("Validate Architecture", width="stretch"):
        if not final_config:
            st.error("Please select a configuration file first")
        else:
            with st.spinner("Running architecture check…"):
                result = validate_training_config(final_config, training_mode="bvr")
            render_validation_result(result)

    if st.button("Start Training", type="primary", width="stretch"):
        if not final_config:
            st.error("Please select or enter a configuration file")
        else:
            launch_training(
                config=final_config,
                seed=seed,
                steps=steps,
                model_name=model_name,
                resume_checkpoint=resume_checkpoint,
            )

    st.markdown("---")
    _render_training_quick_actions()


def _render_training_quick_actions() -> None:
    st.markdown("#### Quick Actions")

    if st.button("Open Models Folder", width="stretch"):
        _open_models_folder()

    if st.button("Open TensorBoard", width="stretch"):
        st.info("TensorBoard integration coming soon!")


def _open_models_folder() -> None:
    models_dir = project_root() / "models"
    if not models_dir.exists():
        st.warning("Models directory not found (models/)")
        return

    if os.name == "nt":
        os.startfile(models_dir)
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(models_dir)])


def _render_simplified_training_tab() -> None:
    st.subheader("Simplified Environment Training")
    st.info("""
    **Simplified Training Features:**
    - Truth position observations (no radar simulation)
    - 4D action space (energy, lift_phi, lift_N, missile_trigger)
    - Faster training for algorithm development
    """)

    col_config, col_actions = st.columns([2, 1])
    with col_config:
        simple_config = _render_simplified_config_selector()
        simple_seed, simple_steps, simple_model_name = _render_simplified_parameters()

    with col_actions:
        _render_simplified_training_controls(
            simple_config=simple_config,
            simple_seed=simple_seed,
            simple_steps=simple_steps,
            simple_model_name=simple_model_name,
        )


def _render_simplified_config_selector() -> str:
    st.markdown("#### Simplified Training Configuration")
    st.info("Configs are loaded from: `configs/training/` (project) or package defaults")

    config_files = find_config_files()
    simplified_configs = [filename for filename in config_files if _is_simplified_config(filename)]

    if simplified_configs:
        simple_config = st.selectbox(
            "Select simplified config:",
            options=simplified_configs,
            index=0,
            help="Configs with 'simplified' in name or training_mode: simplified set in the builder",
        )
    elif config_files:
        simple_config = st.selectbox(
            "Select config (all available):",
            options=config_files,
            help="No simplified configs found — showing all. Any config can be run in simplified mode.",
        )
        st.warning(
            "Tip: set **Environment Type -> Simplified** in the Config Builder to tag a config for this list."
        )
    else:
        st.error("No configuration files found in `configs/training/`.")
        simple_config = st.text_input(
            "Config filename:", value="simple.yaml", placeholder="simple.yaml"
        )

    if simple_config:
        st.success(f"✓ Using: `{simple_config}`")
    return simple_config


def _is_simplified_config(filename: str) -> bool:
    from bvr_marl_core.utils.paths import rl_configs_root

    if "simplified" in filename.lower() or "simple" in filename.lower():
        return True
    for cfg_dir in [project_root() / "configs" / "training", rl_configs_root()]:
        cfg_path = cfg_dir / filename
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            if (data or {}).get("training_mode") == "simplified":
                return True
        except (OSError, yaml.YAMLError):
            pass
    return False


def _render_simplified_parameters() -> tuple[int, int, str]:
    simple_seed = st.number_input("Random seed (Simple)", value=42, min_value=0, key="simple_seed")
    simple_steps = st.number_input(
        "Training steps (Simple)", value=500, min_value=1, key="simple_steps"
    )
    simple_model_name = st.text_input(
        "Model name (Simple):", value="simple_model", key="simple_model_name"
    )
    return int(simple_seed), int(simple_steps), simple_model_name


def _render_simplified_training_controls(
    *,
    simple_config: str,
    simple_seed: int,
    simple_steps: int,
    simple_model_name: str,
) -> None:
    if st.button("Validate Architecture", key="validate_simple", width="stretch"):
        if not simple_config:
            st.error("Please select a configuration file first")
        else:
            with st.spinner("Running architecture check…"):
                result = validate_training_config(simple_config, training_mode="simplified")
            render_validation_result(result)

    if st.button("Start Simplified Training", type="primary", width="stretch"):
        if not simple_config:
            st.error("Please enter a simplified configuration file")
        else:
            launch_simplified_training(
                config=simple_config,
                seed=simple_seed,
                steps=simple_steps,
                model_name=simple_model_name,
            )


def _render_batch_operations_tab(default_config_name: str) -> None:
    st.subheader("Batch Training Operations")
    st.info("Batch training allows you to run multiple training jobs with different parameters.")

    batch_mode = st.selectbox(
        "Batch mode:", ["Multiple Seeds", "Config Variations", "Hyperparameter Sweep"]
    )
    if batch_mode == "Multiple Seeds":
        _render_multiple_seeds_batch(default_config_name)
    elif batch_mode == "Config Variations":
        _render_config_variations_batch()
    elif batch_mode == "Hyperparameter Sweep":
        _render_hyperparameter_sweep_batch(default_config_name)


def _render_multiple_seeds_batch(default_config_name: str) -> None:
    st.info("Configs are loaded from: `configs/training/` (project) or package defaults")
    base_config = _render_batch_base_config_selector(
        default_config_name,
        label="Base config file:",
        placeholder="base_config.yaml",
        help_text="Choose base configuration for batch training",
    )
    seed_list = st.text_input(
        "Seeds (comma-separated):",
        placeholder="42, 123, 456, 789",
        help="Enter seeds separated by commas",
    )

    if st.button("Start Batch Training", type="primary"):
        if base_config and seed_list:
            seeds = [int(seed.strip()) for seed in seed_list.split(",")]
            launch_batch_training_seeds(base_config, seeds)
        else:
            st.error("Please enter base config and seeds")


def _render_config_variations_batch() -> None:
    st.info("Configs are loaded from: `configs/training/` (project) or package defaults")

    config_files = find_config_files()
    selected_configs: list[str] = []
    if config_files:
        selected_configs = st.multiselect(
            "Select config files for batch training:",
            options=config_files,
            help="Choose multiple configurations to train with",
        )
        config_list = "\n".join(selected_configs) if selected_configs else ""
    else:
        st.error("No configuration files found in `configs/training/`.")
        config_list = st.text_area(
            "Config files (one per line):",
            placeholder="config1.yaml\nconfig2.yaml\nconfig3.yaml",
            help="Enter configuration filenames, one per line",
        )

    if st.button("Start Config Batch", type="primary"):
        if config_files and selected_configs:
            launch_batch_training_configs(selected_configs)
        elif config_list:
            configs = [config.strip() for config in config_list.split("\n") if config.strip()]
            launch_batch_training_configs(configs)
        else:
            st.error("Please select configuration files")


def _render_hyperparameter_sweep_batch(default_config_name: str) -> None:
    st.info("Configs are loaded from: `configs/training/` (project) or package defaults")
    sweep_config = _render_batch_base_config_selector(
        default_config_name,
        label="Base config for sweep:",
        placeholder="sweep_base.yaml",
        help_text="Choose base configuration for hyperparameter sweep",
    )
    sweep_param = st.text_input("Parameter to sweep:", placeholder="learning_rate")
    sweep_values = st.text_input(
        "Values (comma-separated):",
        placeholder="0.001, 0.0005, 0.0001",
        help="Enter parameter values separated by commas",
    )

    if st.button("Start Hyperparameter Sweep", type="primary"):
        if sweep_config and sweep_param and sweep_values:
            launch_hyperparameter_sweep(sweep_config, sweep_param, sweep_values)
        else:
            st.error("Please fill in all sweep parameters")


def _render_batch_base_config_selector(
    default_config_name: str,
    *,
    label: str,
    placeholder: str,
    help_text: str,
) -> str:
    config_files = find_config_files()
    if not config_files:
        st.error("No configuration files found in `configs/training/`.")
        return st.text_input("Base config filename:", placeholder=placeholder)

    default_index = (
        config_files.index(default_config_name) if default_config_name in config_files else 0
    )
    return st.selectbox(
        label,
        options=config_files,
        index=default_index,
        help=help_text,
    )


def _render_training_output_info() -> None:
    st.markdown("---")
    st.subheader("Training Output Information")

    col1, col2 = st.columns(2)
    with col1:
        display_compact_output_paths("training")
    with col2:
        display_recent_outputs("training")


def _register_and_display(process, cmd, config, model_name, steps, log_file, label="Training"):
    """Register a launched process with the monitor and show status messages."""
    if "training_monitor" not in st.session_state:
        st.session_state.training_monitor = TrainingMonitor()

    config_name = Path(config).name if config else "Unknown"
    training_process = st.session_state.training_monitor.register_process(
        process, config_name, model_name, log_file=str(log_file)
    )

    if steps:
        training_process.total_steps = steps
        processes = st.session_state.training_monitor.load_processes()
        processes[process.pid] = training_process
        st.session_state.training_monitor.save_processes(processes)

    st.success(f"{label} launched! Process ID: {process.pid}")
    st.info(f"Command: {' '.join(cmd)}")
    st.info(f"Log file: `{log_file}`")
    st.info(
        "Training is running in the background. Check the 'Active Training Monitor' tab for progress."
    )


def launch_training(config, seed, steps, model_name, resume_checkpoint=None):
    """Launch standard training."""
    with st.spinner("Launching training..."):
        overrides = [
            f"seed={seed}",
            f"training.steps={steps}",
            f"logging.model_name={model_name}",
        ]
        if resume_checkpoint:
            overrides.append(f"resume_checkpoint={resume_checkpoint}")

        cmd = build_training_cmd(config, overrides)
        try:
            process, log_file = launch_background_process(cmd, f"training_{model_name}")
            _register_and_display(process, cmd, config, model_name, steps, log_file)
        except Exception as e:
            st.error(f"Failed to launch training: {e}")


def launch_simplified_training(config, seed, steps, model_name):
    """Launch simplified training."""
    with st.spinner("Launching simplified training..."):
        overrides = [
            f"seed={seed}",
            f"training.steps={steps}",
            f"logging.model_name={model_name}",
        ]
        cmd = build_simple_training_cmd(config, overrides)
        try:
            process, log_file = launch_background_process(cmd, f"training_{model_name}")
            _register_and_display(
                process, cmd, config, model_name, steps, log_file, label="Simplified training"
            )
        except Exception as e:
            st.error(f"Failed to launch simplified training: {e}")


def launch_batch_training_seeds(config, seeds):
    """Launch batch training with multiple seeds."""
    with st.spinner("Launching batch training..."):
        from bvr_marl_core.services.batch import build_batch_jobs

        build_batch_jobs(base_config=config, seeds=seeds)
        [
            sys.executable,
            "-m",
            "bvr_marl_core.services.batch",
            "--config",
            config,
            "--seeds",
        ] + [str(s) for s in seeds]
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"from bvr_marl_core.services.batch import build_batch_jobs, run_batch_jobs; "
                    f"jobs = build_batch_jobs(base_config={config!r}, seeds={seeds}); "
                    f"run_batch_jobs(jobs)",
                ]
            )
            st.success(f"Batch training launched! Process ID: {process.pid}")
            st.info(f"Training {len(seeds)} models with seeds: {seeds}")
        except Exception as e:
            st.error(f"Failed to launch batch training: {e}")


def launch_batch_training_configs(configs):
    """Launch batch training with multiple configs."""
    with st.spinner("Launching config batch..."):
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"from bvr_marl_core.services.batch import build_batch_jobs, run_batch_jobs; "
                    f"jobs = build_batch_jobs(configs={configs!r}); "
                    f"run_batch_jobs(jobs)",
                ]
            )
            st.success(f"Config batch launched! Process ID: {process.pid}")
            st.info(f"Training {len(configs)} configurations")
        except Exception as e:
            st.error(f"Failed to launch config batch: {e}")


def launch_hyperparameter_sweep(config, param, values):
    """Launch hyperparameter sweep."""
    with st.spinner("Launching hyperparameter sweep..."):
        sweep_values = [v.strip() for v in values.split(",")]
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"from bvr_marl_core.services.batch import build_batch_jobs, run_batch_jobs; "
                    f"jobs = build_batch_jobs(base_config={config!r}, sweep_param={param!r}, "
                    f"sweep_values={sweep_values!r}); "
                    f"run_batch_jobs(jobs)",
                ]
            )
            st.success(f"Hyperparameter sweep launched! Process ID: {process.pid}")
            st.info(f"Sweeping {param} with values: {values}")
        except Exception as e:
            st.error(f"Failed to launch hyperparameter sweep: {e}")
