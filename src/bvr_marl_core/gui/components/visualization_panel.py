"""
Visualization Control Panel

Interface for launching different visualization modes with process monitoring.
"""

import inspect
import os
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

from bvr_marl_core.services import visualization as _viz_svc
from bvr_marl_core.services.processes import ProcessMonitor, ProcessRecord
from bvr_marl_core.services.training import launch_background_process
from bvr_marl_core.visualization.scenario_overlays import (
    get_visualization_scenario,
    list_visualization_scenarios,
)

from .checkpoint_picker import (
    checkpoint_picker,
    find_campaign_checkpoints,
    find_train_config_for_checkpoint,
)
from .output_paths import display_compact_output_paths, display_recent_outputs


class VizProcessMonitor(ProcessMonitor):
    """Visualization-specific process monitor (backed by ``viz_processes.json``)."""

    def __init__(self) -> None:
        super().__init__("viz_processes.json")


def _render_viz_process_card(vp: ProcessRecord, monitor: VizProcessMonitor, is_dead: bool):
    runtime = datetime.now() - vp.start_time
    runtime_str = str(runtime).split(".")[0]
    status = "Stopped" if is_dead else "Running"

    st.markdown(
        f"<div style='border:1px solid {'#dc3545' if is_dead else '#28a745'};"
        f"border-radius:8px;padding:12px;margin:6px 0;"
        f"background-color:{'rgba(220,53,69,0.05)' if is_dead else 'rgba(40,167,69,0.05)'}'>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_title, col_stop = st.columns([4, 1])
    with col_title:
        st.markdown(f"**{status} — {vp.label}**")
        st.caption(f"PID: {vp.pid} | Runtime: {runtime_str}")
    with col_stop:
        if not is_dead:
            if st.button(
                "Stop",
                key=f"viz_stop_{vp.pid}",
                width="stretch",
                help="Kill this process and all its children",
            ):
                with st.spinner("Stopping…"):
                    monitor.terminate(vp.pid)
                st.rerun()

    # Log file controls
    log_path = Path(vp.log_file)
    col_path, col_open = st.columns([4, 1])
    with col_path:
        st.caption(f"Log: `{vp.log_file}`")
    with col_open:
        if st.button("Open", key=f"viz_open_{vp.pid}", width="stretch"):
            if log_path.exists():
                if os.name == "nt":
                    os.startfile(str(log_path))
                else:
                    subprocess.Popen(["xdg-open", str(log_path)])
            else:
                st.warning("Log file not created yet.")

    if log_path.exists():
        with st.expander("Log Output (last 30 lines)", expanded=is_dead):
            try:
                with open(log_path, encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                cleaned = raw.replace("\r", "\n")
                lines = [ln for ln in cleaned.splitlines() if ln.strip()]
                st.code("\n".join(lines[-30:]), language="text")
            except Exception as e:
                st.warning(f"Could not read log: {e}")

    with st.expander("Command"):
        st.code(vp.command, language="bash")


def _render_viz_monitor():
    """Active visualization process monitor section."""
    if "viz_monitor" not in st.session_state:
        st.session_state.viz_monitor = VizProcessMonitor()

    monitor: VizProcessMonitor = st.session_state.viz_monitor
    procs = monitor.update()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.caption(
            f"{len([p for p in procs.values() if p.is_alive])} running, "
            f"{len([p for p in procs.values() if not p.is_alive])} completed/failed"
        )
    with col2:
        if st.button("Refresh", key="viz_refresh", width="stretch"):
            st.rerun()
    with col3:
        if st.button("Clean Up", key="viz_cleanup", width="stretch"):
            monitor.cleanup_dead()
            st.rerun()

    alive = {pid: vp for pid, vp in procs.items() if vp.is_alive}
    dead = {pid: vp for pid, vp in procs.items() if not vp.is_alive}

    if alive:
        st.markdown("### Running Visualizations")
        for pid, vp in alive.items():
            _render_viz_process_card(vp, monitor, is_dead=False)

    if dead:
        with st.expander(f"Completed / Failed ({len(dead)})"):
            for pid, vp in dead.items():
                _render_viz_process_card(vp, monitor, is_dead=True)

    if not procs:
        st.info("No visualization processes tracked yet. Launch one below.")


def find_checkpoint_files():
    return _viz_svc.find_checkpoint_files()


def find_config_files():
    from bvr_marl_core.services.configs import list_visualization_configs

    return list_visualization_configs()


def visualization_panel():
    """Render the visualization control panel."""
    st.header("Visualization Panel")
    st.caption(
        "Launch live views, inspect process state, and configure analysis-oriented visualization runs."
    )

    # --- Active process monitor ---
    with st.expander("Active Visualization Processes", expanded=True):
        _render_viz_monitor()

    st.markdown("---")

    # --- Launch section ---
    st.subheader("Launch New Visualization")

    viz_mode = st.selectbox(
        "Select visualization mode:",
        [
            "Standard 2D Live View",
            "Behavior Tree Controllers",
            "RL Commands Panel",
        ],
        help="Choose the type of visualization to launch",
    )

    mode_descriptions = {
        "Standard 2D Live View": "Standard real-time 2D battle visualization with aircraft and missiles.",
        "Behavior Tree Controllers": "Visualizes scripted behavior tree controllers in action.",
        "RL Commands Panel": "Displays RL action commands (energy, pitch, roll) as time series.",
    }
    st.info(mode_descriptions[viz_mode])

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuration")

        # Model selection
        selected_checkpoint = None
        if viz_mode != "Behavior Tree Controllers":
            st.markdown("#### Model Selection")
            selected_checkpoint = checkpoint_picker(
                key_prefix="viz",
                allow_none=True,
                none_label="Random Actions",
                label="Trained Model",
            )
            with st.expander("Custom checkpoint path"):
                custom_checkpoint = st.text_input(
                    "Paste a full checkpoint path:",
                    placeholder="models/my_model/PPO_.../checkpoint_000010",
                    key="viz_custom_ckpt",
                )
                if custom_checkpoint:
                    selected_checkpoint = custom_checkpoint

            # --- Campaign-trained models ---
            _campaign_ckpts = find_campaign_checkpoints()

            if _campaign_ckpts:
                with st.expander("Campaign-trained Models", expanded=True):
                    _campaign_labels = [lbl for _, lbl, _ in _campaign_ckpts]
                    _campaign_sel = st.selectbox(
                        "Campaign model:",
                        options=["— none —"] + _campaign_labels,
                        key="viz_campaign_ckpt",
                        help=(
                            "Automatically discovered checkpoints from campaign training runs. "
                            "Selecting one overrides the model picker above."
                        ),
                    )
                    if _campaign_sel != "— none —":
                        selected_checkpoint = next(
                            ckpt for _, lbl, ckpt in _campaign_ckpts if lbl == _campaign_sel
                        )
                        st.caption(f"`{selected_checkpoint}`")
        else:
            custom_checkpoint = ""

        # Config files
        st.markdown("#### Configuration Files")
        st.info("Visualization configs are loaded from: `visualization/configs/`.")

        # Auto-detect training config from checkpoint
        _auto_train_cfg: str | None = None
        if selected_checkpoint:
            _auto_train_cfg = find_train_config_for_checkpoint(selected_checkpoint)
            if _auto_train_cfg:
                st.success(f"Auto-detected training config: `{_auto_train_cfg}`")

        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.markdown("**Training Config:**")
            from bvr_marl_core.services.training import list_training_configs

            _train_cfg_options = ["Default"] + list_training_configs()
            _train_cfg_default = 0
            if _auto_train_cfg and _auto_train_cfg not in _train_cfg_options:
                # Prepend the auto-detected path so it can be pre-selected
                _train_cfg_options = [_auto_train_cfg] + _train_cfg_options
            if _auto_train_cfg:
                _train_cfg_default = _train_cfg_options.index(_auto_train_cfg)
            train_config = st.selectbox(
                "Select training config:",
                options=_train_cfg_options,
                index=_train_cfg_default,
            )
        with col1_2:
            st.markdown("**Visualization Config:**")
            config_files = find_config_files()
            if config_files:
                viz_config = st.selectbox(
                    "Select visualization config:",
                    options=["Default"] + config_files,
                )
                if viz_config != "Default":
                    st.success(f"Using: `visualization/configs/{viz_config}`")
            else:
                st.warning("No visualization configs found in `visualization/configs/`.")
                viz_config = "Default"

        scenario_options = list_visualization_scenarios()
        scenario_labels = [scenario.label for scenario in scenario_options]
        scenario_override_label = st.selectbox(
            "Scenario Overlay:",
            options=["Use Visualization Config"] + scenario_labels,
            help="Choose a scenario overlay for the 2D map, or leave it to the visualization config.",
        )
        selected_scenario_key = None
        selected_scenario = None
        if scenario_override_label != "Use Visualization Config":
            selected_scenario = get_visualization_scenario(scenario_override_label)
            selected_scenario_key = selected_scenario.key
            st.info(selected_scenario.description)

        st.markdown("#### Line Overlay")
        show_line_overlay = st.checkbox(
            "Show line of engagement on the map",
            value=False,
            help=(
                "Adds a movable LOE to any scenario. Built-in scenario lines still appear "
                "even when this is unchecked."
            ),
        )
        line_overlay_available = show_line_overlay or (
            selected_scenario is not None and selected_scenario.has_line_of_engagement
        )
        line_east_km = st.slider(
            "Line X Offset (km)",
            min_value=-300,
            max_value=300,
            value=0,
            step=5,
            disabled=not line_overlay_available,
            help="Move the LOE west/east along the x-axis. Negative is west, positive is east.",
        )

        st.markdown("#### Map Extents")
        map_extents_label = st.selectbox(
            "Displayed Map Area:",
            options=["Auto", "Combat Zone", "Full Scenario"],
            index=0,
            help=(
                "Auto follows the selected scenario default. Full Scenario shows the full "
                "display bounds, including side zones such as AWACS areas."
            ),
        )
        map_extents_mode = {
            "Auto": "auto",
            "Combat Zone": "combat",
            "Full Scenario": "full",
        }[map_extents_label]

        # Parameters
        st.markdown("#### Visualization Parameters")
        col1_3, col1_4, col1_5 = st.columns(3)
        with col1_3:
            frames = st.number_input("Frames", value=100, min_value=1, max_value=10000)
        with col1_4:
            real_time = st.checkbox(
                "Real Time", help="Run at simulation tick rate (ignores Interval)"
            )
            interval_disabled = real_time
            interval = st.number_input(
                "Interval (ms)", value=100, min_value=10, max_value=1000, disabled=interval_disabled
            )
        with col1_5:
            save_video = st.checkbox("Save video", key="viz_save_video")
        if real_time:
            st.info("Real-time mode sets animation speed to the simulation tick rate.")
        show_text = st.checkbox(
            "Show map text",
            value=True,
            key="viz_show_text",
            help="Show aircraft labels, status overlays, and the scale-bar label on the 2D map.",
        )

        with st.expander("Advanced Options"):
            save_rewards = st.checkbox("Save reward logs", value=True)
            aircraft_type = None
            if viz_mode == "Behavior Tree Controllers":
                aircraft_type = st.selectbox("Aircraft type:", ["F22", "Eurofighter", "F35"])
            elif viz_mode == "RL Commands Panel":
                st.info("Shows time-series plots of RL actions (energy, pitch, roll commands)")

    with col2:
        st.subheader("Launch")

        if st.button("Launch Visualization", type="primary", width="stretch"):
            final_checkpoint = (
                selected_checkpoint if viz_mode != "Behavior Tree Controllers" else None
            )

            final_train_config = None if train_config == "Default" else train_config
            final_viz_config = None if viz_config == "Default" else viz_config

            launch_visualization(
                mode=viz_mode,
                checkpoint=final_checkpoint,
                train_config=final_train_config,
                viz_config=final_viz_config,
                scenario=selected_scenario_key,
                show_line_overlay=show_line_overlay,
                line_east_km=float(line_east_km) if line_overlay_available else None,
                map_extents_mode=map_extents_mode,
                frames=frames,
                interval=interval,
                save_video=save_video,
                save_rewards=save_rewards,
                aircraft_type=aircraft_type,
                real_time=real_time,
                show_text=show_text,
            )

        st.markdown("---")
        st.markdown("#### Quick Launch")

        if viz_mode != "Behavior Tree Controllers":
            if st.button(f"Random Actions ({viz_mode})", width="stretch"):
                launch_quick_visualization(
                    viz_mode,
                    None,
                    real_time=real_time,
                    save_video=save_video,
                    scenario=selected_scenario_key,
                    show_line_overlay=show_line_overlay,
                    line_east_km=float(line_east_km) if line_overlay_available else None,
                    map_extents_mode=map_extents_mode,
                    show_text=show_text,
                )

            if selected_checkpoint and st.button(
                f"Selected Checkpoint ({viz_mode})", width="stretch"
            ):
                launch_quick_visualization(
                    viz_mode,
                    selected_checkpoint,
                    real_time=real_time,
                    save_video=save_video,
                    scenario=selected_scenario_key,
                    show_line_overlay=show_line_overlay,
                    line_east_km=float(line_east_km) if line_overlay_available else None,
                    map_extents_mode=map_extents_mode,
                    show_text=show_text,
                )

        if st.button("Behavior Trees", width="stretch"):
            launch_quick_visualization(
                "Behavior Tree Controllers",
                None,
                real_time=real_time,
                save_video=save_video,
                scenario=selected_scenario_key,
                show_line_overlay=show_line_overlay,
                line_east_km=float(line_east_km) if line_overlay_available else None,
                map_extents_mode=map_extents_mode,
                show_text=show_text,
            )

    st.markdown("---")
    st.subheader("Output Information")
    col1, col2 = st.columns(2)
    with col1:
        display_compact_output_paths("visualization")
        st.markdown("""
        **Tips:**
        - Use **Standard** mode for general visualization
        - Use **Behavior Tree** mode to see scripted AI
        - Use **Commands** mode to analyze RL actions
        """)
    with col2:
        display_recent_outputs("visualization")
        if selected_checkpoint:
            st.markdown("**Selected Checkpoint:**")
            st.text(Path(selected_checkpoint).name)
        else:
            st.markdown("**No checkpoint selected**")
            st.text("Select a model above or train one first.")


def launch_visualization(
    mode,
    checkpoint=None,
    train_config=None,
    viz_config=None,
    scenario=None,
    show_line_overlay=False,
    line_east_km=None,
    map_extents_mode=None,
    frames=100,
    interval=100,
    save_video=False,
    save_rewards=True,
    aircraft_type=None,
    real_time=False,
    show_text=True,
):
    """Launch a visualization process with log capture."""
    if save_video:
        from bvr_marl_core.utils.paths import core_project_root

        (core_project_root() / "output" / "videos").mkdir(parents=True, exist_ok=True)

    cmd_kwargs = {
        "mode": mode,
        "checkpoint": checkpoint,
        "train_config": train_config,
        "viz_config": viz_config,
        "frames": frames,
        "interval": interval,
        "save_video": save_video,
        "save_rewards": save_rewards,
        "aircraft_type": aircraft_type,
        "real_time": real_time,
        "show_line_overlay": show_line_overlay,
        "line_east_km": line_east_km,
        "map_extents_mode": map_extents_mode,
        "show_text": show_text,
    }
    if "scenario" in inspect.signature(_viz_svc.build_visualization_cmd).parameters:
        cmd_kwargs["scenario"] = scenario
    cmd = _viz_svc.build_visualization_cmd(**cmd_kwargs)

    with st.spinner(f"Launching {mode}…"):
        try:
            process, log_file = launch_background_process(cmd, label=f"viz_{mode}")

            if "viz_monitor" not in st.session_state:
                st.session_state.viz_monitor = VizProcessMonitor()
            st.session_state.viz_monitor.register(process, mode, str(log_file))

            st.success(f"{mode} launched! PID: {process.pid}")
            st.info(f"Log: `{log_file}`")
            st.code(" ".join(cmd), language="bash")
            st.info("Check 'Active Visualization Processes' above for output.")

        except Exception as e:
            st.error(f"Failed to launch visualization: {e}")


def launch_quick_visualization(
    mode,
    checkpoint=None,
    real_time=False,
    save_video=False,
    scenario=None,
    show_line_overlay=False,
    line_east_km=None,
    map_extents_mode=None,
    show_text=True,
):
    """Launch visualization with default settings."""
    train_config = find_train_config_for_checkpoint(checkpoint) if checkpoint else None
    launch_visualization(
        mode=mode,
        checkpoint=checkpoint,
        train_config=train_config,
        scenario=scenario,
        show_line_overlay=show_line_overlay,
        line_east_km=line_east_km,
        map_extents_mode=map_extents_mode,
        frames=100,
        interval=100,
        save_video=save_video,
        save_rewards=True,
        real_time=real_time,
        show_text=show_text,
    )
