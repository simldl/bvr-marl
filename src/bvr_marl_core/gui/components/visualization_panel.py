"""
Visualization Control Panel

Interface for launching different visualization modes with process monitoring.
"""

import inspect
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

from bvr_marl_core.gui.components.checkpoint_picker import (
    checkpoint_picker,
    find_campaign_checkpoints,
    find_train_config_for_checkpoint,
)
from bvr_marl_core.gui.components.output_paths import (
    display_compact_output_paths,
    display_recent_outputs,
)
from bvr_marl_core.services import visualization as _viz_svc
from bvr_marl_core.services.processes import ProcessMonitor, ProcessRecord
from bvr_marl_core.services.training import launch_background_process
from bvr_marl_core.visualization.scenario_overlays import (
    get_visualization_scenario,
    list_visualization_scenarios,
)


class VizProcessMonitor(ProcessMonitor):
    """Visualization-specific process monitor (backed by ``viz_processes.json``)."""

    def __init__(self) -> None:
        super().__init__("viz_processes.json")


@dataclass(slots=True)
class _VisualizationInputs:
    mode: str
    selected_checkpoint: str | None
    train_config: str
    viz_config: str
    selected_scenario_key: str | None
    show_line_overlay: bool
    line_overlay_available: bool
    line_east_km: int
    map_extents_mode: str
    frames: int
    interval: int
    save_video: bool
    save_rewards: bool
    aircraft_type: str | None
    real_time: bool
    show_text: bool


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

    with st.expander("Active Visualization Processes", expanded=True):
        _render_viz_monitor()

    st.markdown("---")
    st.subheader("Launch New Visualization")

    viz_mode = _select_visualization_mode()
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        inputs = _render_visualization_inputs(viz_mode)
    with col2:
        _render_visualization_launch_controls(inputs)

    _render_visualization_output_info(inputs)


def _select_visualization_mode() -> str:
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
    return viz_mode


def _render_visualization_inputs(viz_mode: str) -> _VisualizationInputs:
    st.subheader("Configuration")
    selected_checkpoint = _render_model_selection(viz_mode)
    train_config, viz_config = _render_config_file_selectors(selected_checkpoint)
    (
        selected_scenario_key,
        show_line_overlay,
        line_overlay_available,
        line_east_km,
        map_extents_mode,
    ) = _render_scenario_overlay_settings()
    (
        frames,
        interval,
        save_video,
        save_rewards,
        aircraft_type,
        real_time,
        show_text,
    ) = _render_visualization_parameters(viz_mode)

    return _VisualizationInputs(
        mode=viz_mode,
        selected_checkpoint=selected_checkpoint,
        train_config=train_config,
        viz_config=viz_config,
        selected_scenario_key=selected_scenario_key,
        show_line_overlay=show_line_overlay,
        line_overlay_available=line_overlay_available,
        line_east_km=line_east_km,
        map_extents_mode=map_extents_mode,
        frames=frames,
        interval=interval,
        save_video=save_video,
        save_rewards=save_rewards,
        aircraft_type=aircraft_type,
        real_time=real_time,
        show_text=show_text,
    )


def _render_model_selection(viz_mode: str) -> str | None:
    if viz_mode == "Behavior Tree Controllers":
        return None

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
    return _select_campaign_checkpoint(selected_checkpoint)


def _select_campaign_checkpoint(selected_checkpoint: str | None) -> str | None:
    campaign_ckpts = find_campaign_checkpoints()
    if not campaign_ckpts:
        return selected_checkpoint

    with st.expander("Campaign-trained Models", expanded=True):
        campaign_labels = [lbl for _, lbl, _ in campaign_ckpts]
        campaign_sel = st.selectbox(
            "Campaign model:",
            options=["— none —"] + campaign_labels,
            key="viz_campaign_ckpt",
            help=(
                "Automatically discovered checkpoints from campaign training runs. "
                "Selecting one overrides the model picker above."
            ),
        )
        if campaign_sel == "— none —":
            return selected_checkpoint

        selected_checkpoint = next(ckpt for _, lbl, ckpt in campaign_ckpts if lbl == campaign_sel)
        st.caption(f"`{selected_checkpoint}`")
        return selected_checkpoint


def _render_config_file_selectors(selected_checkpoint: str | None) -> tuple[str, str]:
    st.markdown("#### Configuration Files")
    st.info("Visualization configs are loaded from: `visualization/configs/`.")
    auto_train_cfg = _auto_detect_train_config(selected_checkpoint)

    col1, col2 = st.columns(2)
    with col1:
        train_config = _select_training_config(auto_train_cfg)
    with col2:
        viz_config = _select_viz_config()
    return train_config, viz_config


def _auto_detect_train_config(selected_checkpoint: str | None) -> str | None:
    if not selected_checkpoint:
        return None
    auto_train_cfg = find_train_config_for_checkpoint(selected_checkpoint)
    if auto_train_cfg:
        st.success(f"Auto-detected training config: `{auto_train_cfg}`")
    return auto_train_cfg


def _select_training_config(auto_train_cfg: str | None) -> str:
    st.markdown("**Training Config:**")
    from bvr_marl_core.services.training import list_training_configs

    train_cfg_options = ["Default"] + list_training_configs()
    train_cfg_default = 0
    if auto_train_cfg and auto_train_cfg not in train_cfg_options:
        train_cfg_options = [auto_train_cfg] + train_cfg_options
    if auto_train_cfg:
        train_cfg_default = train_cfg_options.index(auto_train_cfg)
    return st.selectbox(
        "Select training config:",
        options=train_cfg_options,
        index=train_cfg_default,
    )


def _select_viz_config() -> str:
    st.markdown("**Visualization Config:**")
    config_files = find_config_files()
    if not config_files:
        st.warning("No visualization configs found in `visualization/configs/`.")
        return "Default"

    viz_config = st.selectbox(
        "Select visualization config:",
        options=["Default"] + config_files,
    )
    if viz_config != "Default":
        st.success(f"Using: `visualization/configs/{viz_config}`")
    return viz_config


def _render_scenario_overlay_settings() -> tuple[str | None, bool, bool, int, str]:
    selected_scenario_key, selected_scenario = _select_scenario_overlay()
    show_line_overlay, line_overlay_available, line_east_km = _render_line_overlay_controls(
        selected_scenario
    )
    map_extents_mode = _select_map_extents()
    return (
        selected_scenario_key,
        show_line_overlay,
        line_overlay_available,
        line_east_km,
        map_extents_mode,
    )


def _select_scenario_overlay():
    scenario_options = list_visualization_scenarios()
    scenario_labels = [scenario.label for scenario in scenario_options]
    scenario_override_label = st.selectbox(
        "Scenario Overlay:",
        options=["Use Visualization Config"] + scenario_labels,
        help="Choose a scenario overlay for the 2D map, or leave it to the visualization config.",
    )
    if scenario_override_label == "Use Visualization Config":
        return None, None

    selected_scenario = get_visualization_scenario(scenario_override_label)
    st.info(selected_scenario.description)
    return selected_scenario.key, selected_scenario


def _render_line_overlay_controls(selected_scenario) -> tuple[bool, bool, int]:
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
    return show_line_overlay, line_overlay_available, line_east_km


def _select_map_extents() -> str:
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
    return {
        "Auto": "auto",
        "Combat Zone": "combat",
        "Full Scenario": "full",
    }[map_extents_label]


def _render_visualization_parameters(
    viz_mode: str,
) -> tuple[int, int, bool, bool, str | None, bool, bool]:
    st.markdown("#### Visualization Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        frames = st.number_input("Frames", value=100, min_value=1, max_value=10000)
    with col2:
        real_time = st.checkbox("Real Time", help="Run at simulation tick rate (ignores Interval)")
        interval = st.number_input(
            "Interval (ms)",
            value=100,
            min_value=10,
            max_value=1000,
            disabled=real_time,
        )
    with col3:
        save_video = st.checkbox("Save video", key="viz_save_video")

    if real_time:
        st.info("Real-time mode sets animation speed to the simulation tick rate.")
    show_text = st.checkbox(
        "Show map text",
        value=True,
        key="viz_show_text",
        help="Show aircraft labels, status overlays, and the scale-bar label on the 2D map.",
    )
    save_rewards, aircraft_type = _render_advanced_options(viz_mode)
    return frames, interval, save_video, save_rewards, aircraft_type, real_time, show_text


def _render_advanced_options(viz_mode: str) -> tuple[bool, str | None]:
    with st.expander("Advanced Options"):
        save_rewards = st.checkbox("Save reward logs", value=True)
        aircraft_type = None
        if viz_mode == "Behavior Tree Controllers":
            aircraft_type = st.selectbox("Aircraft type:", ["F22", "Eurofighter", "F35"])
        elif viz_mode == "RL Commands Panel":
            st.info("Shows time-series plots of RL actions (energy, pitch, roll commands)")
    return save_rewards, aircraft_type


def _render_visualization_launch_controls(inputs: _VisualizationInputs) -> None:
    st.subheader("Launch")
    if st.button("Launch Visualization", type="primary", width="stretch"):
        launch_visualization(
            mode=inputs.mode,
            checkpoint=_final_checkpoint(inputs),
            train_config=None if inputs.train_config == "Default" else inputs.train_config,
            viz_config=None if inputs.viz_config == "Default" else inputs.viz_config,
            scenario=inputs.selected_scenario_key,
            show_line_overlay=inputs.show_line_overlay,
            line_east_km=_line_east_launch_value(inputs),
            map_extents_mode=inputs.map_extents_mode,
            frames=inputs.frames,
            interval=inputs.interval,
            save_video=inputs.save_video,
            save_rewards=inputs.save_rewards,
            aircraft_type=inputs.aircraft_type,
            real_time=inputs.real_time,
            show_text=inputs.show_text,
        )

    st.markdown("---")
    st.markdown("#### Quick Launch")
    _render_quick_launch_controls(inputs)


def _final_checkpoint(inputs: _VisualizationInputs) -> str | None:
    if inputs.mode == "Behavior Tree Controllers":
        return None
    return inputs.selected_checkpoint


def _line_east_launch_value(inputs: _VisualizationInputs) -> float | None:
    if not inputs.line_overlay_available:
        return None
    return float(inputs.line_east_km)


def _render_quick_launch_controls(inputs: _VisualizationInputs) -> None:
    if inputs.mode != "Behavior Tree Controllers":
        if st.button(f"Random Actions ({inputs.mode})", width="stretch"):
            _launch_quick_from_inputs(inputs, inputs.mode, None)

        if inputs.selected_checkpoint and st.button(
            f"Selected Checkpoint ({inputs.mode})",
            width="stretch",
        ):
            _launch_quick_from_inputs(inputs, inputs.mode, inputs.selected_checkpoint)

    if st.button("Behavior Trees", width="stretch"):
        _launch_quick_from_inputs(inputs, "Behavior Tree Controllers", None)


def _launch_quick_from_inputs(
    inputs: _VisualizationInputs,
    mode: str,
    checkpoint: str | None,
) -> None:
    launch_quick_visualization(
        mode,
        checkpoint,
        real_time=inputs.real_time,
        save_video=inputs.save_video,
        scenario=inputs.selected_scenario_key,
        show_line_overlay=inputs.show_line_overlay,
        line_east_km=_line_east_launch_value(inputs),
        map_extents_mode=inputs.map_extents_mode,
        show_text=inputs.show_text,
    )


def _render_visualization_output_info(inputs: _VisualizationInputs) -> None:
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
        if inputs.selected_checkpoint:
            st.markdown("**Selected Checkpoint:**")
            st.text(Path(inputs.selected_checkpoint).name)
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
