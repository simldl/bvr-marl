"""
Tacview Generator Interface

Generates Tacview ACMI scenarios via installed commands:
  - RL controller: ``bvr-tacview`` / ``python -m bvr_marl_core.tacview.generate``
  - Behavior-tree controller: optional ``bvr-tacview-bt`` behavior command

Supported arguments for RL mode (generate.py):
  --checkpoint PATH
  --model-config PATH
  --train-config PATH
  --frames N
  --acmi PATH
  --num-scenarios N
  --seed-start N

Supported arguments for behavior-tree mode:
  --aircraft F22|Eurofighter|F35
  --frames N
  --acmi PATH
  --num-scenarios N
  --seed-start N
"""

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
from bvr_marl_core.services.processes import ProcessMonitor, ProcessRecord
from bvr_marl_core.services.tacview import build_tacview_cmd
from bvr_marl_core.services.training import launch_background_process


class TacviewProcessMonitor(ProcessMonitor):
    """Tacview-specific process monitor (backed by ``tacview_processes.json``)."""

    def __init__(self) -> None:
        super().__init__("tacview_processes.json")


@dataclass(slots=True)
class _RlScenarioInputs:
    selected_ckpt: str | None
    train_cfg: str | None
    num_scenarios: int
    frames: int
    seed_start: int
    acmi_path: str


@dataclass(slots=True)
class _BtScenarioInputs:
    aircraft: str
    num_scenarios: int
    frames: int
    seed_start: int
    acmi_path: str


def _render_tacview_card(tp: ProcessRecord, monitor: TacviewProcessMonitor, is_dead: bool):
    runtime = str(datetime.now() - tp.start_time).split(".")[0]
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
        st.markdown(f"**{status} — {tp.label}**")
        st.caption(f"PID: {tp.pid} | Runtime: {runtime}")
    with col_stop:
        if not is_dead:
            if st.button("Stop", key=f"tv_stop_{tp.pid}", width="stretch"):
                with st.spinner("Stopping…"):
                    monitor.terminate(tp.pid)
                st.rerun()

    log_path = Path(tp.log_file)
    col_path, col_open = st.columns([4, 1])
    with col_path:
        st.caption(f"Log: `{tp.log_file}`")
    with col_open:
        if st.button("Open", key=f"tv_open_{tp.pid}", width="stretch"):
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
                lines = [ln for ln in raw.replace("\r", "\n").splitlines() if ln.strip()]
                st.code("\n".join(lines[-30:]), language="text")
            except Exception as e:
                st.warning(f"Could not read log: {e}")

    with st.expander("Command"):
        st.code(tp.command, language="bash")


def _render_monitor():
    if "tacview_monitor" not in st.session_state:
        st.session_state.tacview_monitor = TacviewProcessMonitor()

    monitor: TacviewProcessMonitor = st.session_state.tacview_monitor
    procs = monitor.update()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        alive_n = sum(1 for p in procs.values() if p.is_alive)
        dead_n = len(procs) - alive_n
        st.caption(f"{alive_n} running, {dead_n} completed/failed")
    with col2:
        if st.button("Refresh", key="tv_refresh", width="stretch"):
            st.rerun()
    with col3:
        if st.button("Clean Up", key="tv_cleanup", width="stretch"):
            monitor.cleanup_dead()
            st.rerun()

    alive = {pid: tp for pid, tp in procs.items() if tp.is_alive}
    dead = {pid: tp for pid, tp in procs.items() if not tp.is_alive}

    if alive:
        st.markdown("### Running")
        for pid, tp in alive.items():
            _render_tacview_card(tp, monitor, is_dead=False)

    if dead:
        with st.expander(f"Completed / Failed ({len(dead)})"):
            for pid, tp in dead.items():
                _render_tacview_card(tp, monitor, is_dead=True)

    if not procs:
        st.info("No tacview processes tracked yet. Launch one below.")


def find_checkpoint_files():
    from bvr_marl_core.services.visualization import find_checkpoint_files as _find

    return _find()


def find_train_configs():
    from bvr_marl_core.services.training import list_training_configs

    return list_training_configs()


def _launch(label: str, cmd: list[str], monitor: TacviewProcessMonitor):
    """Launch *cmd* as a background process and register it with *monitor*."""
    with st.spinner(f"Launching: {label}…"):
        try:
            process, log_file = launch_background_process(cmd, label=f"tacview_{label}")
            monitor.register(process, label, str(log_file))
            st.success(f"Started — PID {process.pid}")
            st.info(f"Log: `{log_file}`")
            st.code(" ".join(cmd), language="bash")
            st.info("Check 'Active Tacview Processes' above for output.")
        except Exception as e:
            st.error(f"Failed to launch: {e}")


def tacview_generator():
    """Render the Tacview generator interface."""
    st.header("Tacview Generator")
    st.caption("Generate ACMI scenarios for RL and behavior-tree controllers.")

    st.info(
        "Generates `.acmi` scenario files via installed commands "
        "(`bvr-tacview` for RL, `bvr-tacview-bt` for behavior tree)."
    )

    # Active process monitor
    with st.expander("Active Tacview Processes", expanded=True):
        _render_monitor()

    st.markdown("---")
    st.subheader("Generate New Scenario")

    if "tacview_monitor" not in st.session_state:
        st.session_state.tacview_monitor = TacviewProcessMonitor()
    monitor: TacviewProcessMonitor = st.session_state.tacview_monitor

    tab1, tab2 = st.tabs(["RL Model", "Behavior Tree"])

    with tab1:
        _render_rl_scenario_tab(monitor)

    with tab2:
        _render_bt_scenario_tab(monitor)

    _render_tacview_output_info()


def _render_rl_scenario_tab(monitor: TacviewProcessMonitor) -> None:
    st.subheader("RL Model Scenarios")
    col1, col2 = st.columns([2, 1])
    with col1:
        inputs = _render_rl_scenario_inputs()
    with col2:
        _render_rl_launch_controls(monitor, inputs)


def _render_rl_scenario_inputs() -> _RlScenarioInputs:
    st.markdown("#### Model")
    selected_ckpt = _select_rl_checkpoint()
    train_cfg = _select_rl_training_config(selected_ckpt)

    st.markdown("#### Scenario Parameters")
    col1, col2 = st.columns(2)
    with col1:
        rl_num = st.number_input(
            "Number of scenarios:",
            min_value=1,
            max_value=50,
            value=1,
            key="rl_num",
        )
        rl_frames = st.number_input(
            "Frames per scenario:",
            min_value=50,
            max_value=5000,
            value=500,
            key="rl_frames",
        )
    with col2:
        rl_seed_start = st.number_input("Seed start:", min_value=0, value=0, key="rl_seed")
        rl_acmi = st.text_input(
            "ACMI output path (optional):",
            placeholder="Auto-generated if empty",
            key="rl_acmi",
        )

    return _RlScenarioInputs(
        selected_ckpt=selected_ckpt,
        train_cfg=train_cfg,
        num_scenarios=rl_num,
        frames=rl_frames,
        seed_start=rl_seed_start,
        acmi_path=rl_acmi,
    )


def _select_rl_checkpoint() -> str | None:
    selected_ckpt = checkpoint_picker(
        key_prefix="tacview_rl",
        allow_none=True,
        none_label="Random Actions (no checkpoint)",
        label="RL Checkpoint",
    )
    with st.expander("Custom checkpoint path"):
        custom_ckpt = st.text_input(
            "Paste a full checkpoint path:",
            placeholder="models/my_model/PPO_.../checkpoint_000010",
            key="rl_custom_ckpt",
        )
        if custom_ckpt:
            selected_ckpt = custom_ckpt
    return _select_campaign_checkpoint(selected_ckpt)


def _select_campaign_checkpoint(selected_ckpt: str | None) -> str | None:
    campaign_ckpts = find_campaign_checkpoints()
    if not campaign_ckpts:
        return selected_ckpt

    with st.expander("Campaign-trained Models", expanded=True):
        campaign_labels = [label for _, label, _ in campaign_ckpts]
        campaign_selection = st.selectbox(
            "Campaign model:",
            options=["No campaign model"] + campaign_labels,
            key="tacview_campaign_ckpt",
            help=(
                "Automatically discovered checkpoints from campaign training runs. "
                "Selecting one overrides the model picker above."
            ),
        )
        if campaign_selection == "No campaign model":
            return selected_ckpt

        selected_ckpt = next(
            ckpt for _, label, ckpt in campaign_ckpts if label == campaign_selection
        )
        st.caption(f"`{selected_ckpt}`")
        return selected_ckpt


def _select_rl_training_config(selected_ckpt: str | None) -> str | None:
    auto_train_cfg = find_train_config_for_checkpoint(selected_ckpt) if selected_ckpt else None
    if auto_train_cfg:
        st.success(f"Auto-detected training config: `{auto_train_cfg}`")

    train_configs = find_train_configs()
    if not train_configs:
        return st.text_input(
            "Training config path:",
            value=auto_train_cfg or "",
            key="rl_train_cfg_text",
        )

    options = ["Default"] + train_configs
    default_index = 0
    if auto_train_cfg and auto_train_cfg not in options:
        options = [auto_train_cfg] + options
    if auto_train_cfg and auto_train_cfg in options:
        default_index = options.index(auto_train_cfg)
    return st.selectbox(
        "Training config (for env setup):",
        options=options,
        index=default_index,
        key="rl_train_cfg",
    )


def _render_rl_launch_controls(
    monitor: TacviewProcessMonitor,
    inputs: _RlScenarioInputs,
) -> None:
    st.markdown("#### Launch")
    if st.button("Generate RL Scenario", type="primary", width="stretch", key="btn_rl"):
        cfg = None if inputs.train_cfg in ("Default", "") else inputs.train_cfg
        cmd = build_tacview_cmd(
            controller="rl",
            checkpoint=inputs.selected_ckpt,
            train_config=cfg,
            frames=inputs.frames,
            num_scenarios=inputs.num_scenarios,
            seed_start=inputs.seed_start,
            acmi_path=inputs.acmi_path or None,
        )
        _launch(f"RL_{inputs.num_scenarios}sc", cmd, monitor)

    st.markdown("---")
    st.markdown("#### Quick Launch")
    if st.button("Quick Demo (random actions)", width="stretch", key="btn_rl_demo"):
        _launch(
            "RL_demo",
            build_tacview_cmd(controller="rl", frames=500, num_scenarios=1),
            monitor,
        )

    if inputs.selected_ckpt and st.button(
        "Selected Checkpoint (1 scenario)",
        width="stretch",
        key="btn_rl_latest",
    ):
        _launch(
            "RL_latest",
            build_tacview_cmd(
                controller="rl",
                checkpoint=inputs.selected_ckpt,
                frames=500,
                num_scenarios=1,
            ),
            monitor,
        )


def _render_bt_scenario_tab(monitor: TacviewProcessMonitor) -> None:
    st.subheader("Behavior Tree Scenarios")
    col1, col2 = st.columns([2, 1])
    with col1:
        inputs = _render_bt_scenario_inputs()
    with col2:
        _render_bt_launch_controls(monitor, inputs)


def _render_bt_scenario_inputs() -> _BtScenarioInputs:
    st.markdown("#### Aircraft & Parameters")
    aircraft = st.selectbox("Aircraft type:", ["F22", "Eurofighter", "F35"], key="bt_aircraft")

    col1, col2 = st.columns(2)
    with col1:
        bt_num = st.number_input(
            "Number of scenarios:",
            min_value=1,
            max_value=50,
            value=3,
            key="bt_num",
        )
        bt_frames = st.number_input(
            "Frames per scenario:",
            min_value=50,
            max_value=5000,
            value=500,
            key="bt_frames",
        )
    with col2:
        bt_seed_start = st.number_input("Seed start:", min_value=0, value=0, key="bt_seed")
        bt_acmi = st.text_input(
            "ACMI output path (optional):",
            placeholder="Auto-generated if empty",
            key="bt_acmi",
        )

    return _BtScenarioInputs(
        aircraft=aircraft,
        num_scenarios=bt_num,
        frames=bt_frames,
        seed_start=bt_seed_start,
        acmi_path=bt_acmi,
    )


def _render_bt_launch_controls(
    monitor: TacviewProcessMonitor,
    inputs: _BtScenarioInputs,
) -> None:
    st.markdown("#### Launch")
    if st.button("Generate BT Scenario", type="primary", width="stretch", key="btn_bt"):
        cmd = build_tacview_cmd(
            controller="behavior-tree",
            aircraft=inputs.aircraft,
            frames=inputs.frames,
            num_scenarios=inputs.num_scenarios,
            seed_start=inputs.seed_start,
            acmi_path=inputs.acmi_path or None,
        )
        _launch(f"BT_{inputs.aircraft}_{inputs.num_scenarios}sc", cmd, monitor)

    st.markdown("---")
    st.markdown("#### Quick Launch")
    if st.button("Random BT Scenarios (3x)", width="stretch", key="btn_bt_random"):
        _launch(
            "BT_random_3",
            build_tacview_cmd(
                controller="behavior-tree",
                frames=500,
                num_scenarios=3,
                seed_start=0,
            ),
            monitor,
        )

    if st.button("Single BT Scenario", width="stretch", key="btn_bt_single"):
        _launch(
            "BT_single",
            build_tacview_cmd(controller="behavior-tree", frames=500, num_scenarios=1),
            monitor,
        )


def _render_tacview_output_info() -> None:
    st.markdown("---")
    st.subheader("Tacview Output Information")

    col1, col2 = st.columns(2)
    with col1:
        display_compact_output_paths("tacview")
    with col2:
        display_recent_outputs("tacview")
