"""
Analysis Interface

Interface for analyzing training runs, exporting plots, and launching TensorBoard.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import yaml

from air_to_air_rl.core.paths import project_root, runtime_root
from air_to_air_rl.services import tensorboard as tb_service

from .output_paths import display_compact_output_paths, display_recent_outputs

# Root directory where Ray Tune saves checkpoints/logs — resolved via
# project_root() so it works from any working directory.
MODELS_ROOT = project_root() / "models"


def _tensorboard_processes_file() -> Path:
    return runtime_root() / "gui" / "tensorboard_processes.json"


@dataclass
class TensorboardProcess:
    pid: int
    logdir: str
    port: int
    host: str
    log_file: str
    start_time: str
    label: str
    status: str = "running"  # running | stopped | failed
    exit_code: int | None = None

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class TensorboardProcessMonitor:
    def __init__(self):
        self.processes: list[TensorboardProcess] = []
        self._load()

    def _load(self):
        try:
            if _tensorboard_processes_file().exists():
                with open(_tensorboard_processes_file()) as f:
                    data = json.load(f)
                self.processes = [TensorboardProcess.from_dict(d) for d in data]
        except Exception:
            self.processes = []

    def _save(self):
        _tensorboard_processes_file().parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_tensorboard_processes_file(), "w") as f:
                json.dump([p.to_dict() for p in self.processes], f, indent=2)
        except Exception:
            pass

    def register(self, proc: TensorboardProcess):
        self.processes.append(proc)
        self._save()

    def _is_alive(self, pid: int) -> bool:
        try:
            import psutil

            return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def update_statuses(self):
        changed = False
        for p in self.processes:
            if p.status == "running":
                if not self._is_alive(p.pid):
                    p.status = "stopped"
                    changed = True
        if changed:
            self._save()

    def terminate(self, pid: int):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                import psutil

                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        except Exception:
            pass
        for p in self.processes:
            if p.pid == pid:
                p.status = "stopped"
        self._save()

    def remove(self, pid: int):
        self.processes = [p for p in self.processes if p.pid != pid]
        self._save()

    def render(self):
        self.update_statuses()
        active = [p for p in self.processes if p.status == "running"]
        if not active:
            return
        with st.expander(f"🟢 Active TensorBoard Processes ({len(active)})", expanded=True):
            for proc in active:
                self._render_card(proc)

    def render_all(self):
        """Render all (including stopped) — for the history section."""
        self.update_statuses()
        if not self.processes:
            st.info("No TensorBoard processes launched yet.")
            return
        for proc in reversed(self.processes):
            self._render_card(proc)

    def _render_card(self, proc: TensorboardProcess):
        is_alive = proc.status == "running"
        status_icon = "🟢" if is_alive else "🔴"
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{status_icon} {proc.label}**")
                url = f"http://{proc.host}:{proc.port}"
                st.markdown(f"🌐 [{url}]({url})")
                st.caption(f"PID {proc.pid} · started {proc.start_time}")
            with col2:
                st.caption(f"logdir: `{proc.logdir}`")
            with col3:
                if is_alive:
                    if st.button("⏹ Stop", key=f"tb_stop_{proc.pid}", use_container_width=True):
                        self.terminate(proc.pid)
                        st.rerun()
                else:
                    if st.button("🗑 Remove", key=f"tb_rm_{proc.pid}", use_container_width=True):
                        self.remove(proc.pid)
                        st.rerun()


def find_model_runs():
    """Scan models/ for model directories.

    Delegates to ``services.tensorboard`` for discovery, then augments with
    ``has_events`` / ``num_events`` flags used by the GUI.
    """
    runs = []
    if not MODELS_ROOT.exists():
        return runs
    for model_dir in sorted(MODELS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        event_files = list(model_dir.glob("**/events.out.tfevents.*"))
        runs.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "has_events": bool(event_files),
                "num_events": len(event_files),
            }
        )
    return runs


def _models_root_has_events():
    if not MODELS_ROOT.exists():
        return False
    return bool(list(MODELS_ROOT.glob("**/events.out.tfevents.*")))


def _launch_tensorboard(
    logdir: str, port: int, host: str, label: str, monitor: TensorboardProcessMonitor
):
    """Spawn TensorBoard as a background process and register it."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Build the command via the service layer (not a script path)
    if "," in logdir and ":" in logdir:
        cmd = ["tensorboard", "--logdir_spec", logdir, "--port", str(port), "--host", host]
    else:
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", host]

    try:
        process = subprocess.Popen(cmd, env=env)

        proc = TensorboardProcess(
            pid=process.pid,
            logdir=logdir,
            port=port,
            host=host,
            log_file="",
            start_time=datetime.now().strftime("%H:%M:%S"),
            label=label,
            status="running",
        )
        monitor.register(proc)

        url = f"http://{host}:{port}"
        st.success(f"TensorBoard launched! Open [{url}]({url}) (may take a few seconds to start)")
        st.caption(f"PID {process.pid}")
    except Exception as e:
        st.error(f"Failed to launch TensorBoard: {e}")


def analysis_interface():
    """Render the analysis interface."""
    st.header("📈 Analysis Interface")

    if "tb_monitor" not in st.session_state:
        st.session_state.tb_monitor = TensorboardProcessMonitor()
    monitor: TensorboardProcessMonitor = st.session_state.tb_monitor
    monitor._load()  # refresh from disk each render

    # Active process banner (always visible)
    monitor.render()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "TensorBoard",
            "Export Plots",
            "Run Comparison",
            "Model Evaluation",
        ]
    )

    with tab1:
        st.subheader("🔥 TensorBoard Launch")

        model_runs = find_model_runs()
        models_root_str = str(MODELS_ROOT)

        # Settings in a compact row
        col_port, col_host, _ = st.columns([1, 1, 2])
        with col_port:
            port = st.number_input(
                "Port", value=6006, min_value=1024, max_value=65535, key="tb_port"
            )
        with col_host:
            host = st.text_input("Host", value="localhost", key="tb_host")

        st.markdown("---")

        # --- Launch All ---
        st.markdown("#### Launch All Models")
        col_all, col_all_info = st.columns([1, 3])
        with col_all:
            all_disabled = not _models_root_has_events()
            if st.button(
                "🚀 Launch All",
                type="primary",
                use_container_width=True,
                disabled=all_disabled,
                key="tb_launch_all",
            ):
                _launch_tensorboard(
                    logdir=models_root_str,
                    port=port,
                    host=host,
                    label=f"All models ({models_root_str})",
                    monitor=monitor,
                )
                st.rerun()
        with col_all_info:
            if all_disabled:
                st.warning(f"No TensorBoard event files found in `{models_root_str}`")
            else:
                num = len([r for r in model_runs if r["has_events"]])
                st.info(
                    f"Points TensorBoard at `{models_root_str}` — covers all {num} model(s) with logs"
                )

        st.markdown("---")

        # --- Select specific models ---
        st.markdown("#### Launch Specific Model(s)")

        available = [r for r in model_runs if r["has_events"]]
        all_names = [r["name"] for r in available]

        if not available:
            st.warning(f"No models with TensorBoard event files found in `{models_root_str}`")
            if not MODELS_ROOT.exists():
                st.caption(
                    f"Directory `{models_root_str}` does not exist yet — train a model first."
                )
        else:
            selected_names = st.multiselect(
                "Select model(s):",
                options=all_names,
                help="Select one or more models. Single selection → dedicated logdir; multiple → TensorBoard covers all selected.",
            )

            col_sel, col_sel_info = st.columns([1, 3])
            with col_sel:
                sel_disabled = not selected_names
                if st.button(
                    "📊 Launch Selected",
                    use_container_width=True,
                    disabled=sel_disabled,
                    key="tb_launch_sel",
                ):
                    if len(selected_names) == 1:
                        run = next(r for r in available if r["name"] == selected_names[0])
                        _launch_tensorboard(
                            logdir=run["path"],
                            port=port,
                            host=host,
                            label=selected_names[0],
                            monitor=monitor,
                        )
                    else:
                        # Pass the parent dir — TensorBoard will show each sub-run as a separate experiment
                        # We create a temporary multi-logdir argument using name:path syntax via the script
                        # Simplest approach: point at models root and note which are selected
                        selected_paths = [
                            r["path"] for r in available if r["name"] in selected_names
                        ]
                        logdir_arg = ",".join(
                            f"{n}:{p}" for n, p in zip(selected_names, selected_paths)
                        )
                        _launch_tensorboard(
                            logdir=logdir_arg,
                            port=port,
                            host=host,
                            label=f"{len(selected_names)} models: {', '.join(selected_names)}",
                            monitor=monitor,
                        )
                    st.rerun()
            with col_sel_info:
                if selected_names:
                    st.info(f"Selected: {', '.join(selected_names)}")

            # Table of available models
            st.markdown("#### Available Models")
            rows = []
            for r in model_runs:
                rows.append(
                    {
                        "Model": r["name"],
                        "TB Logs": "✅" if r["has_events"] else "—",
                        "Event files": r["num_events"] if r["has_events"] else 0,
                        "Path": r["path"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Custom path ---
        st.markdown("#### Custom Log Directory")
        col_custom, col_custom_btn = st.columns([3, 1])
        with col_custom:
            custom_logdir = st.text_input(
                "Path:",
                value=models_root_str,
                placeholder="models",
                key="tb_custom_logdir",
            )
        with col_custom_btn:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("📊 Launch", use_container_width=True, key="tb_launch_custom"):
                if Path(custom_logdir).exists():
                    _launch_tensorboard(
                        logdir=custom_logdir,
                        port=port,
                        host=host,
                        label=f"custom: {custom_logdir}",
                        monitor=monitor,
                    )
                    st.rerun()
                else:
                    st.error(f"Path does not exist: {custom_logdir}")

        st.markdown("---")
        st.markdown("#### Process History")
        monitor.render_all()

    with tab2:
        st.subheader("📊 Export Training Plots")
        st.info("Select a model run and export TensorBoard scalar data as plots.")

        model_runs = find_model_runs()
        available = [r for r in model_runs if r["has_events"]]

        col1, col2 = st.columns([2, 1])
        with col1:
            if available:
                selected_export = st.selectbox(
                    "Training run:", options=[r["name"] for r in available]
                )
                run_sel = next(r for r in available if r["name"] == selected_export)
                logdir = run_sel["path"]
                st.info(f"Log directory: `{logdir}`")
            else:
                st.warning("No models with TensorBoard logs found")
                logdir = st.text_input("Manual log directory:", placeholder=models_root_str)

            output_dir = st.text_input("Output directory:", value="exported_plots")
            smoothing = st.slider(
                "Smoothing:", 0.0, 0.99, 0.9, help="Exponential smoothing (0=none, 0.99=heavy)"
            )

        with col2:
            st.markdown("#### Export Actions")
            if st.button("📈 Export Plots", type="primary", use_container_width=True):
                if logdir and Path(logdir).exists():
                    _export_plots(logdir, output_dir, smoothing)
                else:
                    st.error("Select a valid log directory")

    with tab3:
        st.subheader("🔍 Run Comparison")

        model_runs = find_model_runs()
        available = [r for r in model_runs if r["has_events"]]
        all_names = [r["name"] for r in available]

        if len(available) < 2:
            st.warning("Need at least 2 models with TensorBoard logs for comparison")
        else:
            selected = st.multiselect(
                "Select runs to compare:",
                options=all_names,
                default=all_names[:2],
            )

            if len(selected) >= 2:
                col_port2, col_host2, _ = st.columns([1, 1, 2])
                with col_port2:
                    port2 = st.number_input(
                        "Port", value=6007, min_value=1024, max_value=65535, key="tb_cmp_port"
                    )
                with col_host2:
                    host2 = st.text_input("Host", value="localhost", key="tb_cmp_host")

                if st.button(
                    "📊 Launch TensorBoard Comparison", type="primary", use_container_width=True
                ):
                    selected_paths = [r["path"] for r in available if r["name"] in selected]
                    logdir_arg = ",".join(f"{n}:{p}" for n, p in zip(selected, selected_paths))
                    _launch_tensorboard(
                        logdir=logdir_arg,
                        port=port2,
                        host=host2,
                        label=f"Comparison: {', '.join(selected)}",
                        monitor=monitor,
                    )
                    st.rerun()

                rows = [
                    {"Model": n, "Path": next(r["path"] for r in available if r["name"] == n)}
                    for n in selected
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Select at least 2 runs")

    with tab4:
        st.subheader("🎯 Model Evaluation")
        st.info("Model evaluation interface — coming soon!")

        model_runs = find_model_runs()
        if model_runs:
            st.markdown("#### Available Models")
            for r in model_runs:
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.write(f"**{r['name']}**")
                        st.caption(r["path"])
                    with c2:
                        if st.button(
                            "👁 Visualize", key=f"eval_viz_{r['name']}", use_container_width=True
                        ):
                            st.info("Visualization feature coming soon!")
        else:
            st.warning("No trained models found")

    st.markdown("---")
    st.subheader("Analysis Output Information")
    col1, col2 = st.columns(2)
    with col1:
        display_compact_output_paths("analysis")
    with col2:
        display_recent_outputs("analysis")


def _export_plots(logdir, output_dir, smoothing):
    cmd = [
        sys.executable,
        "-m",
        "air_to_air_rl.analysis.export_plots",
        "--logdir",
        logdir,
        "--output-dir",
        output_dir,
        "--smoothing",
        str(smoothing),
    ]
    with st.spinner("Exporting plots..."):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"Plots exported to: `{output_dir}`")
                if st.button("📁 Open export directory"):
                    _open_directory(output_dir)
            else:
                st.error(f"Export failed:\n{result.stderr}")
        except Exception as e:
            st.error(f"Failed to export plots: {e}")


def _open_directory(path):
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        st.error(f"Failed to open directory: {e}")
