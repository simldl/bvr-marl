"""
Analysis Interface

Interface for analyzing training runs, exporting plots, and launching TensorBoard.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from bvr_marl_core.gui.components.output_paths import (
    display_compact_output_paths,
    display_recent_outputs,
)
from bvr_marl_core.utils.paths import core_runtime_root as runtime_root
from bvr_marl_core.utils.paths import exported_plots_root, models_root, sibling_model_roots

# Root directory where Ray Tune saves checkpoints/logs — resolved via
# models_root() so it works from any working directory.
MODELS_ROOT = models_root()


def _tensorboard_processes_file() -> Path:
    return runtime_root() / "gui" / "tensorboard_processes.json"


def _campaign_paths_file() -> Path:
    return runtime_root() / "gui" / "campaign_paths.json"


def _load_campaign_paths() -> list[dict]:
    """Load persisted campaign paths from disk."""
    try:
        f = _campaign_paths_file()
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return []


def _save_campaign_paths(paths: list[dict]) -> None:
    """Persist campaign paths to disk."""
    try:
        f = _campaign_paths_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def _candidate_model_roots() -> list[Path]:
    """Return model roots worth scanning for GUI analysis runs."""
    roots = [MODELS_ROOT]
    for sibling in sibling_model_roots():
        if sibling not in roots:
            roots.append(sibling)
    return [root for root in roots if root.exists()]


def _contains_event_files(path: Path) -> bool:
    return bool(list(path.glob("**/events.out.tfevents.*")))


def _campaign_run_dirs(campaign_dir: Path) -> list[Path]:
    return sorted(path.parent for path in campaign_dir.glob("*/run_manifest.json"))


def _campaign_sort_time(campaign: dict) -> float:
    run_dirs = campaign.get("runs", [])
    mtimes = []
    for run in run_dirs:
        manifest = Path(run["path"]) / "run_manifest.json"
        try:
            mtimes.append(manifest.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _latest_campaign_path_entry(campaigns: dict[str, dict]) -> dict | None:
    """Return a persisted-path entry for the newest discovered campaign."""
    if not campaigns:
        return None
    _cid, campaign = max(campaigns.items(), key=lambda item: _campaign_sort_time(item[1]))
    path = str(campaign.get("path") or "").strip()
    if not path:
        return None
    return {"label": f"campaign_{campaign['name']}", "path": path, "auto": True}


def _ensure_latest_campaign_path_seeded(campaigns: dict[str, dict]) -> None:
    """Pre-populate custom/campaign paths with the latest campaign when empty."""
    if st.session_state.get("export_auto_campaign_seeded"):
        return
    st.session_state.export_auto_campaign_seeded = True
    if st.session_state.export_custom_paths:
        return
    entry = _latest_campaign_path_entry(campaigns)
    if entry is None:
        return
    st.session_state.export_custom_paths = [entry]
    _save_campaign_paths(st.session_state.export_custom_paths)


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
        except (OSError, TypeError, ValueError):
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
        except Exception:  # noqa: BLE001 - psutil is imported inside this try, so psutil.Error cannot be named here; killing an already-dead process tree is expected.
            pass
        for p in self.processes:
            if p.pid == pid:
                p.status = "stopped"
        self._save()

    def remove(self, pid: int):
        self.processes = [p for p in self.processes if p.pid != pid]
        self._save()

    def ports_in_use(self) -> set[int]:
        """Return ports held by currently running registered processes."""
        return {p.port for p in self.processes if p.status == "running"}

    def render(self):
        self.update_statuses()
        active = [p for p in self.processes if p.status == "running"]
        if not active:
            return
        with st.expander(f"Active TensorBoard Processes ({len(active)})", expanded=True):
            for idx, proc in enumerate(active):
                self._render_card(proc, idx, ns="a")

    def render_all(self):
        """Render all (including stopped) — for the history section."""
        self.update_statuses()
        if not self.processes:
            st.info("No TensorBoard processes launched yet.")
            return
        for idx, proc in enumerate(reversed(self.processes)):
            self._render_card(proc, idx, ns="h")

    def _render_card(self, proc: TensorboardProcess, idx: int = 0, ns: str = ""):
        is_alive = proc.status == "running"
        status_label = "Running" if is_alive else "Stopped"
        suffix = f"_{ns}_{idx}"
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{proc.label}**")
                url = f"http://{proc.host}:{proc.port}"
                st.markdown(f"[{url}]({url})")
                st.caption(
                    f"Status: {status_label} | PID {proc.pid} | Port {proc.port} | Started {proc.start_time}"
                )
            with col2:
                st.caption(f"logdir: `{proc.logdir}`")
            with col3:
                if is_alive:
                    if st.button("Stop", key=f"tb_stop_{proc.pid}{suffix}", width="stretch"):
                        self.terminate(proc.pid)
                        st.rerun()
                else:
                    if st.button("Remove", key=f"tb_rm_{proc.pid}{suffix}", width="stretch"):
                        self.remove(proc.pid)
                        st.rerun()


def find_campaigns(models_root: Path | None = None) -> dict[str, dict]:
    """
    Scan models_root for runs with a campaign_id in run_manifest.json.

    Returns:
        {campaign_id: {"name": str, "path": str, "runs": [{"name": str, "path": str}]}}
        Only includes runs that have TensorBoard event files.
    """
    roots = [models_root] if models_root is not None else _candidate_model_roots()
    campaigns: dict[str, dict] = {}
    for root in roots:
        if root is None or not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue

            manifest = model_dir / "run_manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    campaign_id = data.get("campaign_id")
                    if not campaign_id or not _contains_event_files(model_dir):
                        continue
                    campaign_name = data.get("campaign_name") or campaign_id
                    if campaign_id not in campaigns:
                        campaigns[campaign_id] = {
                            "name": campaign_name,
                            "path": str(model_dir.parent),
                            "runs": [],
                        }
                    campaigns[campaign_id]["runs"].append(
                        {"name": model_dir.name, "path": str(model_dir)}
                    )
                except (OSError, AttributeError, TypeError, ValueError, KeyError, IndexError):
                    continue
                continue

            run_dirs = _campaign_run_dirs(model_dir)
            runs_with_events = [run_dir for run_dir in run_dirs if _contains_event_files(run_dir)]
            if not runs_with_events:
                continue
            campaign_id = model_dir.name
            if campaign_id not in campaigns:
                campaigns[campaign_id] = {
                    "name": campaign_id,
                    "path": str(model_dir),
                    "runs": [],
                }
            for run_dir in runs_with_events:
                campaigns[campaign_id]["runs"].append(
                    {
                        "name": f"{campaign_id}/{run_dir.name}",
                        "path": str(run_dir),
                    }
                )
    return campaigns


def find_model_runs():
    """Scan models/ for model directories.

    Delegates to ``services.tensorboard`` for discovery, then augments with
    ``has_events`` / ``num_events`` flags used by the GUI.
    """
    runs = []
    seen_paths: set[str] = set()
    for root in _candidate_model_roots():
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            run_dirs = _campaign_run_dirs(model_dir)
            scan_dirs = run_dirs if run_dirs else [model_dir]
            for run_dir in scan_dirs:
                resolved = str(run_dir.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                event_files = list(run_dir.glob("**/events.out.tfevents.*"))
                name = run_dir.name if run_dir == model_dir else f"{model_dir.name}/{run_dir.name}"
                runs.append(
                    {
                        "name": name,
                        "path": str(run_dir),
                        "has_events": bool(event_files),
                        "num_events": len(event_files),
                    }
                )
    return runs


def _model_run_widget_key(prefix: str, run: dict) -> str:
    """Build a stable Streamlit key for a model run using its unique path."""
    source = str(Path(run["path"]).resolve())
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _models_root_has_events():
    return any(_contains_event_files(root) for root in _candidate_model_roots())


def _find_free_port(start: int, host: str, exclude: set[int]) -> int:
    """Return the first free TCP port at or above *start* that is not in *exclude*."""
    import socket

    port = start
    while port < 65535:
        if port not in exclude:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, port))
                    return port
                except OSError:
                    pass
        port += 1
    return start


def _launch_tensorboard(
    logdir: str, port: int, host: str, label: str, monitor: TensorboardProcessMonitor
):
    """Spawn TensorBoard as a background process and register it."""
    # Auto-select a free port so a second launch never collides with an existing one.
    free_port = _find_free_port(port, host, monitor.ports_in_use())
    if free_port != port:
        st.info(f"Port {port} is in use — using port {free_port} instead.")
    port = free_port

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # --samples_per_plugin limits scalars to 2000 points per metric so large
    # event files load in seconds, not minutes.
    # --reload_interval reduces how often TensorBoard re-reads the files.
    extra = ["--samples_per_plugin", "scalars=2000", "--reload_interval", "120"]
    if "," in logdir and ":" in logdir:
        cmd = ["tensorboard", "--logdir_spec", logdir, "--port", str(port), "--host", host] + extra
    else:
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", host] + extra

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
        st.success(
            f"TensorBoard launched on port {port}! Open [{url}]({url}) (may take a few seconds to start)"
        )
        st.caption(f"PID {process.pid}")
    except Exception as e:
        st.error(f"Failed to launch TensorBoard: {e}")


def analysis_interface():
    """Render the analysis interface."""
    st.header("Analysis Tools")
    st.caption("Launch TensorBoard, export plots, compare runs, and review model evaluation state.")

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
        _render_tensorboard_tab(monitor)

    with tab2:
        _render_export_plots_tab()

    with tab3:
        _render_run_comparison_tab(monitor)

    with tab4:
        _render_model_evaluation_tab()

    _render_analysis_output_info()


def _render_tensorboard_tab(monitor: TensorboardProcessMonitor) -> None:
    st.subheader("TensorBoard Launch")

    model_runs = find_model_runs()
    models_root_str = str(MODELS_ROOT)
    port, host = _render_tensorboard_settings()

    st.markdown("---")
    _render_launch_all_models(model_runs, models_root_str, port, host, monitor)

    st.markdown("---")
    _render_launch_selected_models(model_runs, models_root_str, port, host, monitor)

    st.markdown("---")
    _render_custom_tensorboard_launcher(models_root_str, port, host, monitor)

    st.markdown("---")
    st.markdown("#### Process History")
    monitor.render_all()


def _render_tensorboard_settings() -> tuple[int, str]:
    col_port, col_host, _ = st.columns([1, 1, 2])
    with col_port:
        port = st.number_input("Port", value=6006, min_value=1024, max_value=65535, key="tb_port")
    with col_host:
        host = st.text_input("Host", value="localhost", key="tb_host")
    return int(port), host


def _render_launch_all_models(
    model_runs: list[dict],
    models_root_str: str,
    port: int,
    host: str,
    monitor: TensorboardProcessMonitor,
) -> None:
    st.markdown("#### Launch All Models")
    col_all, col_all_info = st.columns([1, 3])
    with col_all:
        all_disabled = not _models_root_has_events()
        if st.button(
            "Launch All",
            type="primary",
            width="stretch",
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
            num = len([run for run in model_runs if run["has_events"]])
            st.info(
                f"Points TensorBoard at `{models_root_str}` and covers all {num} model(s) with logs."
            )


def _render_launch_selected_models(
    model_runs: list[dict],
    models_root_str: str,
    port: int,
    host: str,
    monitor: TensorboardProcessMonitor,
) -> None:
    st.markdown("#### Launch Specific Model(s)")

    available = [run for run in model_runs if run["has_events"]]
    all_names = [run["name"] for run in available]

    if not available:
        st.warning(f"No models with TensorBoard event files found in `{models_root_str}`")
        if not MODELS_ROOT.exists():
            st.caption(f"Directory `{models_root_str}` does not exist yet — train a model first.")
        return

    selected_names = st.multiselect(
        "Select model(s):",
        options=all_names,
        help="Select one or more models. Single selection uses a dedicated logdir; multiple selections launch a combined TensorBoard view.",
    )

    col_sel, col_sel_info = st.columns([1, 3])
    with col_sel:
        if st.button(
            "Launch Selected",
            width="stretch",
            disabled=not selected_names,
            key="tb_launch_sel",
        ):
            _launch_selected_tensorboard_runs(selected_names, available, port, host, monitor)
            st.rerun()
    with col_sel_info:
        if selected_names:
            st.info(f"Selected: {', '.join(selected_names)}")

    _render_available_model_table(model_runs)


def _launch_selected_tensorboard_runs(
    selected_names: list[str],
    available: list[dict],
    port: int,
    host: str,
    monitor: TensorboardProcessMonitor,
) -> None:
    if len(selected_names) == 1:
        run = next(run for run in available if run["name"] == selected_names[0])
        _launch_tensorboard(
            logdir=run["path"],
            port=port,
            host=host,
            label=selected_names[0],
            monitor=monitor,
        )
        return

    selected_paths = [run["path"] for run in available if run["name"] in selected_names]
    logdir_arg = ",".join(f"{name}:{path}" for name, path in zip(selected_names, selected_paths))
    _launch_tensorboard(
        logdir=logdir_arg,
        port=port,
        host=host,
        label=f"{len(selected_names)} models: {', '.join(selected_names)}",
        monitor=monitor,
    )


def _render_available_model_table(model_runs: list[dict]) -> None:
    st.markdown("#### Available Models")
    rows = [
        {
            "Model": run["name"],
            "TB Logs": "Available" if run["has_events"] else "Not found",
            "Event files": run["num_events"] if run["has_events"] else 0,
            "Path": run["path"],
        }
        for run in model_runs
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_custom_tensorboard_launcher(
    models_root_str: str,
    port: int,
    host: str,
    monitor: TensorboardProcessMonitor,
) -> None:
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
        if st.button("Launch", width="stretch", key="tb_launch_custom"):
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


def _render_export_plots_tab() -> None:
    st.subheader("Export Training Plots")
    st.info(
        "Select runs to compare on the same plots. "
        "Exports: mean return, kills, deaths, missiles fired, win rate."
    )

    _ensure_export_session_state()
    model_runs = find_model_runs()
    available = [run for run in model_runs if run["has_events"]]
    campaigns = find_campaigns()
    _ensure_latest_campaign_path_seeded(campaigns)

    col_setup, col_actions = st.columns([2, 1])
    with col_setup:
        _render_export_campaign_selector(campaigns)
        selected_exports = _render_export_run_selector(available)
        _render_export_custom_paths()
        output_dir, smoothing = _render_export_settings()

    with col_actions:
        run_dirs = _build_export_run_dirs(selected_exports, available)
        _render_export_actions(run_dirs, output_dir, smoothing)


def _ensure_export_session_state() -> None:
    if "export_custom_paths" not in st.session_state:
        st.session_state.export_custom_paths = _load_campaign_paths()
    if "export_campaign_runs" not in st.session_state:
        st.session_state.export_campaign_runs = []


def _render_export_campaign_selector(campaigns: dict[str, dict]) -> None:
    if not campaigns:
        return

    campaign_options = {
        f"{campaign['name']}  ({len(campaign['runs'])} models)": cid
        for cid, campaign in sorted(campaigns.items(), key=lambda item: item[1]["name"])
    }
    with st.expander("Select by campaign", expanded=bool(st.session_state.export_campaign_runs)):
        chosen_label = st.selectbox(
            "Campaign:",
            options=[""] + list(campaign_options.keys()),
            format_func=lambda label: "— choose a campaign —" if label == "" else label,
            key="ep_campaign_select",
        )
        if chosen_label:
            cid = campaign_options[chosen_label]
            run_names_in_campaign = [run["name"] for run in campaigns[cid]["runs"]]
            st.caption(
                f"{len(run_names_in_campaign)} model(s): "
                f"{', '.join(_short_run_name(name) for name in run_names_in_campaign)}"
            )
            if st.button("Load all models from this campaign", key="ep_load_campaign"):
                st.session_state.export_campaign_runs = run_names_in_campaign
                st.rerun()
        if st.session_state.export_campaign_runs:
            if st.button("Clear campaign selection", key="ep_clear_campaign"):
                st.session_state.export_campaign_runs = []
                st.rerun()


def _short_run_name(name: str) -> str:
    return f"{name[:19]}…" if len(name) > 20 else name


def _render_export_run_selector(available: list[dict]) -> list[str]:
    available_names = [run["name"] for run in available]
    if not available:
        st.warning(f"No models with TensorBoard logs found in `{MODELS_ROOT}`")
        return []

    default_selection = [
        name for name in st.session_state.export_campaign_runs if name in available_names
    ]
    return st.multiselect(
        "Training run(s):",
        options=available_names,
        default=default_selection,
        help="Select one or more runs to compare on the same plots. Use the campaign selector above to bulk-load a campaign.",
    )


def _render_export_custom_paths() -> None:
    with st.expander(
        "Add custom / campaign paths", expanded=bool(st.session_state.export_custom_paths)
    ):
        add_col1, add_col2, add_col3 = st.columns([2, 4, 1])
        with add_col1:
            new_label = st.text_input("Label", key="ep_new_label", placeholder="my_run")
        with add_col2:
            new_path = st.text_input("Path", key="ep_new_path", placeholder="C:/path/to/logs")
        with add_col3:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("Add", key="ep_add_path", width="stretch"):
                _add_export_custom_path(new_label, new_path)

        _render_export_custom_path_list()


def _add_export_custom_path(new_label: str, new_path: str) -> None:
    label = new_label.strip() or Path(new_path.strip()).name
    path = new_path.strip()
    if path and Path(path).exists():
        st.session_state.export_custom_paths.append({"label": label, "path": path})
        _save_campaign_paths(st.session_state.export_custom_paths)
        st.rerun()
    elif path:
        st.error(f"Path does not exist: {path}")
    else:
        st.error("Enter a path")


def _render_export_custom_path_list() -> None:
    if not st.session_state.export_custom_paths:
        return

    st.markdown("**Added paths:**")
    for idx, entry in enumerate(list(st.session_state.export_custom_paths)):
        c1, c2, c3 = st.columns([2, 4, 1])
        with c1:
            st.markdown(f"`{entry['label']}`")
        with c2:
            st.caption(entry["path"])
        with c3:
            if st.button("Remove", key=f"ep_rm_{idx}", width="stretch"):
                st.session_state.export_custom_paths.pop(idx)
                _save_campaign_paths(st.session_state.export_custom_paths)
                st.rerun()


def _render_export_settings() -> tuple[str, float]:
    output_dir = st.text_input("Output directory:", value=str(exported_plots_root()))
    smoothing = st.slider(
        "Smoothing:", 0.0, 0.99, 0.9, help="Exponential smoothing (0=none, 0.99=heavy)"
    )
    return output_dir, float(smoothing)


def _build_export_run_dirs(selected_exports: list[str], available: list[dict]) -> dict[str, str]:
    run_dirs: dict[str, str] = {}
    for name in selected_exports:
        run = next(run for run in available if run["name"] == name)
        run_dirs[name] = run["path"]
    for entry in st.session_state.export_custom_paths:
        run_dirs[entry["label"]] = entry["path"]
    return run_dirs


def _render_export_actions(
    run_dirs: dict[str, str],
    output_dir: str,
    smoothing: float,
) -> None:
    st.markdown("#### Export Actions")
    export_ready = bool(run_dirs)
    if st.button(
        "Export Plots",
        type="primary",
        width="stretch",
        disabled=not export_ready,
    ):
        if run_dirs:
            _export_plots(run_dirs, output_dir, smoothing)
        else:
            st.error("Select at least one run or add a custom path")

    # This button must live at the top level of the tab. A Streamlit button only
    # returns True on the rerun immediately after its own click, so a button
    # rendered inside the export branch disappears before the user can click it.
    last_export_dir = st.session_state.get("analysis_last_export_dir")
    if last_export_dir:
        st.caption(f"Last export: `{last_export_dir}`")
        if st.button("Open export directory", width="stretch", key="open_export_dir"):
            _open_directory(last_export_dir)


def _render_run_comparison_tab(monitor: TensorboardProcessMonitor) -> None:
    st.subheader("Run Comparison")

    model_runs = find_model_runs()
    available = [run for run in model_runs if run["has_events"]]
    all_names = [run["name"] for run in available]

    if len(available) < 2:
        st.warning("Need at least 2 models with TensorBoard logs for comparison")
        return

    selected = st.multiselect(
        "Select runs to compare:",
        options=all_names,
        default=all_names[:2],
    )

    if len(selected) < 2:
        st.info("Select at least 2 runs")
        return

    port, host = _render_comparison_tensorboard_settings()
    if st.button("Launch TensorBoard Comparison", type="primary", width="stretch"):
        _launch_tensorboard_comparison(selected, available, port, host, monitor)
        st.rerun()

    _render_selected_comparison_table(selected, available)


def _render_comparison_tensorboard_settings() -> tuple[int, str]:
    col_port, col_host, _ = st.columns([1, 1, 2])
    with col_port:
        port = st.number_input(
            "Port", value=6007, min_value=1024, max_value=65535, key="tb_cmp_port"
        )
    with col_host:
        host = st.text_input("Host", value="localhost", key="tb_cmp_host")
    return int(port), host


def _launch_tensorboard_comparison(
    selected: list[str],
    available: list[dict],
    port: int,
    host: str,
    monitor: TensorboardProcessMonitor,
) -> None:
    selected_paths = [run["path"] for run in available if run["name"] in selected]
    logdir_arg = ",".join(f"{name}:{path}" for name, path in zip(selected, selected_paths))
    _launch_tensorboard(
        logdir=logdir_arg,
        port=port,
        host=host,
        label=f"Comparison: {', '.join(selected)}",
        monitor=monitor,
    )


def _render_selected_comparison_table(selected: list[str], available: list[dict]) -> None:
    rows = [
        {
            "Model": name,
            "Path": next(run["path"] for run in available if run["name"] == name),
        }
        for name in selected
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_model_evaluation_tab() -> None:
    st.subheader("Model Evaluation")
    st.info("Model evaluation interface coming soon.")

    model_runs = find_model_runs()
    if not model_runs:
        st.warning("No trained models found")
        return

    st.markdown("#### Available Models")
    for run in model_runs:
        _render_evaluation_model_row(run)


def _render_evaluation_model_row(run: dict) -> None:
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{run['name']}**")
            st.caption(run["path"])
        with c2:
            if st.button(
                "Visualize",
                key=_model_run_widget_key("eval_viz", run),
                width="stretch",
            ):
                st.info("Visualization feature coming soon!")


def _render_analysis_output_info() -> None:
    st.markdown("---")
    st.subheader("Analysis Output Information")
    col1, col2 = st.columns(2)
    with col1:
        display_compact_output_paths("analysis")
    with col2:
        display_recent_outputs("analysis")


def _export_plots(run_dirs: dict[str, str], output_dir: str, smoothing: float):
    """Run export_plots as a subprocess, passing all runs as resolved name:path pairs."""
    run_path_args = [f"{name}:{path}" for name, path in run_dirs.items()]
    cmd = [
        sys.executable,
        "-m",
        "bvr_marl_core.analysis.export_plots",
        "--run-paths",
        *run_path_args,
        "--output-dir",
        output_dir,
        "--smoothing",
        str(smoothing),
    ]
    with st.spinner("Exporting plots..."):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                resolved = str(Path(output_dir).resolve())
                # Persist so the top-level "Open export directory" button (which
                # survives the next rerun) can act on it. Rendering the button
                # here would not work: it vanishes before the user can click it.
                st.session_state["analysis_last_export_dir"] = resolved
                st.success(f"Plots exported to: `{resolved}`")
            else:
                st.error(f"Export failed:\n{result.stderr}")
        except Exception as e:
            st.error(f"Failed to export plots: {e}")


def _open_directory(path):
    """Open *path* in the OS file browser (server-side / local Streamlit host)."""
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"Could not create directory `{target}`: {exc}")
        return

    target_str = str(target.resolve())
    try:
        if os.name == "nt":
            os.startfile(target_str)  # noqa: S606 — local desktop convenience
        elif sys.platform == "darwin":
            subprocess.run(["open", target_str], check=True)
        else:
            subprocess.run(["xdg-open", target_str], check=True)
        st.success(f"Opened `{target_str}`.")
    except Exception as e:
        st.error(f"Failed to open directory `{target_str}`: {e}")
