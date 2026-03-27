"""
Training Monitor

Real-time monitoring of running training jobs with progress tracking.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import psutil
import streamlit as st

from air_to_air_rl.core.paths import project_root, runtime_root


class TrainingProcess:
    """Represents a running training process with its metadata and progress."""

    def __init__(
        self,
        pid: int,
        command: str,
        config_name: str,
        model_name: str,
        start_time: datetime,
        log_file: str | None = None,
    ):
        self.pid = pid
        self.command = command
        self.config_name = config_name
        self.model_name = model_name
        self.start_time = start_time
        self.log_file = log_file

        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        self.current_episode_reward = 0.0
        self.mean_episode_reward = 0.0
        self.loss = 0.0
        self.learning_rate = 0.0
        self.fps = 0.0
        self.last_update = datetime.now()
        self.is_alive = True
        self.failure_reason: str | None = None
        self.exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pid": self.pid,
            "command": self.command,
            "config_name": self.config_name,
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "log_file": self.log_file,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_episode_reward": self.current_episode_reward,
            "mean_episode_reward": self.mean_episode_reward,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "fps": self.fps,
            "last_update": self.last_update.isoformat(),
            "is_alive": self.is_alive,
            "failure_reason": self.failure_reason,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingProcess":
        """Create from dictionary."""
        process = cls(
            pid=data["pid"],
            command=data["command"],
            config_name=data["config_name"],
            model_name=data["model_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            log_file=data.get("log_file"),
        )
        process.current_step = data.get("current_step", 0)
        process.total_steps = data.get("total_steps", 0)
        process.current_episode_reward = data.get("current_episode_reward", 0.0)
        process.mean_episode_reward = data.get("mean_episode_reward", 0.0)
        process.loss = data.get("loss", 0.0)
        process.learning_rate = data.get("learning_rate", 0.0)
        process.fps = data.get("fps", 0.0)
        process.last_update = datetime.fromisoformat(
            data.get("last_update", datetime.now().isoformat())
        )
        process.is_alive = data.get("is_alive", True)
        process.failure_reason = data.get("failure_reason")
        process.exit_code = data.get("exit_code")
        return process


class TrainingMonitor:
    """Monitors and manages running training processes."""

    def __init__(self):
        self.processes_file = runtime_root() / "gui" / "training_processes.json"
        # Patterns for ProgressCallback output:
        #   Progress: [###---] 20.0% | Iter: 2/10 | Reward: -4.32 | Time: 0.5m
        self.log_patterns = {
            "iter": re.compile(r"Iter:\s*(\d+)/\d+"),
            "total_iter": re.compile(r"Iter:\s*\d+/(\d+)"),
            "reward": re.compile(r"Reward:\s*([-+]?\d*\.?\d+)"),
            "time_total_m": re.compile(r"Time:\s*([\d.]+)m"),
            # Legacy / fallback patterns
            "step": re.compile(r"Step:\s*(\d+)"),
            "episode_reward": re.compile(r"episode_reward_mean:\s*([-+]?\d*\.?\d+)"),
            "loss": re.compile(r"(?:policy_loss|total_loss|loss):\s*([-+]?\d*\.?\d+)"),
            "learning_rate": re.compile(
                r"(?:lr|learning_rate):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            ),
            "fps": re.compile(r"fps:\s*([-+]?\d*\.?\d+)"),
            "timestep": re.compile(r"num_env_steps_sampled_lifetime[/\s:]*(\d+)"),
            "iteration": re.compile(r"training_iteration[/\s:]*(\d+)"),
        }

    def load_processes(self) -> dict[int, TrainingProcess]:
        """Load tracked processes from file."""
        if not self.processes_file.exists():
            return {}

        try:
            with open(self.processes_file) as f:
                data = json.load(f)

            processes = {}
            for pid_str, process_data in data.items():
                processes[int(pid_str)] = TrainingProcess.from_dict(process_data)

            return processes
        except Exception as e:
            st.error(f"Error loading processes: {e}")
            return {}

    def save_processes(self, processes: dict[int, TrainingProcess]):
        """Save tracked processes to file."""
        try:
            self.processes_file.parent.mkdir(parents=True, exist_ok=True)
            data = {str(pid): process.to_dict() for pid, process in processes.items()}
            with open(self.processes_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving processes: {e}")

    def register_process(
        self,
        process: subprocess.Popen,
        config_name: str,
        model_name: str,
        log_file: str | None = None,
    ):
        """Register a new training process for monitoring."""
        training_process = TrainingProcess(
            pid=process.pid,
            command=" ".join(process.args) if hasattr(process, "args") else "Unknown",
            config_name=config_name,
            model_name=model_name,
            start_time=datetime.now(),
            log_file=log_file,
        )

        processes = self.load_processes()
        processes[process.pid] = training_process
        self.save_processes(processes)

        return training_process

    def update_process_status(self) -> dict[int, TrainingProcess]:
        """Update the status of all tracked processes."""
        processes = self.load_processes()
        updated_processes = {}

        for pid, process in processes.items():
            was_alive = process.is_alive
            # Check if process is still alive
            try:
                psutil_process = psutil.Process(pid)
                process.is_alive = psutil_process.is_running()

                if process.is_alive:
                    # Try to parse recent logs for progress
                    self._parse_process_logs(process)
                elif was_alive and process.failure_reason is None:
                    # Process just died — capture exit code and failure reason
                    try:
                        process.exit_code = psutil_process.wait(timeout=1)
                    except Exception:
                        pass
                    process.failure_reason = self._capture_failure_reason(process)
            except psutil.NoSuchProcess:
                process.is_alive = False
                if was_alive and process.failure_reason is None:
                    process.failure_reason = self._capture_failure_reason(process)
            except Exception as e:
                st.warning(f"Error checking process {pid}: {e}")
                process.is_alive = False

            updated_processes[pid] = process

        self.save_processes(updated_processes)
        return updated_processes

    def _capture_failure_reason(self, process: TrainingProcess) -> str | None:
        """Read the tail of the log file and return the last lines as the failure reason."""
        if not process.log_file:
            return None
        log_path = Path(process.log_file)
        if not log_path.exists():
            return None
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail = "".join(lines[-100:]).strip()
            return tail if tail else None
        except Exception:
            return None

    def _parse_process_logs(self, process: TrainingProcess):
        """Parse training logs to extract progress information."""
        # Try to find log files in common locations
        log_paths = []

        if process.log_file:
            log_paths.append(Path(process.log_file))

        # Common log locations (resolved relative to project root, not CWD)
        _root = project_root()
        log_paths.extend(
            [
                _root / "logs" / f"{process.model_name}.log",
                _root / "logs" / "training" / f"{process.model_name}.log",
                _root / "outputs" / process.model_name / "training.log",
                _root / "logs" / "training.log",
            ]
        )

        for log_path in log_paths:
            if log_path.exists():
                self._parse_log_file(process, log_path)
                break

    def _parse_log_file(self, process: TrainingProcess, log_path: Path):
        """Parse a specific log file for training progress.

        ProgressCallback writes lines with \\r (carriage return) so we split
        on both \\r and \\n to get individual progress entries.
        """
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Split on both \r and \n so carriage-return progress lines are
            # treated as separate entries. Take the last 200 segments.
            segments = [s for s in re.split(r"[\r\n]", content) if s.strip()]
            recent = segments[-200:]

            for line in recent:
                # ProgressCallback: Iter: 3/100
                m = self.log_patterns["iter"].search(line)
                if m:
                    process.current_step = int(m.group(1))

                # ProgressCallback: auto-fill total from log if not set
                m = self.log_patterns["total_iter"].search(line)
                if m and process.total_steps == 0:
                    process.total_steps = int(m.group(1))

                # ProgressCallback: Reward: -4.32
                m = self.log_patterns["reward"].search(line)
                if m:
                    process.mean_episode_reward = float(m.group(1))

                # Legacy / alternative patterns
                m = self.log_patterns["step"].search(line)
                if m:
                    process.current_step = int(m.group(1))

                m = self.log_patterns["timestep"].search(line)
                if m:
                    process.current_step = int(m.group(1))

                m = self.log_patterns["iteration"].search(line)
                if m:
                    process.current_step = int(m.group(1))

                m = self.log_patterns["episode_reward"].search(line)
                if m:
                    process.mean_episode_reward = float(m.group(1))

                m = self.log_patterns["loss"].search(line)
                if m:
                    process.loss = float(m.group(1))

                m = self.log_patterns["learning_rate"].search(line)
                if m:
                    process.learning_rate = float(m.group(1))

                m = self.log_patterns["fps"].search(line)
                if m:
                    process.fps = float(m.group(1))

            process.last_update = datetime.now()

        except Exception:
            pass

    def terminate_process(self, pid: int) -> bool:
        """Terminate a training process and its entire process tree.

        Uses platform-specific methods to kill all descendants, including Ray
        workers that may not appear as direct children of the launcher PID.
        On Windows: taskkill /F /T is the only reliable way to kill a full tree.
        On Unix: SIGKILL the process group.
        """
        try:
            if os.name == "nt":
                # Windows: taskkill /F (force) /T (tree) kills all descendants
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                )
            else:
                try:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    for proc in [parent] + children:
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass

            # Mark process as dead and capture final log output
            processes = self.load_processes()
            if pid in processes:
                proc_record = processes[pid]
                proc_record.is_alive = False
                if proc_record.failure_reason is None:
                    proc_record.failure_reason = self._capture_failure_reason(proc_record)
                self.save_processes(processes)

            return True
        except Exception as e:
            st.error(f"Error terminating process {pid}: {e}")
            return False

    def cleanup_dead_processes(self):
        """Remove dead processes from tracking."""
        processes = self.load_processes()
        alive_processes = {pid: proc for pid, proc in processes.items() if proc.is_alive}
        self.save_processes(alive_processes)


def render_training_monitor():
    """Render the training monitoring interface."""
    st.subheader("📊 Active Training Monitor")

    # Initialize monitor
    if "training_monitor" not in st.session_state:
        st.session_state.training_monitor = TrainingMonitor()

    monitor = st.session_state.training_monitor

    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        auto_refresh = st.checkbox(
            "Auto-refresh every 5 seconds",
            value=st.session_state.get("auto_refresh_training", False),
            key="auto_refresh_training",
        )

    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("🧹 Cleanup Dead", use_container_width=True):
            monitor.cleanup_dead_processes()
            st.success("Cleaned up dead processes")
            st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        # Display auto-refresh status
        st.caption("🔄 Auto-refresh enabled (every 5 seconds)")

        # Use st.rerun() with a delay mechanism
        time.sleep(1)
        st.rerun()

    # Get current processes
    processes = monitor.update_process_status()

    if not processes:
        st.info("📭 No active training processes found.")
        return

    # Display active processes
    alive_processes = {pid: proc for pid, proc in processes.items() if proc.is_alive}
    dead_processes = {pid: proc for pid, proc in processes.items() if not proc.is_alive}

    if alive_processes:
        st.markdown("### 🟢 Active Training Jobs")

        for pid, process in alive_processes.items():
            render_process_card(process, monitor)

    if dead_processes:
        with st.expander(f"💀 Recently Completed/Failed Jobs ({len(dead_processes)})"):
            for pid, process in dead_processes.items():
                render_process_card(process, monitor, is_dead=True)


def render_process_card(process: TrainingProcess, monitor: TrainingMonitor, is_dead: bool = False):
    """Render a card for a single training process."""
    # Calculate runtime
    runtime = datetime.now() - process.start_time
    runtime_str = str(runtime).split(".")[0]  # Remove microseconds

    # Status indicator
    status_color = "🟢" if process.is_alive else "🔴"
    status_text = "Running" if process.is_alive else "Stopped"

    with st.container():
        # Create a border using columns and styling
        st.markdown(
            f"""
        <div style="
            border: 1px solid {"#28a745" if process.is_alive else "#dc3545"};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: {"rgba(40, 167, 69, 0.05)" if process.is_alive else "rgba(220, 53, 69, 0.05)"};
        ">
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Process header
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{status_color} {process.model_name}** ({status_text})")
            st.caption(
                f"PID: {process.pid} | Config: {process.config_name} | Runtime: {runtime_str}"
            )

        with col2:
            if not is_dead and process.is_alive:
                if st.button(
                    "⏹️ Stop",
                    key=f"stop_{process.pid}",
                    help="Terminates the training process and all child processes (Ray workers etc.)",
                ):
                    with st.spinner("Stopping all processes in tree…"):
                        if monitor.terminate_process(process.pid):
                            st.success(f"Stopped job {process.pid}")
                            st.rerun()
                        else:
                            st.error(f"Failed to stop job {process.pid}")

        with col3:
            # Progress percentage
            if process.total_steps > 0:
                progress = min(100, (process.current_step / process.total_steps) * 100)
                st.metric("Progress", f"{progress:.1f}%")
            else:
                st.metric("Step", f"{process.current_step:,}")

        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if process.mean_episode_reward != 0:
                st.metric(
                    "Mean Reward",
                    f"{process.mean_episode_reward:.2f}",
                    help="Average episode reward",
                )
            else:
                st.metric("Mean Reward", "N/A")

        with col2:
            if process.loss != 0:
                st.metric("Loss", f"{process.loss:.4f}", help="Current training loss")
            else:
                st.metric("Loss", "N/A")

        with col3:
            if process.learning_rate != 0:
                st.metric(
                    "Learning Rate", f"{process.learning_rate:.2e}", help="Current learning rate"
                )
            else:
                st.metric("Learning Rate", "N/A")

        with col4:
            if process.fps > 0:
                st.metric("FPS", f"{process.fps:.1f}", help="Training frames per second")
            else:
                st.metric("FPS", "N/A")

        # Progress bar
        if process.total_steps > 0:
            progress = min(1.0, process.current_step / process.total_steps)
            st.progress(progress)
            st.caption(f"Step {process.current_step:,} / {process.total_steps:,}")
        elif process.current_step > 0:
            st.caption(f"Step {process.current_step:,}")

        # Last update time
        if process.last_update:
            time_since_update = datetime.now() - process.last_update
            if time_since_update.total_seconds() < 60:
                update_str = f"{int(time_since_update.total_seconds())}s ago"
            else:
                update_str = f"{int(time_since_update.total_seconds() // 60)}m ago"
            st.caption(f"Last update: {update_str}")

        # Log file controls
        if process.log_file:
            log_path = Path(process.log_file)
            col_log1, col_log2 = st.columns([3, 1])
            with col_log1:
                st.caption(f"📄 Log: `{process.log_file}`")
            with col_log2:
                if st.button(
                    "📂 Open Log",
                    key=f"open_log_{process.pid}",
                    use_container_width=True,
                    help="Open log file in system editor",
                ):
                    if log_path.exists():
                        if os.name == "nt":
                            os.startfile(str(log_path))
                        else:
                            import subprocess as _sp

                            _sp.Popen(["xdg-open", str(log_path)])
                    else:
                        st.warning("Log file not found yet.")

            # Live log tail — auto-expanded when job is dead so output is immediately visible
            if log_path.exists():
                with st.expander("📋 Log Output (last 30 lines)", expanded=is_dead):
                    try:
                        with open(log_path, encoding="utf-8", errors="ignore") as f:
                            raw = f.read()
                        # Replace \r with \n so carriage-return lines display clearly
                        cleaned = raw.replace("\r", "\n")
                        lines = [ln for ln in cleaned.splitlines() if ln.strip()]
                        tail = "\n".join(lines[-30:])
                        st.code(tail, language="text")
                    except Exception as e:
                        st.warning(f"Could not read log: {e}")

        # Command details (expandable)
        with st.expander("🔍 Command Details"):
            st.code(process.command, language="bash")

        # Failure reason (dead processes only)
        if is_dead and process.failure_reason and not process.log_file:
            exit_label = (
                f" (exit code {process.exit_code})" if process.exit_code is not None else ""
            )
            with st.expander(f"❌ Failure Output{exit_label}", expanded=True):
                st.code(process.failure_reason, language="text")
