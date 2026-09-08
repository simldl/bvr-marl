"""
TensorBoard service — log-directory discovery and launch helpers.

Used by both ``scripts/analysis/launch_tensorboard.py`` and the GUI
analysis interface.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from bvr_marl_core.utils.paths import core_project_root as project_root


def find_log_directories() -> list[str]:
    """Return sorted TensorBoard log directories found under the project root."""
    log_dirs: list[str] = []
    root = project_root()

    # Primary location: models/ at project root
    models_root = root / "models"
    if models_root.exists():
        event_files = list(models_root.glob("**/events.out.tfevents.*"))
        if event_files:
            log_dirs.append(str(models_root))

    # Legacy / fallback patterns relative to project root
    search_patterns = [
        "outputs/*/logs",
        "outputs/*/*/logs",
        "logs",
        "tensorboard_logs",
        "runs",
    ]
    for pattern in search_patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                event_files = list(path.glob("**/events.out.tfevents.*"))
                if event_files:
                    log_dirs.append(str(path))

    return sorted(set(log_dirs))


def find_training_runs() -> list[dict]:
    """Return dicts describing available training runs with TensorBoard data.

    Each dict has keys ``name``, ``path`` (log dir), ``run_dir``.
    """
    run_dirs: list[dict] = []
    root = project_root()

    models_root = root / "models"
    if models_root.exists():
        for model_dir in sorted(models_root.iterdir()):
            if model_dir.is_dir():
                event_files = list(model_dir.glob("**/events.out.tfevents.*"))
                if event_files:
                    run_dirs.append(
                        {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "run_dir": str(model_dir),
                        }
                    )

    # Also check legacy outputs/ directory
    outputs_dir = root / "outputs"
    if outputs_dir.exists():
        for run_dir in sorted(outputs_dir.iterdir()):
            if run_dir.is_dir():
                logs_dir = run_dir / "logs"
                if logs_dir.exists():
                    run_dirs.append(
                        {
                            "name": run_dir.name,
                            "path": str(logs_dir),
                            "run_dir": str(run_dir),
                        }
                    )

    return run_dirs


def launch_tensorboard(
    logdir: str,
    port: int = 6006,
    host: str = "localhost",
    samples_per_plugin: int = 2000,
    reload_interval: int = 120,
) -> int:
    """Launch TensorBoard for a single log directory or a name:path spec string.

    If *logdir* contains commas (``"run1:path1,run2:path2"``) it is treated as
    a ``--logdir_spec`` argument so TensorBoard labels each run separately.

    Args:
        samples_per_plugin: Max scalar data points loaded per metric (default 2000).
            Large training runs produce multi-hundred-MB event files; limiting
            samples keeps load time under a second instead of 60+ seconds.
        reload_interval: Seconds between TensorBoard data reloads (default 120).
            The default Ray/TF value is 5 s, which causes perpetual reloading of
            large event files and makes the UI appear empty.

    Returns the TensorBoard process exit code.
    """
    extra = [
        "--samples_per_plugin",
        f"scalars={samples_per_plugin}",
        "--reload_interval",
        str(reload_interval),
    ]
    if "," in logdir and ":" in logdir:
        cmd = ["tensorboard", "--logdir_spec", logdir, "--port", str(port), "--host", host] + extra
    else:
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", host] + extra

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error launching TensorBoard: {e}")
        return 1
    except FileNotFoundError:
        print("Error: TensorBoard not found. Install with: pip install tensorboard")
        return 1
    return 0


def launch_tensorboard_multi(
    logdirs: list[str],
    port: int = 6006,
    host: str = "localhost",
) -> int:
    """Launch TensorBoard comparing multiple log directories via symlinks."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, logdir in enumerate(logdirs):
            logdir_path = Path(logdir)
            run_name = (
                f"run_{i + 1}_{logdir_path.parent.name}"
                if logdir_path.parent.name != "."
                else f"run_{i + 1}"
            )
            target_path = temp_path / run_name

            if os.name == "nt":
                try:
                    subprocess.run(
                        ["mklink", "/J", str(target_path), str(logdir_path)],
                        shell=True,
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    shutil.copytree(logdir_path, target_path)
            else:
                target_path.symlink_to(logdir_path.resolve())

        cmd = [
            "tensorboard",
            "--logdir",
            str(temp_path),
            "--port",
            str(port),
            "--host",
            host,
            "--samples_per_plugin",
            "scalars=2000",
            "--reload_interval",
            "120",
        ]
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nTensorBoard stopped.")
        except subprocess.CalledProcessError as e:
            print(f"Error launching TensorBoard: {e}")
            return 1
        except FileNotFoundError:
            print("Error: TensorBoard not found. Install with: pip install tensorboard")
            return 1

    return 0
