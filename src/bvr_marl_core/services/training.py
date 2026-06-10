"""
Training service - shared logic for CLI and GUI.

Provides command construction, config discovery, and background process
launching so both the GUI (training_dashboard) and CLI (batch_train) use
identical mechanics.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bvr_marl_core.utils.paths import core_project_root, core_runtime_root, rl_configs_root


def list_training_configs() -> list[str]:
    """Return sorted config entries available for training.

    Searches (in order):
    1. ``configs/training/`` at the project working directory (user configs)
    2. The legacy package ``rl/configs/`` directory, if present

    Returns deduplicated bare filenames for backwards compatibility.
    """
    seen: set[str] = set()
    results: list[str] = []
    project_root = core_project_root()

    dirs = [
        project_root / "configs" / "training",
        rl_configs_root(),
    ]
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.yaml")):
                if f.name not in seen:
                    seen.add(f.name)
                    results.append(f.name)

    return results


def build_training_cmd(
    config: "str | None" = None,
    overrides: "list[str] | None" = None,
) -> list[str]:
    """Build the command list for core BVR training."""
    cmd = [sys.executable, "-m", "bvr_marl_core.training.train"]
    if config:
        cmd += ["--config", config]
    if overrides:
        cmd += ["--overrides"] + overrides
    return cmd


def build_simple_training_cmd(
    config: "str | None" = None,
    overrides: "list[str] | None" = None,
) -> list[str]:
    """Build the command list for core simplified training."""
    cmd = [sys.executable, "-m", "bvr_marl_core.training.train_simple"]
    if config:
        cmd += ["--config", config]
    if overrides:
        cmd += ["--overrides"] + overrides
    return cmd


def build_batch_cmd(batch_args: list[str]) -> list[str]:
    """Build the command list for a single batch training run.

    Delegates to the core BVR training entry point.
    """
    return [sys.executable, "-m", "bvr_marl_core.training.train"] + batch_args


def launch_background_process(
    cmd: list[str],
    label: str,
    log_dir: "Path | None" = None,
) -> "tuple[subprocess.Popen, Path]":
    """Launch *cmd* as a background process, capturing stdout+stderr to a log file.

    Args:
        cmd: Command list to execute (e.g. from a ``build_*`` helper).
        label: Short label used in the log filename (e.g. model name).
        log_dir: Directory for the log file.  Defaults to
                 ``runtime_root() / "gui" / "logs"``.

    Returns:
        ``(process, log_file_path)`` where *process* is the running
        :class:`subprocess.Popen` instance.
    """
    if log_dir is None:
        log_dir = core_runtime_root() / "gui" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_").replace("/", "_")
    log_file = log_dir / f"{safe_label}_{timestamp}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(core_project_root()),
        )

    return process, log_file
