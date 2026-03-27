"""
Training service — shared logic for CLI and GUI.

Provides command construction, config discovery, and background process
launching so both the GUI (training_dashboard) and CLI (batch_train) use
identical mechanics.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from air_to_air_rl.core.paths import project_root, rl_configs_root, runtime_root


def list_training_configs() -> list[str]:
    """Return sorted config filenames available for training.

    Searches (in order):
    1. ``configs/training/`` at the project working directory (user configs)
    2. The package ``rl/configs/`` directory (shipped defaults)

    Returns bare filenames (e.g. ``"aggressive.yaml"``), deduplicated,
    alphabetically sorted.
    """
    seen: set[str] = set()
    results: list[str] = []

    dirs = [
        project_root() / "configs" / "training",
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
    """Build the command list for ``air_to_air_rl.training.train``."""
    cmd = [sys.executable, "-m", "air_to_air_rl.training.train"]
    if config:
        cmd += ["--config", config]
    if overrides:
        cmd += ["--overrides"] + overrides
    return cmd


def build_simple_training_cmd(
    config: "str | None" = None,
    overrides: "list[str] | None" = None,
) -> list[str]:
    """Build the command list for ``air_to_air_rl.training.train_simple``."""
    cmd = [sys.executable, "-m", "air_to_air_rl.training.train_simple"]
    if config:
        cmd += ["--config", config]
    if overrides:
        cmd += ["--overrides"] + overrides
    return cmd


def build_batch_cmd(batch_args: list[str]) -> list[str]:
    """Build the command list for ``scripts/batch/batch_train`` as a module."""
    return [sys.executable, "-m", "scripts.batch.batch_train"] + batch_args


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
        log_dir = runtime_root() / "gui" / "logs"
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
            cwd=str(project_root()),
        )

    return process, log_file
