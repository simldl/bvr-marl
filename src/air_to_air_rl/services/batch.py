"""
Batch training and run-comparison service.

Used by ``scripts/batch/batch_train.py`` and ``scripts/batch/compare_runs.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def build_batch_jobs(
    configs: list[str] | None = None,
    base_config: str | None = None,
    seeds: list[int] | None = None,
    sweep_param: str | None = None,
    sweep_values: list[str] | None = None,
    output_prefix: str = "batch_training",
) -> list[dict]:
    """Return a list of training job descriptors from the given batch spec.

    Each descriptor is a dict with keys: ``name``, ``config``, and optionally
    ``seed`` and ``override``.
    """
    jobs: list[dict] = []

    if configs:
        for config in configs:
            if Path(config).exists():
                jobs.append(
                    {
                        "config": config,
                        "name": f"{output_prefix}_{Path(config).stem}",
                    }
                )
            else:
                print(f"Warning: Config file not found: {config}")

    elif seeds:
        for seed in seeds:
            jobs.append(
                {
                    "config": base_config,
                    "seed": seed,
                    "name": f"{output_prefix}_seed_{seed}",
                }
            )

    elif sweep_param and sweep_values:
        for value in sweep_values:
            jobs.append(
                {
                    "config": base_config,
                    "override": f"{sweep_param}={value.strip()}",
                    "name": f"{output_prefix}_{sweep_param}_{value.strip()}",
                }
            )

    return jobs


def run_batch_jobs(jobs: list[dict]) -> tuple[int, int]:
    """Execute a list of training job descriptors sequentially.

    Returns ``(successful_count, failed_count)``.
    """
    successful = 0
    failed = 0

    for i, job in enumerate(jobs, 1):
        print(f"\n{'=' * 20} Job {i}/{len(jobs)} {'=' * 20}")
        print(f"Name: {job['name']}")

        cmd = [sys.executable, "-m", "air_to_air_rl.training.train"]

        if job.get("config"):
            cmd.extend(["--config", job["config"]])

        overrides: list[str] = []
        if job.get("seed"):
            overrides.append(f"seed={job['seed']}")
        if job.get("override"):
            overrides.append(job["override"])
        overrides.append(f"logging.model_name={job['name']}")

        if overrides:
            cmd.extend(["--overrides"] + overrides)

        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)

        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"\nJob {i} completed successfully in {duration:.1f}s")
            successful += 1
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"\nJob {i} failed with exit code {e.returncode} after {duration:.1f}s")
            failed += 1
        except KeyboardInterrupt:
            print("\nBatch training interrupted by user")
            break

    return successful, failed


def collect_run_dirs(
    runs: list[str] | None = None,
    runs_dir: str | None = None,
) -> list[Path]:
    """Collect validated run directories from explicit paths and/or a parent dir."""
    run_dirs: list[Path] = []

    if runs:
        for run in runs:
            p = Path(run)
            if p.exists():
                run_dirs.append(p)
            else:
                print(f"Warning: Run directory not found: {run}")

    if runs_dir:
        base = Path(runs_dir)
        if base.exists():
            for subdir in sorted(base.iterdir()):
                if subdir.is_dir() and (subdir / "checkpoint").exists():
                    run_dirs.append(subdir)
        else:
            print(f"Warning: Runs directory not found: {runs_dir}")

    return run_dirs


def compare_runs_summary(run_dirs: list[Path]) -> None:
    """Print a basic textual summary of the given run directories."""
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\nRun {i}: {run_dir.name}")

        if (run_dir / "checkpoint").exists():
            print("  Has checkpoints")
        if (run_dir / "logs").exists():
            print("  Has training logs")

        for config_file in ["config.yaml", "train_config.yaml", ".hydra/config.yaml"]:
            if (run_dir / config_file).exists():
                print(f"  Config: {config_file}")
                break


def launch_tensorboard_comparison(run_dirs: list[Path], port: int = 6006) -> None:
    """Launch TensorBoard comparing all given run directories."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, run_dir in enumerate(run_dirs):
            link_name = f"run_{i + 1}_{run_dir.name}"
            link_path = temp_path / link_name

            log_dir: Path | None = None
            for possible_log in [run_dir / "logs", run_dir, run_dir / "tensorboard"]:
                if possible_log.exists():
                    log_dir = possible_log
                    break

            if log_dir:
                if os.name != "nt":
                    os.symlink(log_dir, link_path)
                else:
                    print(f"Run {i + 1}: {log_dir}")

        if os.name == "nt":
            print("Note: On Windows, manually start TensorBoard with:")
            for i, run_dir in enumerate(run_dirs):
                print(f"  tensorboard --logdir {run_dir}:run_{i + 1}")
            return

        cmd = ["tensorboard", "--logdir", str(temp_path), "--port", str(port)]
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)
