"""
Profile a complete training run including RL overhead.

Delegates to air_to_air_rl.training.train.train_main so the profiled code path
is identical to a real run.  Since Ray uses separate worker processes, cProfile
only captures the driver process (orchestration, PPO loss, checkpoint I/O).
Useful metrics:

  - Training setup time (Ray init, env registration, PPO build)
  - Per-iteration wall-clock time and throughput
  - PPO loss computation (num_learners=0 → learner in driver process)

Usage:
    python tools/profiling/profile_training.py --iterations 20
    python tools/profiling/profile_training.py --iterations 5 --config stable
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def profile_training(num_iterations: int = 20, config_name: str = "aggressive") -> None:
    from air_to_air_rl.core.paths import project_root, rl_configs_root
    from air_to_air_rl.training.train import train_main
    from air_to_air_rl.utils.simple_config import apply_overrides, load_config

    print("=" * 80)
    print("TRAINING PROFILING")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Iterations:    {num_iterations}")
    print("=" * 80)
    print()

    # Resolve config path: prefer top-level configs/training/, fall back to package defaults
    top_level = project_root() / "configs" / "training" / f"{config_name}.yaml"
    pkg_default = rl_configs_root() / f"{config_name}.yaml"
    config_file = top_level if top_level.exists() else pkg_default
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return

    config = load_config(config_file)

    # Minimal resource overrides for profiling — 1 env runner, CPU only
    apply_overrides(
        config,
        [
            f"training.steps={num_iterations}",
            "num_env_runners=1",
            "num_gpus=0",
        ],
    )

    cfg = config.to_dict()

    output_dir = project_root() / "profiling_results"
    output_dir.mkdir(exist_ok=True)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()

    profiler.enable()
    train_main(cfg)
    profiler.disable()

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"  Total wall-clock time : {elapsed:.1f} s")
    print(f"  Iterations            : {num_iterations}")
    print(f"  Avg per iteration     : {elapsed / num_iterations:.1f} s")
    print()

    # Print top functions (driver process only — workers are separate processes)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    print("Top 40 functions by CUMULATIVE time (driver process):")
    print("-" * 80)
    ps.print_stats(40)
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("tottime")
    print("Top 40 functions by TOTAL (self) time (driver process):")
    print("-" * 80)
    ps.print_stats(40)
    print(s.getvalue())

    # Persist results
    stats_file = output_dir / "training_profile.stats"
    profiler.dump_stats(str(stats_file))

    report_file = output_dir / "training_profile_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING PROFILING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"Total time: {elapsed:.1f} s  ({elapsed / num_iterations:.1f} s/iter)\n")
        f.write("=" * 80 + "\n\n")

        for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "TOTAL (self)")]:
            s2 = io.StringIO()
            psx = pstats.Stats(profiler, stream=s2)
            psx.strip_dirs()
            psx.sort_stats(sort_key)
            f.write(f"Top 100 functions by {label} time:\n")
            f.write("-" * 80 + "\n")
            psx.print_stats(100)
            f.write(s2.getvalue())
            f.write("\n\n")

    print(f"Detailed stats saved to : {stats_file}")
    print(f"Full report saved to    : {report_file}")
    print("=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile complete BVR-MARL training run")
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of training iterations (default: 20)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="aggressive",
        help="Training config name (default: aggressive)",
    )
    args = parser.parse_args()
    profile_training(args.iterations, args.config)
