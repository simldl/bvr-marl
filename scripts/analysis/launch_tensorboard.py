#!/usr/bin/env python3
"""
TensorBoard launcher — thin wrapper around services.tensorboard.

All discovery and launch logic lives in the service layer so the GUI
and this script share identical mechanics.
"""

import argparse
from pathlib import Path

from bvr_marl_core.services import tensorboard as tb_service


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard with automatic log directory detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and launch latest run
  python scripts/analysis/launch_tensorboard.py

  # Launch specific log directory
  python scripts/analysis/launch_tensorboard.py --logdir models/my_model

  # Compare multiple runs
  python scripts/analysis/launch_tensorboard.py --compare models/run1 models/run2

  # Launch on different port
  python scripts/analysis/launch_tensorboard.py --port 6007

  # List available runs without launching
  python scripts/analysis/launch_tensorboard.py --list-only
        """,
    )

    parser.add_argument("--logdir", type=str, help="Specific log directory to use")
    parser.add_argument(
        "--logdir_spec",
        type=str,
        help="Comma-separated name:path pairs for multi-run comparison",
    )
    parser.add_argument("--compare", nargs="*", help="Compare multiple log directories")
    parser.add_argument(
        "--port", type=int, default=6006, help="Port for TensorBoard (default: 6006)"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host for TensorBoard")
    parser.add_argument("--list-only", action="store_true", help="List runs without launching")

    args = parser.parse_args()

    log_dirs = tb_service.find_log_directories()
    training_runs = tb_service.find_training_runs()

    if args.list_only:
        print("=" * 60)
        print("Available Training Runs")
        print("=" * 60)
        if training_runs:
            for i, run in enumerate(training_runs, 1):
                print(f"{i}. {run['name']}")
                print(f"   Path: {run['path']}")
        else:
            print("No training runs found with logs.")
        if log_dirs:
            print("Additional log directories:")
            for logdir in log_dirs:
                if not any(run["path"] == logdir for run in training_runs):
                    print(f"  {logdir}")
        return 0

    if args.logdir_spec:
        return tb_service.launch_tensorboard(args.logdir_spec, args.port, args.host)

    if args.compare:
        return tb_service.launch_tensorboard_multi(args.compare, args.port, args.host)

    if args.logdir:
        if not Path(args.logdir).exists():
            print(f"Error: Log directory does not exist: {args.logdir}")
            return 1
        return tb_service.launch_tensorboard(args.logdir, args.port, args.host)

    # Auto-detect mode
    if not training_runs and not log_dirs:
        print("No TensorBoard logs found in the project.")
        print("Train a model first, or specify --logdir manually.")
        return 1

    if len(training_runs) == 1:
        run = training_runs[0]
        print(f"Found single training run: {run['name']}")
        return tb_service.launch_tensorboard(run["path"], args.port, args.host)

    if len(training_runs) > 1:
        print("=" * 60)
        print("Multiple training runs found")
        print("=" * 60)
        for i, run in enumerate(training_runs, 1):
            print(f"{i}. {run['name']}")
        print(f"{len(training_runs) + 1}. Compare all runs")
        print(f"{len(training_runs) + 2}. Exit")
        try:
            choice_num = int(input("\nSelect run (number): ").strip())
            if choice_num <= len(training_runs):
                return tb_service.launch_tensorboard(
                    training_runs[choice_num - 1]["path"], args.port, args.host
                )
            if choice_num == len(training_runs) + 1:
                all_logdirs = [run["path"] for run in training_runs]
                return tb_service.launch_tensorboard_multi(all_logdirs, args.port, args.host)
            return 0
        except (ValueError, KeyboardInterrupt):
            print("Invalid selection or cancelled.")
            return 1

    return tb_service.launch_tensorboard(log_dirs[0], args.port, args.host)


if __name__ == "__main__":
    raise SystemExit(main())
