#!/usr/bin/env python3
"""
Training run comparison launcher — thin wrapper around services.batch.

All run-discovery and comparison logic lives in services.batch so both
this script and any future GUI integration share identical mechanics.
"""

import argparse

from bvr_marl_core.services.batch import (
    collect_run_dirs,
    compare_runs_summary,
    launch_tensorboard_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare BVR-MARL-Core training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific runs
  python scripts/batch/compare_runs.py --runs outputs/run1 outputs/run2

  # Compare all runs in directory
  python scripts/batch/compare_runs.py --runs-dir outputs/

  # Launch TensorBoard for comparison
  python scripts/batch/compare_runs.py --runs outputs/run1 outputs/run2 --tensorboard
        """,
    )

    parser.add_argument("--runs", nargs="*", help="Specific run directories to compare")
    parser.add_argument("--runs-dir", type=str, help="Directory containing multiple runs")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch TensorBoard for interactive comparison",
    )
    parser.add_argument(
        "--export-plots", action="store_true", help="Export comparison plots (future feature)"
    )
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port (default: 6006)")

    args = parser.parse_args()

    run_dirs = collect_run_dirs(runs=args.runs, runs_dir=args.runs_dir)

    if not run_dirs:
        print("Error: No valid run directories found")
        return 1

    print("=" * 60)
    print("Training Run Comparison")
    print("=" * 60)
    print(f"Runs to compare: {len(run_dirs)}")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  {i}. {run_dir}")
    print("=" * 60)

    if args.tensorboard:
        print("Launching TensorBoard for comparison...")
        launch_tensorboard_comparison(run_dirs, port=args.port)

    if args.export_plots:
        print("Export plots not yet implemented — use export_plots.py per run.")

    if not args.tensorboard and not args.export_plots:
        print("Run comparison summary:")
        print("(Use --tensorboard for interactive analysis)")
        compare_runs_summary(run_dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
