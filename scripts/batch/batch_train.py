#!/usr/bin/env python3
"""
Batch training launcher — thin wrapper around services.batch.

All job-construction and execution logic lives in services.batch so both
this script and any future GUI integration share identical mechanics.
"""

import argparse

from bvr_marl_core.services.batch import build_batch_jobs, run_batch_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch BVR-MARL-Core training operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with multiple seeds
  python scripts/batch/batch_train.py --config basic.yaml --seeds 42 123 456

  # Train with config variations
  python scripts/batch/batch_train.py --configs config1.yaml config2.yaml config3.yaml

  # Hyperparameter sweep
  python scripts/batch/batch_train.py --config basic.yaml --sweep learning_rate 0.001,0.0005,0.0001
        """,
    )

    parser.add_argument("--config", type=str, help="Base config file")
    parser.add_argument("--configs", nargs="*", help="Multiple config files to train")
    parser.add_argument("--seeds", nargs="*", type=int, help="Multiple seeds to use")
    parser.add_argument(
        "--sweep",
        nargs=2,
        metavar=("param", "values"),
        help="Hyperparameter sweep: param_name comma_separated_values",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="batch_training",
        help="Prefix for output directory names",
    )

    args = parser.parse_args()

    sweep_values = args.sweep[1].split(",") if args.sweep else None
    jobs = build_batch_jobs(
        configs=args.configs,
        base_config=args.config,
        seeds=args.seeds,
        sweep_param=args.sweep[0] if args.sweep else None,
        sweep_values=sweep_values,
        output_prefix=args.output_prefix,
    )

    if not jobs:
        print("Error: No batch operation specified. Use --configs, --seeds, or --sweep")
        return 1

    print("=" * 60)
    print("Batch Training Operations")
    print("=" * 60)
    print(f"Jobs to run: {len(jobs)}")
    for i, job in enumerate(jobs, 1):
        print(f"  Job {i}: {job['name']}")
    print("=" * 60)

    successful, failed = run_batch_jobs(jobs)

    print("\n" + "=" * 60)
    print("Batch Training Summary")
    print("=" * 60)
    print(f"Successful jobs: {successful}")
    print(f"Failed jobs:     {failed}")
    print(f"Total jobs:      {len(jobs)}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
