#!/usr/bin/env python3
"""
Model evaluation launcher — thin wrapper around services.visualization.

For visual evaluation, delegates to the visualization service.
"""

import argparse
import subprocess
from pathlib import Path

from air_to_air_rl.services.visualization import build_visualization_cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained air-to-air RL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model with visualization
  python scripts/analysis/evaluate_model.py --checkpoint model.pkl --visualize

  # Evaluate model against specific opponent
  python scripts/analysis/evaluate_model.py --checkpoint blue.pkl --opponent-checkpoint red.pkl

  # Batch evaluation with different seeds
  python scripts/analysis/evaluate_model.py --checkpoint model.pkl --episodes 50 --seeds 5
        """,
    )

    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=str,
        help="Path to opponent model checkpoint (default: random actions)",
    )
    parser.add_argument("--config", type=str, help="Path to evaluation / training config file")
    parser.add_argument(
        "--episodes", "-e", type=int, default=10, help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--seeds", "-s", type=int, default=1, help="Number of different seeds to test (default: 1)"
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Run with visualization")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)",
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file does not exist: {args.checkpoint}")
        return 1

    if args.opponent_checkpoint and not Path(args.opponent_checkpoint).exists():
        print(f"Error: Opponent checkpoint file does not exist: {args.opponent_checkpoint}")
        return 1

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model checkpoint:  {args.checkpoint}")
    print(f"Opponent:          {args.opponent_checkpoint or 'Random actions'}")
    print(f"Episodes:          {args.episodes}")
    print(f"Seeds:             {args.seeds}")
    print(f"Visualization:     {'Yes' if args.visualize else 'No'}")
    print(f"Output directory:  {args.output_dir}")
    print("=" * 60)

    if args.visualize:
        cmd = build_visualization_cmd(
            mode="standard",
            checkpoint=args.checkpoint,
            train_config=args.config,
        )
        print(f"Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed with exit code {e.returncode}")
            return e.returncode
        except FileNotFoundError:
            print("Error: visualization command not found. Please install the package:")
            print("  pip install -e .")
            return 1
    else:
        print("Note: Non-visual evaluation not yet implemented.")
        print("Use --visualize flag for interactive evaluation.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
