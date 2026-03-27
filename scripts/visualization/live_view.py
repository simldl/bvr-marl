#!/usr/bin/env python3
"""
BVR-MARL visualization launcher — unified interface for all visualization modes.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Launch visualization with mode selection."""
    parser = argparse.ArgumentParser(
        description="Launch BVR-MARL visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization Modes:
  standard         Standard 2D live view
  behavior-tree    Behavior tree visualization
  commands         RL commands panel visualization

Examples:
  # Standard visualization
  python scripts/visualization/live_view.py --mode standard


  # Behavior tree visualization
  python scripts/visualization/live_view.py --mode behavior-tree

  # Commands visualization
  python scripts/visualization/live_view.py --mode commands
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["standard", "behavior-tree", "commands"],
        default="standard",
        help="Visualization mode (default: standard)",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--model-config", type=str, help="Path to model config YAML file")
    parser.add_argument("--train-config", type=str, help="Path to train config YAML file")
    parser.add_argument("--viz-config", type=str, help="Path to visualization config YAML file")
    parser.add_argument(
        "--save-episode-logs", action="store_true", help="Save detailed episode logs"
    )
    parser.add_argument(
        "--no-save-episode-logs", action="store_true", help="Disable episode logging"
    )
    parser.add_argument("--frames", type=int, help="Number of frames to simulate")
    parser.add_argument("--interval", type=int, help="Animation interval in milliseconds")
    parser.add_argument("--save-video", action="store_true", help="Save animation as video file")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed (overrides interval)",
    )

    args = parser.parse_args()

    # Map modes to commands
    mode_commands = {
        "standard": "air2air-view",
        "behavior-tree": "air2air-view-behavior-tree",
        "commands": "air2air-view-commands",
    }

    cmd = [mode_commands[args.mode]]

    # Add common arguments
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.model_config:
        cmd.extend(["--model-config", args.model_config])
    if args.train_config:
        cmd.extend(["--train-config", args.train_config])
    if args.viz_config:
        cmd.extend(["--viz-config", args.viz_config])
    # --save-rewards/--no-save-rewards are only supported by the standard mode
    if args.frames:
        cmd.extend(["--frames", str(args.frames)])
    if args.interval:
        cmd.extend(["--interval", str(args.interval)])
    if args.save_video:
        cmd.append("--save-video")
    if args.real_time:
        cmd.append("--real-time")

    print("=" * 60)
    print(f"BVR-MARL Visualization Launcher ({args.mode})")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Visualization failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"Error: {mode_commands[args.mode]} command not found. Please install the package:")
        print("  pip install -e .")
        return 1


if __name__ == "__main__":
    exit(main())
