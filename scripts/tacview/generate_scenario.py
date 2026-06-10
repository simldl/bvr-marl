#!/usr/bin/env python3
"""
BVR-MARL-Core Tacview scenario generation launcher — unified interface for RL and behavior-tree controllers.
"""

import argparse
import subprocess


def main():
    """Launch Tacview scenario generation."""
    parser = argparse.ArgumentParser(
        description="Generate BVR-MARL-Core Tacview scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controller Types:
  rl            Use trained RL models (default)
  behavior-tree Use behavior tree controllers

Examples:
  # Generate scenario with RL model
  python scripts/tacview/generate_scenario.py --checkpoint model.pkl

  # Generate scenario with behavior trees
  python scripts/tacview/generate_scenario.py --controller behavior-tree

  # Generate multiple scenarios
  python scripts/tacview/generate_scenario.py --num-scenarios 5 --seed-start 100

  # Generate long scenario
  python scripts/tacview/generate_scenario.py --frames 1000 --checkpoint model.pkl
        """,
    )

    parser.add_argument(
        "--controller",
        type=str,
        choices=["rl", "behavior-tree"],
        default="rl",
        help="Controller type to use (default: rl)",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to trained model checkpoint (for RL controller)"
    )
    parser.add_argument("--model-config", type=str, help="Path to model config YAML file")
    parser.add_argument("--train-config", type=str, help="Path to training config YAML file")
    parser.add_argument(
        "--frames", type=int, default=500, help="Number of simulation steps to run (default: 500)"
    )
    parser.add_argument(
        "--acmi", type=str, help="ACMI output file path (auto-generates if not provided)"
    )
    parser.add_argument("--seed", type=int, help="Random seed for environment reset")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=1,
        help="Number of scenarios to generate with different seeds (default: 1)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value when generating multiple scenarios (default: 0)",
    )
    parser.add_argument(
        "--aircraft",
        type=str,
        default="F22",
        choices=["F22", "Eurofighter", "F35"],
        help="Aircraft type to use (for behavior-tree controller, default: F22)",
    )

    args = parser.parse_args()

    # Choose command based on controller type
    if args.controller == "behavior-tree":
        cmd = ["bvr-tacview-bt"]

        # Add behavior-tree specific arguments
        if args.aircraft != "F22":
            cmd.extend(["--aircraft", args.aircraft])

    else:
        cmd = ["bvr-tacview"]

        # Add RL-specific arguments
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.model_config:
            cmd.extend(["--model-config", args.model_config])
        if args.train_config:
            cmd.extend(["--train-config", args.train_config])

    # Add common arguments
    if args.frames != 500:
        cmd.extend(["--frames", str(args.frames)])
    if args.acmi:
        cmd.extend(["--acmi", args.acmi])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.num_scenarios != 1:
        cmd.extend(["--num-scenarios", str(args.num_scenarios)])
    if args.seed_start != 0:
        cmd.extend(["--seed-start", str(args.seed_start)])

    print("=" * 60)
    print(f"Tacview Scenario Generation ({args.controller})")
    print("=" * 60)
    print(f"Controller: {args.controller}")
    if args.controller == "rl":
        print(f"Checkpoint: {args.checkpoint or 'Random actions'}")
    else:
        print(f"Aircraft: {args.aircraft}")
    print(f"Scenarios: {args.num_scenarios}")
    print(f"Frames per scenario: {args.frames}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("\nTacview scenario generation completed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tacview generation failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        cmd_name = cmd[0]
        print(f"Error: {cmd_name} command not found. Please install the package:")
        print("  pip install -e .")
        return 1


if __name__ == "__main__":
    exit(main())
