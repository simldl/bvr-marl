"""
Tacview scenario generation with Enhanced Tactical Controller and Behavior Trees.

This script generates Tacview scenarios using behavior tree controllers instead of RL models.

"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.automation.scripted_control.enhanced_tactical_controller import (
    EnhancedTacticalController,
)
from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.simulator.utils.map_limits import MapLimits
from air_to_air_rl.tacview.logger import TacviewLogger


class BehaviorTreeModel:
    """Model that uses Enhanced Tactical Controller with Behavior Tree for all agents."""

    def __init__(self, env):
        self.env = env
        self.controllers = {}  # Store controllers per agent

    def compute_single_action(self, observation, agent_id):
        """Compute action using Enhanced Tactical Controller with Behavior Tree."""
        try:
            # Get the aircraft for this agent
            aircraft = None
            for uid, unit in self.env.simulator.active_units.items():
                if hasattr(unit, "id") and unit.id == agent_id:
                    aircraft = unit
                    break

            if aircraft is None:
                return np.random.rand(10).astype(np.float32)

            # Initialize controller for this agent if not exists
            if agent_id not in self.controllers:
                print(
                    f"Initializing Enhanced Tactical Controller for Agent {agent_id} ({aircraft.name})"
                )

                # Create controller with enhanced configuration
                try:
                    from air_to_air_rl.automation.core.auto_helper import AutoHelperConfig

                    config = AutoHelperConfig(
                        max_missile_range_km=100,  # 100km max range for BVR
                        weapon_lock_threshold_km=100,  # Allow locking at maximum range
                        defensive_threshold_km=25,  # Defensive when threats < 25km
                        notch_angle_tolerance_deg=15,  # Improved notching accuracy
                    )
                except ImportError:
                    # Fallback if AutoHelperConfig is not available
                    print("Warning: AutoHelperConfig not available, using default configuration")
                    config = None

                # Create the Enhanced Tactical Controller
                controller = EnhancedTacticalController(aircraft, config)
                self.controllers[agent_id] = controller

            # Get the controller for this agent
            controller = self.controllers[agent_id]

            # Compute action using the behavior tree
            action = controller.compute_action(observation, aircraft, self.env.simulator)

            # Ensure action is in the correct format (numpy array)
            if not isinstance(action, np.ndarray):
                action = np.array(action)

            # Ensure correct data type
            action = action.astype(np.float32)

            # Ensure correct shape (should be 10-dimensional for the action space)
            if len(action) != 10:
                print(
                    f"Warning: Action has wrong dimension {len(action)}, padding/truncating to 10"
                )
                if len(action) < 10:
                    # Pad with zeros
                    padded_action = np.zeros(10, dtype=np.float32)
                    padded_action[: len(action)] = action
                    action = padded_action
                else:
                    # Truncate to 10 dimensions
                    action = action[:10]

            return action

        except Exception as e:
            print(f"Error in BehaviorTreeModel.compute_single_action for {agent_id}: {e}")
            # Return random action as fallback
            return np.random.rand(10).astype(np.float32)


def generate_unique_filename(base_path="tacview/logs/scenario_bt.acmi"):
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_path)
    # Extract directory and filename
    directory = path.parent
    stem = path.stem if path.stem != "scenario_bt" else "scenario_bt"
    extension = path.suffix

    # Create unique filename
    unique_path = directory / f"{stem}_{timestamp}{extension}"

    # Ensure directory exists
    unique_path.parent.mkdir(parents=True, exist_ok=True)

    return str(unique_path)


def run_behavior_tree_tacview(
    frames=200, acmi_path=None, progress_interval=50, seed=None, aircraft_type=F22
):
    """
    Run Tacview scenario generation with behavior tree controllers.

    Args:
        frames: Number of simulation steps to run
        acmi_path: Path to save ACMI file (if None, auto-generates unique name)
        progress_interval: Print progress every N steps
        seed: Random seed for environment reset
        aircraft_type: Aircraft type to use for all agents
    """
    # Generate unique filename if acmi_path is None
    if acmi_path is None:
        acmi_path = generate_unique_filename("tacview/logs/scenario_bt.acmi")
        print(f"Auto-generated output path: {acmi_path}")

    # Initialize simulator
    sim = Simulator(tick_secs=0.1)
    _half = (300.0 / 2.0) / 111.0
    map_limits = MapLimits(
        left_lon=-_half,
        bottom_lat=-_half,
        right_lon=_half,
        top_lat=_half,
        min_alt=0.0,
        max_alt=15000.0,
    )

    # Configure environment for behavior tree testing
    env_config = {
        "simulator": sim,
        "map_limits": map_limits,
        "num_agents_per_team": 2,
        "map_size": 300,
        "max_steps": frames,
        "debug": True,
        "simulation_config": {},
        "weapon_config": {},
        "aircraft_types": {"agent": aircraft_type, "opponent": aircraft_type},
    }

    # Initialize environment
    env = BVRMultiAgentEnv(env_config)

    # Initialize Tacview logger
    tacview_logger = TacviewLogger(acmi_path)

    # Register the Tacview logger with the simulator
    sim.tacview_logger = tacview_logger

    # Initialize behavior tree model
    model = BehaviorTreeModel(env)

    print("=" * 60)
    print("Behavior Tree Tacview Scenario Generation")
    print("=" * 60)
    print(f"Aircraft Type: {aircraft_type.__name__}")
    print(f"Frames: {frames}")
    print(f"Output: {acmi_path}")
    print(f"Seed: {seed if seed is not None else 'Random'}")
    print("=" * 60)

    # Run simulation
    obs, _ = env.reset(seed=seed)
    if seed is not None:
        print(f"Environment reset with seed: {seed}")

    print("\nActive units after reset:")
    for uid, unit in sim.active_units.items():
        print(f"  - Unit ID {uid}: {getattr(unit, 'name', None)}")

    print(f"\nRunning behavior tree simulation for {frames} steps...")
    dones = {"__all__": False}
    steps = 0

    while steps < frames and not dones.get("__all__", False):
        actions = {}
        for aid in env.all_agent_ids:
            # Only get actions for agents that are still active (have observations)
            if aid in obs:
                obs_agent = obs[aid]
                actions[aid] = model.compute_single_action(obs_agent, aid)

        obs, rewards, dones, trunc, infos = env.step(actions)
        steps += 1

        if steps % progress_interval == 0:
            print(f"  Step {steps}/{frames}")

    print(f"\nSimulation completed after {steps} steps")

    if acmi_path:
        print(f"Tacview file saved to: {acmi_path}")

    print("=" * 60)
    print("Behavior Tree scenario generation complete!")
    print("=" * 60)


def main():
    """Main entry point for air2air-tacview-bt command."""
    parser = argparse.ArgumentParser(
        description="Tacview scenario generation with Enhanced Tactical Controller and Behavior Trees"
    )
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
        help="Aircraft type to use for all agents (default: F22)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N steps (default: 50)",
    )

    args = parser.parse_args()

    # Map aircraft type string to class
    {
        "F22": F22,
        "Eurofighter": lambda: (
            __import__(
                "air_to_air_rl.aircrafts.types.eurofighter", fromlist=["Eurofighter"]
            ).Eurofighter
        ),
        "F35": lambda: __import__("air_to_air_rl.aircrafts.types.f35", fromlist=["F35"]).F35,
    }

    try:
        if args.aircraft == "Eurofighter":
            from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter

            aircraft_type = Eurofighter
        elif args.aircraft == "F35":
            from air_to_air_rl.aircrafts.types.f35 import F35

            aircraft_type = F35
        else:
            aircraft_type = F22
    except ImportError as e:
        print(f"Warning: Could not import {args.aircraft}, using F22 as fallback: {e}")
        aircraft_type = F22

    # Determine number of scenarios and seeds
    num_scenarios = args.num_scenarios
    seed_start = args.seed_start

    # If --seed is provided, use it (overrides num_scenarios to 1)
    if args.seed is not None:
        seeds = [args.seed]
        num_scenarios = 1
    elif num_scenarios > 1:
        # Generate multiple scenarios with sequential seeds
        seeds = list(range(seed_start, seed_start + num_scenarios))
    else:
        # Single scenario with no specific seed
        seeds = [None]

    print("=" * 60)
    print("Behavior Tree Tacview Scenario Generation")
    print("=" * 60)
    print(f"Scenarios to generate: {num_scenarios}")
    if num_scenarios > 1:
        print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    elif seeds[0] is not None:
        print(f"Seed: {seeds[0]}")
    print(f"Frames per scenario: {args.frames}")
    print(f"Aircraft type: {aircraft_type.__name__}")
    print(f"Output: {args.acmi if args.acmi else 'Auto-generated (timestamp-based)'}")
    print("=" * 60 + "\n")

    # Generate scenarios
    for i, seed in enumerate(seeds, 1):
        if num_scenarios > 1:
            print(f"\n{'=' * 60}")
            print(f"Generating scenario {i}/{num_scenarios} (seed={seed})")
            print(f"{'=' * 60}\n")

            # Generate unique output path for each scenario
            if args.acmi is None:
                # Auto-generate with seed in filename
                scenario_acmi = generate_unique_filename(
                    f"tacview/logs/scenario_bt_seed{seed}.acmi"
                )
            else:
                # User-specified path - add seed suffix
                path = Path(args.acmi)
                scenario_acmi = str(path.parent / f"{path.stem}_seed{seed}{path.suffix}")
        else:
            scenario_acmi = args.acmi

        # Run behavior tree scenario
        run_behavior_tree_tacview(
            frames=args.frames,
            acmi_path=scenario_acmi,
            progress_interval=args.progress_interval,
            seed=seed,
            aircraft_type=aircraft_type,
        )

    if num_scenarios > 1:
        print(f"\n{'=' * 60}")
        print(f"Successfully generated {num_scenarios} behavior tree scenarios!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
