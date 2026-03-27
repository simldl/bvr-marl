"""
Live 2D visualization with behavior tree controllers.

This script runs visualization using behavior tree controllers instead of trained RL models.

"""

import argparse
import os

import numpy as np

# Fix OpenMP library conflict - set before importing any libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from air_to_air_rl.aircrafts.types.debug_plane import DebugPlane
from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.aircrafts.types.f35 import F35
from air_to_air_rl.automation.scripted_control.full_tactical_controller import (
    FullTacticalController,
)
from air_to_air_rl.automation.strategies.aggressive import AggressiveStrategy
from air_to_air_rl.automation.strategies.balanced import BalancedStrategy
from air_to_air_rl.automation.strategies.defensive import DefensiveStrategy
from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.utils.config_loader import load_train_config, load_viz_config
from air_to_air_rl.visualization.scenplotter.video_generation import live_simulation


def get_aircraft_class(aircraft_type):
    """Map aircraft type string to class."""
    aircraft_map = {
        "F22": F22,
        "Eurofighter": Eurofighter,
        "F35": F35,
        "DebugPlane": DebugPlane,
    }
    return aircraft_map.get(aircraft_type, DebugPlane)


class BehaviorTreeModel:
    """Model that uses behavior tree controllers for all agents, replacing DefaultModel."""

    def __init__(self, env):
        self.env = env
        self.controllers = {}
        self.strategies = ["defensive", "balanced", "aggressive"]

    def _get_strategy_config(self, strategy_name):
        """Get configuration for specified strategy."""
        if strategy_name == "defensive":
            return DefensiveStrategy.create_config()
        elif strategy_name == "balanced":
            return BalancedStrategy.create_config()
        elif strategy_name == "aggressive":
            return AggressiveStrategy.create_config()
        else:
            return BalancedStrategy.create_config()

    def setup_controllers(self, aircraft_specs):
        """Setup behavior tree controllers for each agent."""
        agent_ids = list(self.env.observation_space.keys())

        for i, agent_id in enumerate(agent_ids):
            # Determine aircraft type and strategy
            aircraft_type = aircraft_specs.get(agent_id, "DebugPlane")
            strategy_name = self.strategies[i % len(self.strategies)]

            # Create aircraft instance with default parameters
            from air_to_air_rl.simulator.core.helpers import Position

            aircraft_class = get_aircraft_class(aircraft_type)
            # Create default parameters for aircraft initialization
            default_position = Position(lat=0.0, lon=0.0, alt=5000.0)
            default_yaw = 0.0
            default_speed = 250.0
            default_group = "agent" if i % 2 == 0 else "opponent"
            default_map_limits = type("MapLimits", (), {"min_alt": 0.0, "max_alt": 20000.0})()
            default_min_alt = 0.0
            default_max_alt = 20000.0

            aircraft = aircraft_class(
                position=default_position,
                yaw_deg=default_yaw,
                speed_mps=default_speed,
                group=default_group,
                map_limits=default_map_limits,
                min_alt_m=default_min_alt,
                max_alt_m=default_max_alt,
            )

            # Create AutoHelperConfig for the controller
            from air_to_air_rl.automation.core.auto_helper import AutoHelperConfig

            auto_config = AutoHelperConfig()

            # Apply strategy-specific settings if needed
            if strategy_name == "aggressive":
                auto_config.automation_level = "aggressive"
            elif strategy_name == "defensive":
                auto_config.automation_level = "defensive"
            else:
                auto_config.automation_level = "balanced"

            # Create controller
            controller = FullTacticalController(aircraft=aircraft, config=auto_config)

            self.controllers[agent_id] = controller
            print(f"Agent {agent_id}: {aircraft_type} with {strategy_name} strategy")

    def compute_single_action(self, observation, agent_id):
        """Compute action using behavior tree controller."""
        if agent_id not in self.controllers:
            # Fallback to random action
            return np.random.rand(10).astype(np.float32)

        controller = self.controllers[agent_id]

        # Convert observation to controller input format
        # This may need adjustment based on observation structure
        try:
            action = controller.compute_action(observation)
            return action
        except Exception as e:
            print(f"Controller error for {agent_id}: {e}")
            # Fallback to random action
            return np.random.rand(10).astype(np.float32)


def run_behavior_tree_visualization(
    train_config_path=None,
    viz_config_path=None,
    frames=None,
    interval=None,
    save_video=None,
    real_time=False,
):
    """Run visualization with behavior tree controllers."""
    # Load viz config (symbol mode and render settings live here, same as other modes)
    viz_config = load_viz_config(viz_config_path)
    viz_settings = viz_config.get("visualization", {})
    frames = frames if frames is not None else viz_settings.get("frames", 100)
    interval = interval if interval is not None else viz_settings.get("interval", 100)
    save_video = save_video if save_video is not None else viz_settings.get("save_video", False)
    symbol_mode = viz_settings.get("symbol_mode", "nato")
    dpi = viz_settings.get("dpi", 200)
    symbol_scale = viz_settings.get("symbol_scale", 2.0)

    # Load train config
    train_config = load_train_config(train_config_path)
    env_config = train_config.get("env", {})

    # Initialize simulator and environment
    sim = Simulator()
    env_config["simulator"] = sim
    env = BVRMultiAgentEnv(env_config)

    # Initialize behavior tree model
    model = BehaviorTreeModel(env)

    # Setup aircraft specifications (can be customized)
    aircraft_specs = {
        "blue_0": "F22",
        "blue_1": "Eurofighter",
        "red_0": "F35",
        "red_1": "DebugPlane",
    }

    # Setup controllers
    model.setup_controllers(aircraft_specs)

    if real_time:
        tick_secs = env_config.get("simulation_config", {}).get("tick_secs", 1)
        interval = int(tick_secs * 1000)
        print(f"Real-time mode: interval set to {interval} ms")

    print(f"Running behavior tree visualization for {frames} frames...")

    # Run simulation
    live_simulation(
        trained_model=model,
        env=env,
        frames=frames,
        interval=interval,
        save_video=save_video,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
    )


def main():
    """Main entry point for air2air-view-behavior-tree command."""
    parser = argparse.ArgumentParser(
        description="Live Visualization with Behavior Tree Controllers"
    )
    parser.add_argument(
        "--train-config", type=str, default=None, help="Path to train config YAML file"
    )
    parser.add_argument(
        "--viz-config", type=str, default=None, help="Path to visualization config YAML file"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Number of frames (overrides viz config)"
    )
    parser.add_argument(
        "--interval", type=int, default=None, help="Animation interval in ms (overrides viz config)"
    )
    parser.add_argument("--save-video", action="store_true", help="Save animation as video file")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed (overrides interval)",
    )

    args = parser.parse_args()

    run_behavior_tree_visualization(
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        frames=args.frames,
        interval=args.interval,
        save_video=args.save_video if args.save_video else None,
        real_time=args.real_time,
    )


if __name__ == "__main__":
    main()
