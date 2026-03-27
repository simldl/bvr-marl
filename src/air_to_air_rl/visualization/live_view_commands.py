"""
Live 2D visualization with RL command time-series side panel.

Shows the 2D battle map on the left and scrolling line plots of
ps / n / phi (specific excess power, load factor, roll command)
for every active agent on the right — updated each simulation tick.

"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.utils.config_loader import load_train_config, load_viz_config
from air_to_air_rl.visualization.utils.path_utils import resolve_relative_path
from air_to_air_rl.visualization.utils.rl_commands_visualizer import RLCommandsVisualizer

# Fix OpenMP library conflict - set before Ray/torch initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")


def run_rl_commands_visualization(
    checkpoint_path=None,
    model_config_path=None,
    train_config_path=None,
    viz_config_path=None,
    real_time=False,
):
    """Run the 2D sim + RL command plots visualization."""
    # Load config
    viz_config = load_viz_config(viz_config_path)

    # Use viz_config values if command-line arguments are not provided
    if checkpoint_path is None:
        checkpoint_path = viz_config.get("checkpoint_path")
    if model_config_path is None:
        model_config_path = viz_config.get("model_config_path")
    if train_config_path is None:
        train_config_path = viz_config.get("train_config_path")

    # Resolve relative paths to absolute paths
    checkpoint_path = resolve_relative_path(checkpoint_path)
    model_config_path = resolve_relative_path(model_config_path)
    train_config_path = resolve_relative_path(train_config_path)

    # Get visualization settings
    viz_settings = viz_config.get("visualization", {})
    frames = viz_settings.get("frames", 100)
    interval = viz_settings.get("interval", 100)
    save_video = viz_settings.get("save_video", False)
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

    # Always use random model (checkpoint-based inference not available in public release)
    print("Using random model (scripted/checkpoint inference not included in public release)")

    class _DefaultModel:
        def __init__(self, env):
            self.env = env

        def compute_actions(self, obs_dict):
            if hasattr(self.env.action_space, "sample"):
                # Single action space
                return {agent_id: self.env.action_space.sample() for agent_id in obs_dict}
            else:
                # Multi-agent action space (dict)
                return {agent_id: self.env.action_space[agent_id].sample() for agent_id in obs_dict}

        def compute_single_action(self, observation, agent_id):
            """Compute action for a single agent (for compatibility with RL commands visualizer)."""
            if hasattr(self.env.action_space, "sample"):
                # Single action space
                return self.env.action_space.sample()
            else:
                # Multi-agent action space (dict)
                return self.env.action_space[agent_id].sample()

    model = _DefaultModel(env)

    if real_time:
        tick_secs = env_config.get("simulation_config", {}).get("tick_secs", 1)
        interval = int(tick_secs * 1000)
        print(f"Real-time mode: interval set to {interval} ms")

    # Use env's full map limits (includes AWACS zones, same as standard view)
    map_limits = getattr(env, "full_map_limits", env.map_limits)

    # Create RL commands visualizer
    visualizer = RLCommandsVisualizer(
        env=env,
        sim=sim,
        map_limits=map_limits,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
    )

    print(f"Running RL commands visualization for {frames} frames...")

    # Run simulation with command visualization
    visualizer.run_simulation(
        model=model, num_frames=frames, interval=interval, save_video=save_video
    )


def main():
    """Main entry point for air2air-view-commands command."""
    parser = argparse.ArgumentParser(description="Live Visualization with RL Commands Panel")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to trained PPO checkpoint"
    )
    parser.add_argument(
        "--model-config", type=str, default=None, help="Path to model config YAML file"
    )
    parser.add_argument(
        "--train-config", type=str, default=None, help="Path to train config YAML file"
    )
    parser.add_argument(
        "--viz-config", type=str, default=None, help="Path to visualization config YAML file"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed (overrides interval)",
    )

    args = parser.parse_args()

    run_rl_commands_visualization(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        real_time=args.real_time,
    )


if __name__ == "__main__":
    main()
