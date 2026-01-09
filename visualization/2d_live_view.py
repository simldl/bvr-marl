"""
Live 2D visualization for BVR multi-agent environment.

This script automatically uses the correct config files and relative paths.
Just run: python visualization/2d_live_view.py

The script will:
1. Automatically find and use visualization/viz_config.yaml
2. Use relative paths for all file loading
3. Work from any location as long as you run it from the project root

Optional arguments:
  --checkpoint PATH    Override checkpoint path from config
  --frames N          Override number of frames to simulate
  --save-video        Save visualization as video file
"""
import sys
import os
from pathlib import Path
import warnings
import argparse

# Fix OpenMP library conflict - set before importing any libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress Ray GPU environment variable warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress deprecation warnings from Ray/RLlib
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")

# Setup project root path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import modular components
from visualization.config.config_loader import load_train_config, load_viz_config
from visualization.model_wrapper.model_wrapper import DefaultModel, TrainedModelWrapper
from visualization.utils.reward_tracking_wrapper import RewardTrackingEnvWrapper
from visualization.reward_logging.reward_logger import save_reward_logs
from visualization.scenplotter.video_generation import live_simulation

# Import core simulation components
from simulator.simulator import Simulator
from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from aircrafts.types.eurofighter import Eurofighter


def run_visualization(
    checkpoint_path=None,
    model_config_path=None,
    train_config_path=None,
    viz_config_path=None,
    frames=None,
    interval=None,
    save_video=False,
    save_rewards=None
):
    """
    Run live visualization of the BVR environment.

    Args:
        checkpoint_path: Path to trained model checkpoint (relative to project root)
        model_config_path: Path to model configuration file
        train_config_path: Path to training configuration file
        viz_config_path: Path to visualization configuration file
        frames: Number of frames to simulate
        interval: Interval between frames in milliseconds
        save_video: Whether to save the visualization as a video
        save_rewards: Whether to save detailed reward logs
    """
    print("=" * 80)
    print("BVR AIR COMBAT 2D VISUALIZATION")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("=" * 80)

    # Load visualization config (default to viz_config.yaml in visualization folder)
    if viz_config_path is None:
        viz_config_path = project_root / "visualization" / "viz_config.yaml"
    else:
        viz_config_path = project_root / viz_config_path

    viz_config = load_viz_config(str(viz_config_path))
    print(f"Loaded viz config: {viz_config_path}")

    # Use command line args if provided, otherwise use viz_config
    if checkpoint_path is None:
        checkpoint_path = viz_config.get('checkpoint_path')
    if train_config_path is None:
        train_config_path = viz_config.get('train_config_path')
    if model_config_path is None:
        model_config_path = viz_config.get('model_config_path')

    # Convert paths to absolute (relative to project root)
    if checkpoint_path:
        checkpoint_path = project_root / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    if model_config_path:
        model_config_path = project_root / model_config_path if not Path(model_config_path).is_absolute() else Path(model_config_path)
    if train_config_path:
        train_config_path = project_root / train_config_path if not Path(train_config_path).is_absolute() else Path(train_config_path)
    else:
        # Default to train_config.yaml
        train_config_path = project_root / "reinforcement_learning" / "configs" / "train_config.yaml"

    print(f"Training config: {train_config_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint - using random actions")

    # Get visualization settings from config or command line
    viz_settings = viz_config.get('visualization', {})
    if frames is None:
        frames = viz_settings.get('frames', 100)
    if interval is None:
        interval = viz_settings.get('interval', 100)
    if save_rewards is None:
        save_rewards = viz_settings.get('save_rewards', False)

    real_time_speed = viz_settings.get('real_time_speed', False)

    # Load train config
    train_config = load_train_config(str(train_config_path))
    env_config = train_config.get('env', {})

    # Initialize simulator and environment configuration
    sim = Simulator()
    map_size = env_config.get('map_size', 300)
    max_alt = env_config.get('max_alt', 15000)

    # Override interval for real-time speed if enabled
    if real_time_speed:
        sim_config = env_config.get('simulation_config', {})
        tick_secs = sim_config.get('tick_secs', 1)
        interval = int(tick_secs * 1000)  # Convert seconds to milliseconds
        print(f"Real-time speed enabled: using {tick_secs}s tick rate (interval={interval}ms)")

    print(f"Simulation settings: {frames} frames, {interval}ms interval")
    print("=" * 80)

    # Override aircraft types to use Eurofighter for all aircraft
    agent_aircraft = Eurofighter
    opponent_aircraft = Eurofighter

    # Get simulation config
    sim_config = env_config.get('simulation_config', {})

    # Get number of agents per side from training config
    num_agents_per_side = env_config.get('num_agents_per_side', 2)

    # Build environment configuration
    config = {
        "simulator": sim,
        "num_agents_per_team": num_agents_per_side,
        "map_size": map_size,
        "max_alt": max_alt,
        "max_steps": env_config.get('max_steps', 128),
        "debug": env_config.get('debug', True),
        "simulation_config": sim_config,
        "weapon_config": env_config.get('weapon_config', {}),
        "aircraft_types": {
            "agent": agent_aircraft,
            "opponent": opponent_aircraft,
        },
        # Slot definitions for observation_space - must match checkpoint exactly
        "num_fm": env_config.get('num_fm', 4),
        "num_ff": env_config.get('num_ff', 2),
        "num_em": env_config.get('num_em', 4),
        "num_enemy": env_config.get('num_enemy', 2),
        "pr_slots": env_config.get('pr_slots', 2),
        "warn_sectors": env_config.get('warn_sectors', 4),
    }

    print(f"Environment: {map_size}km x {map_size}km map, {num_agents_per_side}v{num_agents_per_side}")
    print(f"Aircraft: {agent_aircraft.__name__} vs {opponent_aircraft.__name__}")
    print("=" * 80)

    # Create environment
    env = BVRMultiAgentEnv(config)

    # Wrap environment with reward tracking if save_rewards is enabled
    if save_rewards:
        env = RewardTrackingEnvWrapper(env)
        print("Reward tracking enabled - will save detailed logs after episode")

    # Debug: Check observation space after environment creation
    obs, _ = env.reset()
    first_agent = list(obs.keys())[0]
    first_obs = obs[first_agent]
    total_features = sum(v.shape[0] if hasattr(v, 'shape') else len(v) for v in first_obs.values())
    print(f"Observation space: {total_features} features")
    print("=" * 80)

    # Load trained model if checkpoint provided, otherwise use random actions
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading model from checkpoint...")
        model = TrainedModelWrapper(str(checkpoint_path), env, str(model_config_path) if model_config_path else None)
        print("Model loaded successfully")
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using random actions")
        model = DefaultModel(env)

    print("=" * 80)
    print("STARTING VISUALIZATION")
    print("=" * 80)

    env.model = model

    # Run live simulation
    live_simulation(
        trained_model=model,
        env=env,
        frames=frames,
        interval=interval,
        save_video=save_video
    )

    # Save reward logs after simulation completes
    if save_rewards and isinstance(env, RewardTrackingEnvWrapper):
        print("=" * 80)
        print("Saving reward logs...")
        save_reward_logs(env.reward_history)
        print("Reward logs saved")
        print("=" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="2D Live Visualization for BVR Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (random actions)
  python visualization/2d_live_view.py

  # Run with specific checkpoint
  python visualization/2d_live_view.py --checkpoint models/my_model/checkpoint_000010

  # Run with custom settings
  python visualization/2d_live_view.py --frames 200 --save-video
        """
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained model checkpoint (relative to project root)")
    parser.add_argument("--model-config", type=str, default=None,
                       help="Path to model config YAML file")
    parser.add_argument("--train-config", type=str, default=None,
                       help="Path to train config YAML file")
    parser.add_argument("--viz-config", type=str, default=None,
                       help="Path to visualization config YAML file")
    parser.add_argument("--frames", type=int, default=None,
                       help="Number of frames to simulate")
    parser.add_argument("--interval", type=int, default=None,
                       help="Interval between frames in milliseconds")
    parser.add_argument("--save-video", action='store_true',
                       help="Save visualization as video file")
    parser.add_argument("--save-rewards", action='store_true',
                       help="Save detailed reward logs to CSV/JSON")
    parser.add_argument("--no-save-rewards", action='store_true',
                       help="Disable reward logging")

    args = parser.parse_args()

    # Determine save_rewards setting
    save_rewards_arg = None
    if args.save_rewards:
        save_rewards_arg = True
    elif args.no_save_rewards:
        save_rewards_arg = False

    run_visualization(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        frames=args.frames,
        interval=args.interval,
        save_video=args.save_video,
        save_rewards=save_rewards_arg
    )


if __name__ == "__main__":
    main()
