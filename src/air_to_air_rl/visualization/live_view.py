"""
Live 2D visualization for BVR multi-agent environment.

This is the main entry point for running live visualization of trained models.
All functionality has been modularized into separate components.

"""

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

from air_to_air_rl.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from air_to_air_rl.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv
from air_to_air_rl.simulator.simulator import Simulator
from air_to_air_rl.utils.config_loader import load_train_config, load_viz_config
from air_to_air_rl.visualization.scenplotter.video_generation import live_simulation
from air_to_air_rl.visualization.utils.path_utils import resolve_relative_path

# Fix OpenMP library conflict - set before Ray/torch initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress Ray GPU environment variable warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress deprecation warnings from Ray/RLlib
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")


class DefaultModel:
    """Default model that takes random actions."""

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
        """Compute action for a single agent (for compatibility with video generation)."""
        if hasattr(self.env.action_space, "sample"):
            # Single action space
            return self.env.action_space.sample()
        else:
            # Multi-agent action space (dict)
            return self.env.action_space[agent_id].sample()


def run_live_view(
    checkpoint_path=None,
    model_config_path=None,
    train_config_path=None,
    viz_config_path=None,
    use_random=False,
    real_time=False,
    save_video=False,
    frames_override=None,
    interval_override=None,
    save_episode_logs=False,
):
    """
    Run live visualization of the BVR environment.

    Args:
        checkpoint_path: Path to trained model checkpoint (unused in public release)
        model_config_path: Path to model configuration file
        train_config_path: Path to training configuration file
        viz_config_path: Path to visualization configuration file
    """
    # Load visualization config
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
    frames = frames_override if frames_override is not None else viz_settings.get("frames", 100)
    interval = (
        interval_override if interval_override is not None else viz_settings.get("interval", 100)
    )
    save_video = save_video or viz_settings.get("save_video", False)
    symbol_mode = viz_settings.get(
        "symbol_mode", "nato"
    )  # "nato", "flying_objects", or "procedural"
    dpi = viz_settings.get("dpi", 200)  # render resolution; 200=default, 300=high, 400=ultra
    symbol_scale = viz_settings.get("symbol_scale", 2.0)  # sprite magnification; 2.0=default
    env_type = viz_config.get("env_type", "bvr")  # "bvr" or "simplified"
    real_time_speed = real_time or viz_settings.get("real_time_speed", False)

    # Load train config
    train_config = load_train_config(train_config_path)
    env_config = train_config.get("env", {})

    # Allow the training config to override env_type inferred from viz config.
    if train_config.get("training_mode") == "simplified":
        env_type = "simplified"
    elif checkpoint_path and "Simplified" in str(checkpoint_path):
        env_type = "simplified"
        print("Auto-detected SimplifiedMultiAgentEnv from checkpoint path.")

    # Initialize simulator and environment configuration
    sim = Simulator()
    env_config.get("map_size", 300)
    env_config.get("max_alt", 15000)

    # Add simulator to environment config
    env_config["simulator"] = sim

    # Override interval for real-time speed if enabled
    if real_time_speed:
        sim_config = env_config.get("simulation_config", {})
        tick_secs = sim_config.get("tick_secs", 1)
        interval = int(tick_secs * 1000)  # Convert to milliseconds

    # Select environment type
    if env_type.lower() == "simplified":
        env_class = SimplifiedMultiAgentEnv
        print("Using SimplifiedMultiAgentEnv")
    else:
        env_class = BVRMultiAgentEnv
        print("Using BVRMultiAgentEnv")

    # Initialize environment
    env = env_class(env_config)

    # Always use random model (checkpoint-based inference not available in public release)
    print("Using random model (scripted/checkpoint inference not included in public release)")
    model = DefaultModel(env)

    # Build a timestamped output path for video saving.
    from air_to_air_rl.core.paths import project_root

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _video_dir = project_root() / "output" / "videos"
    _video_output = _video_dir / f"live_view_{_ts}.mp4"

    # Run simulation
    print(f"Running live visualization for {frames} frames...")
    live_simulation(
        trained_model=model,
        env=env,
        frames=frames,
        interval=interval,
        save_video=save_video,
        video_output_file=_video_output,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
        show_radar_cones=(env_type.lower() != "simplified"),
    )


def main():
    """Main entry point for air2air-view command."""
    parser = argparse.ArgumentParser(description="Live Visualization for BVR Environment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PPO checkpoint (not used in public release)",
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
        "--random",
        action="store_true",
        help="Use random model (default behaviour)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed (overrides interval from config)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the animation as an MP4 (or GIF fallback) to output/videos/",
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Number of frames to render (overrides viz config)"
    )
    parser.add_argument(
        "--interval", type=int, default=None, help="Animation interval in ms (overrides viz config)"
    )
    parser.add_argument(
        "--save-episode-logs",
        action="store_true",
        help="Save detailed episode logs to CSV/JSON files",
    )
    parser.add_argument(
        "--no-save-episode-logs",
        action="store_true",
        help="Disable episode logging (default behavior)",
    )

    args = parser.parse_args()

    save_episode_logs = args.save_episode_logs and not args.no_save_episode_logs
    run_live_view(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        use_random=True,
        real_time=args.real_time,
        save_video=args.save_video,
        frames_override=args.frames,
        interval_override=args.interval,
        save_episode_logs=save_episode_logs,
    )


if __name__ == "__main__":
    main()
