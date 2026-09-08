"""
Live 2D visualization for BVR multi-agent environment.

This is the main entry point for running live visualization of trained models.
All functionality has been modularized into separate components.

Import strategy
---------------
All heavy runtime imports are deferred to ``run_live_view()`` so that importing
this module is always lightweight and import-safe regardless of platform or
environment initialization order.  Only stdlib and the config-loader (which is
itself lightweight) are imported at module level.
"""

import argparse
import os
import warnings
from datetime import datetime

from bvr_marl_core.utils.config_loader import load_train_config, load_viz_config
from bvr_marl_core.visualization.scenario_overlays import (
    normalize_map_extents_mode,
    normalize_visualization_scenario,
)

# Fix OpenMP library conflict - set before Ray/torch initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress Ray GPU environment variable warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Suppress deprecation warnings from Ray/RLlib
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")


def run_live_view(
    checkpoint_path=None,
    model_config_path=None,
    train_config_path=None,
    viz_config_path=None,
    scenario_name=None,
    show_line_overlay=None,
    line_east_km=None,
    map_extents_mode=None,
    save_rewards=True,
    use_random=False,
    real_time=False,
    save_video=False,
    frames_override=None,
    interval_override=None,
    show_text=None,
):
    """
    Run live visualization of the BVR environment.

    Args:
        checkpoint_path: Path to trained model checkpoint
        model_config_path: Path to model configuration file
        train_config_path: Path to training configuration file
        viz_config_path: Path to visualization configuration file
        save_rewards: Whether to save detailed reward logs
    """
    # All heavy imports are deferred here so that:
    # (a) the module stays import-safe for smoke-test discovery, and
    # (b) imports only execute once the full runtime is ready, avoiding
    #     platform-specific initialization-order issues (e.g. Ray on Linux).
    import bvr_marl_core.visualization.reward_logging.reward_logger as _reward_logger_mod
    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from bvr_marl_core.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv
    from bvr_marl_core.simulator import Simulator
    from bvr_marl_core.utils.paths import core_project_root as project_root
    from bvr_marl_core.visualization import ensure_interactive_matplotlib_backend
    from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
        DefaultModel,
        TrainedModelWrapper,
    )
    from bvr_marl_core.visualization.utils.path_utils import resolve_relative_path
    from bvr_marl_core.visualization.utils.reward_tracking_wrapper import RewardTrackingEnvWrapper

    save_reward_logs = _reward_logger_mod.save_reward_logs

    # Load visualization config
    viz_config = load_viz_config(viz_config_path)

    # Use viz_config values if command-line arguments are not provided
    if checkpoint_path is None:
        checkpoint_path = viz_config.get("checkpoint_path")
    if model_config_path is None:
        model_config_path = viz_config.get("model_config_path")
    if train_config_path is None:
        train_config_path = viz_config.get("train_config_path")
    if scenario_name is None:
        scenario_name = viz_config.get("scenario")
    scenario_name = normalize_visualization_scenario(scenario_name)
    overlay_config = viz_config.get("overlay", {})
    if show_line_overlay is None:
        show_line_overlay = bool(overlay_config.get("show_line", False))
    if line_east_km is None:
        line_east_km = overlay_config.get("line_east_km", None)
    map_extents_mode = normalize_map_extents_mode(
        map_extents_mode if map_extents_mode is not None else overlay_config.get("map_extents_mode")
    )

    # Resolve relative paths to absolute paths
    checkpoint_path = resolve_relative_path(checkpoint_path)
    model_config_path = resolve_relative_path(model_config_path)
    train_config_path = resolve_relative_path(train_config_path)

    viz_settings = viz_config.get("visualization", {})
    frames = frames_override if frames_override is not None else viz_settings.get("frames", 100)
    interval = (
        interval_override if interval_override is not None else viz_settings.get("interval", 100)
    )
    save_video = save_video or viz_settings.get("save_video", False)
    save_rewards = viz_settings.get("save_rewards", save_rewards)  # Can be overridden by config
    symbol_mode = viz_settings.get(
        "symbol_mode", "nato"
    )  # "nato", "flying_objects", or "procedural"
    dpi = viz_settings.get("dpi", 200)  # render resolution; 200=default, 300=high, 400=ultra
    symbol_scale = viz_settings.get("symbol_scale", 2.0)  # sprite magnification; 2.0=default
    show_text = viz_settings.get("show_text", True) if show_text is None else bool(show_text)
    label_settings = viz_settings.get("labeling", {})
    focus_unit_id = label_settings.get("focus_unit_id")
    video_focus_rule = str(label_settings.get("video_focus_rule", "none"))
    env_type = viz_config.get("env_type", "bvr")  # "bvr" or "simplified"
    real_time_speed = real_time or viz_settings.get("real_time_speed", False)

    ensure_interactive_matplotlib_backend()
    import bvr_marl_core.visualization.scenplotter.video_generation as _video_gen_mod

    live_simulation = _video_gen_mod.live_simulation

    train_config = load_train_config(train_config_path)
    env_config = train_config.get("env", {})

    # Allow the training config to override env_type inferred from viz config.
    # training_mode: simplified → always use SimplifiedMultiAgentEnv regardless
    # of what the viz config says, so obs shapes match the loaded model.
    if train_config.get("training_mode") == "simplified":
        env_type = "simplified"
    elif checkpoint_path and "Simplified" in str(checkpoint_path):
        # Heuristic fallback: Ray Tune embeds the env class name in the run directory.
        # Catches cases where no train_config was passed but the checkpoint is simplified.
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
        # Get simulation config to find tick rate
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

    # Wrap with reward tracking if enabled
    if save_rewards:
        env = RewardTrackingEnvWrapper(env)

    # Initialize model
    if use_random:
        print("Using random model (--random flag specified)")
        model = DefaultModel(env)
    elif checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained model from: {checkpoint_path}")
        model = TrainedModelWrapper(
            checkpoint_path,
            env,
            model_config_path,
            train_config=train_config,
        )
    else:
        print("Using default model (random actions) - no checkpoint provided")
        model = DefaultModel(env)

    # Build a timestamped output path for video saving.
    # Use an absolute path anchored to the project root so the file always
    # lands in the right place regardless of the subprocess CWD.
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
        scenario_name=scenario_name,
        show_line_overlay=bool(show_line_overlay),
        line_east_km=float(line_east_km) if line_east_km is not None else None,
        map_extents_mode=map_extents_mode,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
        show_radar_cones=(env_type.lower() != "simplified"),
        show_text=show_text,
        focus_unit_id=focus_unit_id,
        video_focus_rule=video_focus_rule,
    )

    # Save reward logs after simulation completes
    if save_rewards and isinstance(env, RewardTrackingEnvWrapper):
        save_reward_logs(env.reward_history)


def main():
    """Main entry point for air2air-view command."""
    parser = argparse.ArgumentParser(description="Live Visualization for BVR Environment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PPO checkpoint (uses random model if not provided)",
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
        "--scenario",
        type=str,
        default=None,
        help="Scenario overlay key that overrides the visualization config",
    )
    parser.add_argument(
        "--show-line",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force a line-of-engagement overlay even for scenarios without one",
    )
    parser.add_argument(
        "--line-east-km",
        type=float,
        default=None,
        help="Move the line of engagement along the x-axis in kilometers",
    )
    parser.add_argument(
        "--map-extents",
        type=str,
        choices=["auto", "combat", "full"],
        default=None,
        help="Override whether the viewer shows the combat zone or full scenario bounds",
    )
    parser.add_argument(
        "--save-rewards", action="store_true", help="Save detailed reward logs to CSV/JSON"
    )
    parser.add_argument("--no-save-rewards", action="store_true", help="Disable reward logging")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random model (ignores checkpoint even if provided)",
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
        "--show-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show map text labels and status overlays",
    )

    args = parser.parse_args()

    # Determine save_rewards setting
    save_rewards_arg = None
    if args.save_rewards:
        save_rewards_arg = True
    elif args.no_save_rewards:
        save_rewards_arg = False

    run_live_view(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        scenario_name=args.scenario,
        show_line_overlay=args.show_line,
        line_east_km=args.line_east_km,
        map_extents_mode=args.map_extents,
        save_rewards=save_rewards_arg if save_rewards_arg is not None else True,
        use_random=args.random,
        real_time=args.real_time,
        save_video=args.save_video,
        frames_override=args.frames,
        interval_override=args.interval,
        show_text=args.show_text,
    )


if __name__ == "__main__":
    main()
