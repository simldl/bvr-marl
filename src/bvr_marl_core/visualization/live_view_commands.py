"""
Live 2D visualization with RL command time-series side panel.

Shows the 2D battle map on the left and scrolling line plots of
ps / n / phi (specific excess power, load factor, roll command)
for every active agent on the right — updated each simulation tick.

Heavy runtime imports (matplotlib, environment, simulator, model wrappers, visualizer) are
deferred to ``run_rl_commands_visualization()`` so that importing this module is lightweight
and import-safe for smoke-test discovery.
"""

import argparse
import os
import warnings

from bvr_marl_core.visualization.scenario_overlays import (
    normalize_map_extents_mode,
    normalize_visualization_scenario,
)

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
    scenario_name=None,
    show_line_overlay=None,
    line_east_km=None,
    map_extents_mode=None,
    real_time=False,
    show_text=None,
):
    """Run the 2D sim + RL command plots visualization."""
    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from bvr_marl_core.simulator import Simulator
    from bvr_marl_core.utils.config_loader import load_train_config, load_viz_config
    from bvr_marl_core.visualization.model_wrapper.model_wrapper import (
        DefaultModel,
        TrainedModelWrapper,
    )
    from bvr_marl_core.visualization.scenario_overlays import select_display_limits
    from bvr_marl_core.visualization.utils.path_utils import resolve_relative_path
    from bvr_marl_core.visualization.utils.rl_commands_visualizer import RLCommandsVisualizer

    # Load config
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

    # Get visualization settings
    viz_settings = viz_config.get("visualization", {})
    frames = viz_settings.get("frames", 100)
    interval = viz_settings.get("interval", 100)
    save_video = viz_settings.get("save_video", False)
    symbol_mode = viz_settings.get("symbol_mode", "nato")
    dpi = viz_settings.get("dpi", 200)
    symbol_scale = viz_settings.get("symbol_scale", 2.0)
    show_text = viz_settings.get("show_text", True) if show_text is None else bool(show_text)

    # Load train config
    train_config = load_train_config(train_config_path)
    env_config = train_config.get("env", {})

    # Initialize simulator and environment
    sim = Simulator()
    env_config["simulator"] = sim
    env = BVRMultiAgentEnv(env_config)

    # Initialize model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained model from: {checkpoint_path}")
        model = TrainedModelWrapper(checkpoint_path, env, model_config_path)
    else:
        print("Using default model (random actions)")
        model = DefaultModel(env)

    if real_time:
        tick_secs = env_config.get("simulation_config", {}).get("tick_secs", 1)
        interval = int(tick_secs * 1000)
        print(f"Real-time mode: interval set to {interval} ms")

    # Use env's full map limits (includes AWACS zones, same as standard view)
    map_limits = select_display_limits(env, scenario_name, map_extents_mode=map_extents_mode)

    # Create RL commands visualizer
    visualizer = RLCommandsVisualizer(
        env=env,
        sim=sim,
        map_limits=map_limits,
        scenario_name=scenario_name,
        force_line_overlay=bool(show_line_overlay),
        line_east_km=float(line_east_km) if line_east_km is not None else None,
        map_extents_mode=map_extents_mode,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
        show_text=show_text,
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
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed (overrides interval)",
    )
    parser.add_argument(
        "--show-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show map text labels and status overlays",
    )

    args = parser.parse_args()

    run_rl_commands_visualization(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        viz_config_path=args.viz_config,
        scenario_name=args.scenario,
        show_line_overlay=args.show_line,
        line_east_km=args.line_east_km,
        map_extents_mode=args.map_extents,
        real_time=args.real_time,
        show_text=args.show_text,
    )


if __name__ == "__main__":
    main()
