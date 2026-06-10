"""
Public / baseline live visualization for BVR-MARL.

Renders a live 2D tactical view of the BVRMultiAgentEnv or
SimplifiedMultiAgentEnv.  No trained model or checkpoint is required:
agents act randomly by default, matching the public bvr-marl release behaviour.

When a checkpoint IS provided (via --checkpoint) the script attempts to load it
using an extension package's TrainedModelWrapper so that the same command works
seamlessly once the full extended stack is installed.

Entry point: bvr-view-public
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

# Fix OpenMP library conflict before any Ray/torch import.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")

from bvr_marl_core.simulator import Simulator  # noqa: E402
from bvr_marl_core.utils.config_loader import (  # noqa: E402
    load_train_config,
    load_viz_config,
    resolve_relative_path,
)
from bvr_marl_core.utils.paths import core_project_root  # noqa: E402
from bvr_marl_core.visualization.scenplotter.video_generation import (  # noqa: E402
    live_simulation,
)


class DefaultModel:
    """Fallback model that samples random actions — no checkpoint required."""

    def __init__(self, env):
        self._env = env

    def compute_single_action(self, observation, agent_id: str):  # noqa: ARG002
        space = self._env.action_space
        if hasattr(space, "sample"):
            return space.sample()
        return space[agent_id].sample()

    def compute_actions(self, obs_dict: dict) -> dict:
        return {aid: self.compute_single_action(obs, aid) for aid, obs in obs_dict.items()}


def _build_env(env_type: str, env_config: dict):
    """Instantiate the correct environment class."""
    if env_type.lower() == "simplified":
        from bvr_marl_core.rl.environment.gym.simplified_env import (  # noqa: PLC0415
            SimplifiedMultiAgentEnv,
        )

        print("Using SimplifiedMultiAgentEnv")
        return SimplifiedMultiAgentEnv(env_config)

    from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import (  # noqa: PLC0415
        BVRMultiAgentEnv,
    )

    print("Using BVRMultiAgentEnv")
    return BVRMultiAgentEnv(env_config)


def _load_model(env, checkpoint_path: str | None):
    """Return a model object.  Falls back to DefaultModel when no checkpoint given."""
    if checkpoint_path is None:
        print("No checkpoint provided — using random DefaultModel.")
        return DefaultModel(env)

    try:
        from bvr_marl_core.visualization.model_wrapper.model_wrapper import (  # noqa: PLC0415
            TrainedModelWrapper,
        )

        print(f"Loading checkpoint: {checkpoint_path}")
        return TrainedModelWrapper(checkpoint_path, env)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load checkpoint ({exc}). Falling back to DefaultModel.")
        return DefaultModel(env)


def run_public_view(
    checkpoint_path: str | None = None,
    train_config_path: str | None = None,
    viz_config_path: str | None = None,
    real_time: bool = False,
    save_video: bool = False,
    frames_override: int | None = None,
    interval_override: int | None = None,
) -> None:
    """Run the public baseline live visualization."""
    viz_config = load_viz_config(viz_config_path)

    # Resolve paths (command-line takes priority over viz config values)
    if checkpoint_path is None:
        checkpoint_path = viz_config.get("checkpoint_path")
    if train_config_path is None:
        train_config_path = viz_config.get("train_config_path")

    checkpoint_path = resolve_relative_path(checkpoint_path)
    train_config_path = resolve_relative_path(train_config_path)

    viz_settings = viz_config.get("visualization", {})
    frames = frames_override if frames_override is not None else viz_settings.get("frames", 200)
    interval = (
        interval_override if interval_override is not None else viz_settings.get("interval", 100)
    )
    save_video = save_video or viz_settings.get("save_video", False)
    symbol_mode = viz_settings.get("symbol_mode", "nato")
    dpi = viz_settings.get("dpi", 200)
    symbol_scale = viz_settings.get("symbol_scale", 2.0)
    show_text = viz_settings.get("show_text", True)
    env_type = viz_config.get("env_type", "bvr")

    # Load environment config from the training config
    train_config = load_train_config(train_config_path) if train_config_path else {}
    env_config = train_config.get("env", {})

    # Auto-detect simplified env from checkpoint path or training config
    if train_config.get("training_mode") == "simplified":
        env_type = "simplified"
    elif checkpoint_path and "Simplified" in str(checkpoint_path):
        env_type = "simplified"
        print("Auto-detected SimplifiedMultiAgentEnv from checkpoint path.")

    sim = Simulator()
    if real_time:
        sim_config = env_config.get("simulation_config", {})
        tick_secs = sim_config.get("tick_secs", 1)
        interval = int(tick_secs * 1000)

    env_config["simulator"] = sim
    env = _build_env(env_type, env_config)
    model = _load_model(env, checkpoint_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_output = core_project_root() / "output" / "videos" / f"public_view_{ts}.mp4"

    print(f"Running live visualization — {frames} frames …")
    live_simulation(
        trained_model=model,
        env=env,
        frames=frames,
        interval=interval,
        save_video=save_video,
        video_output_file=video_output,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
        show_radar_cones=(env_type.lower() != "simplified"),
        show_text=show_text,
    )


def main() -> int:
    """Entry point for ``bvr-view-public``."""
    parser = argparse.ArgumentParser(
        description=(
            "BVR-MARL public baseline visualization (random model by default; checkpoint optional)"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained PPO checkpoint (optional; random model used if omitted)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Path to the training config YAML used to create the environment",
    )
    parser.add_argument(
        "--viz-config",
        type=str,
        default=None,
        help="Path to a visualization config YAML (overrides defaults)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Run at simulation real-time speed",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save animation to output/videos/ as MP4",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to render (overrides viz config)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Animation interval in milliseconds (overrides viz config)",
    )
    args = parser.parse_args()

    try:
        run_public_view(
            checkpoint_path=args.checkpoint,
            train_config_path=args.train_config,
            viz_config_path=args.viz_config,
            real_time=args.real_time,
            save_video=args.save_video,
            frames_override=args.frames,
            interval_override=args.interval,
        )
        return 0
    except Exception as exc:
        print(f"Visualization failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
