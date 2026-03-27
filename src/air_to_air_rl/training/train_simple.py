"""
Training script for the lightweight SimplifiedMultiAgentEnv.

Observation: truth positions + missile warner (no radar pipeline).
Action space: [energy, lift_phi, lift_N, missile_trigger] (4-dim).
No AWACS, no gun, no countermeasures.

"""

from __future__ import annotations

import argparse
import os
import platform
import random
import tempfile
import warnings
from pathlib import Path

# Set environment variables before importing Ray/torch — these affect distributed
# training initialization, OpenMP, and CUDA device selection.
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
if platform.system() == "Windows":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from ray import tune  # noqa: E402

from air_to_air_rl.core.paths import project_root, rl_configs_root  # noqa: E402
from air_to_air_rl.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv  # noqa: E402
from air_to_air_rl.rl.training.adaptive_config import (  # noqa: E402
    create_adaptive_trainer,
    save_adaptive_config,
)
from air_to_air_rl.rl.training.callbacks import (  # noqa: E402
    EpisodeMetricsCallback,
    ProgressCallback,
    SmartCheckpointCallback,
)
from air_to_air_rl.rl.training.checkpoint_utils import (  # noqa: E402
    process_checkpoint_resume,
    process_weight_loading,
)
from air_to_air_rl.rl.training.config_builder import build_ppo_config  # noqa: E402
from air_to_air_rl.rl.utils import create_simplified_env_creator  # noqa: E402
from air_to_air_rl.utils.simple_config import apply_overrides, load_config  # noqa: E402


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print(f"RANDOM SEED SET TO: {seed}")
    print("=" * 80)


def policy_mapping_fn(agent_id, *_args, **_kwargs):
    """All agents use the same policy for symmetric self-play."""
    return "shared_policy"


def setup_directories(cfg: dict) -> tuple[str, str]:
    """Setup and validate save/log directories."""
    save_dir = cfg["logging"]["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(save_dir)

    log_dir = cfg["logging"]["log_dir"]
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(log_dir)

    return save_dir, log_dir


def create_tuner(cfg: dict, ppo_config, run_config, experiment_dir):
    """Create Ray Tune Tuner for training."""
    if experiment_dir:
        return tune.Tuner.restore(
            path=experiment_dir,
            trainable="PPO",
            param_space=ppo_config.to_dict(),
            resume_unfinished=True,
            resume_errored=False,
        )
    else:
        return tune.Tuner(
            trainable="PPO",
            param_space=ppo_config.to_dict(),
            run_config=run_config,
        )


def train_simple_main(cfg: dict, adaptive_config=None):
    """Core simplified training logic."""
    # Use adaptive configuration if provided
    if adaptive_config:
        cfg = adaptive_config

    seed = cfg.get("seed", 42)
    set_random_seeds(seed)

    if torch.cuda.is_available() and cfg.get("num_gpus", 1) > 0:
        torch.cuda.set_device(0)

    if platform.system() == "Windows":
        _tmp_base = Path("C:/tmp")
        _tmp_base.mkdir(parents=True, exist_ok=True)
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="a2a_", dir=_tmp_base)).resolve())
    else:
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="air2air_simple_ray_")).resolve())

    ray.init(
        _temp_dir=ray_tmp,
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "env_vars": {
                "RADAR_FORCE_CPU": "1",
            }
        },
    )

    env_creator = create_simplified_env_creator(cfg)
    tune.register_env("SimplifiedMultiAgentEnv", env_creator)

    model_name = cfg["logging"]["model_name"] if "logging" in cfg else "bvr_model_simple"
    print("=" * 80)
    print("  ENVIRONMENT : SimplifiedMultiAgentEnv (truth obs, no radar)")
    print(f"  MODEL NAME  : {model_name}")
    print(f"  AGENTS/TEAM : {cfg['env'].get('num_agents_per_team', '?')}")
    print("=" * 80)

    spec_env: SimplifiedMultiAgentEnv = env_creator({})
    spec_env.reset()
    obs_spaces = spec_env.observation_space
    act_spaces = spec_env.action_space

    # Use Ray RLlib's default PPO network (no custom RLModule spec).
    ppo_config = build_ppo_config(
        cfg,
        obs_spaces,
        act_spaces,
        policy_mapping_fn,
        EpisodeMetricsCallback,
        env_name="SimplifiedMultiAgentEnv",
    )

    save_dir, log_dir = setup_directories(cfg)

    import yaml as _yaml

    _model_dir = Path(save_dir) / cfg["logging"]["model_name"]
    _model_dir.mkdir(parents=True, exist_ok=True)
    with open(_model_dir / "train_config.yaml", "w", encoding="utf-8") as _f:
        _yaml.dump(cfg, _f, default_flow_style=False, sort_keys=False)

    total_training_iterations = int(cfg["training"].get("steps", 10))
    experiment_dir, _ = process_checkpoint_resume(
        cfg.get("resume_checkpoint", None), str(project_root())
    )

    load_weights_from = None
    if not experiment_dir:
        load_weights_from = process_weight_loading(
            cfg.get("load_weights_from_checkpoint", None), str(project_root())
        )

        if load_weights_from:
            print("=" * 80)
            print(f"LOADING WEIGHTS FROM CHECKPOINT: {load_weights_from}")
            print(f"Target iterations: {total_training_iterations}")
            print("=" * 80)
        else:
            print("=" * 80)
            print("STARTING NEW SIMPLIFIED TRAINING WITH RANDOM WEIGHTS")
            print(f"Target iterations: {total_training_iterations}")
            print("=" * 80)

    progress_callback = ProgressCallback(total_training_iterations)
    checkpoint_callback = SmartCheckpointCallback(total_training_iterations, save_dir)

    callbacks_list = [progress_callback, checkpoint_callback]
    if load_weights_from:
        from air_to_air_rl.rl.training.callbacks import WeightLoadingCallback

        weight_loader = WeightLoadingCallback(load_weights_from)
        callbacks_list.insert(0, weight_loader)

    checkpoint_frequency = max(1, int(0.05 * total_training_iterations))

    print("=" * 80)
    print("SIMPLIFIED TRAINING CONFIGURATION")
    print(f"Checkpoint frequency: Every {checkpoint_frequency} iterations (~5%)")
    print(f"Checkpoints will be saved at: {save_dir}/{cfg['logging']['model_name']}")
    print("Retention policy: Keep all milestones (every 5%) + last 3 backups")
    print("=" * 80)

    run_config = tune.RunConfig(
        name=cfg["logging"]["model_name"],
        storage_path=save_dir,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=checkpoint_frequency,
            num_to_keep=None,
        ),
        log_to_file=True,
        verbose=0,
        callbacks=callbacks_list,
        stop={"training_iteration": total_training_iterations},
    )

    tuner = create_tuner(cfg, ppo_config, run_config, experiment_dir)
    tuner.fit()

    ray.shutdown()


def main():
    """Main entry point for air2air-train-simple command."""
    parser = argparse.ArgumentParser(description="Air-to-Air RL Simplified Training")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to config file or config name (default: simple)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to config directory (default: air_to_air_rl/rl/configs)",
    )
    parser.add_argument("--overrides", nargs="*", help="Config overrides in the format key=value")
    parser.add_argument(
        "--adaptive",
        "-a",
        action="store_true",
        help="Use adaptive training configuration based on system resources",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic fallback to lower tiers on training failure",
    )

    args = parser.parse_args()

    config_name = args.config or "simple"
    config_path = args.config_path

    # If config is a path, split it and resolve to absolute path
    if args.config and ("/" in args.config or "\\" in args.config):
        config_file_path = Path(args.config)
        if not config_file_path.is_absolute():
            config_file_path = project_root() / config_file_path
        config_path = str(config_file_path.parent)
        config_name = config_file_path.stem
    else:
        if config_path is None:
            top_level = project_root() / "configs" / "training"
            config_path = str(top_level if top_level.exists() else rl_configs_root())

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_file = Path(config_path) / config_name
    # Also try package defaults as fallback when using top-level dir
    if not config_file.exists():
        pkg_fallback = rl_configs_root() / config_name
        if pkg_fallback.exists():
            config_file = pkg_fallback

    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    overrides = args.overrides or []
    if overrides:
        apply_overrides(config, overrides)

    cfg_dict = config.to_dict()

    # Handle adaptive training
    if args.adaptive:
        print("=" * 80)
        print("ADAPTIVE SIMPLIFIED TRAINING MODE ENABLED")
        print("=" * 80)

        try:
            # Create adaptive trainer
            adaptive_trainer = create_adaptive_trainer(str(config_file), auto_detect=True)
            if adaptive_trainer is None:
                print("User declined adaptive configuration, proceeding with original config...")
                return train_simple_with_fallback(cfg_dict, args.no_fallback)

            # Get adaptive configuration - simplified environment prefers lower tiers
            adaptive_cfg = adaptive_trainer.get_adaptive_config()

            # Force simplified environment for adaptive configs
            if "env" not in adaptive_cfg:
                adaptive_cfg["env"] = {}
            adaptive_cfg["env"]["environment_class"] = "SimplifiedMultiAgentEnv"

            print(f"Using adaptive configuration tier: {adaptive_trainer.current_tier}")

            # Save adaptive configuration for reference
            save_path = (
                Path(cfg_dict["logging"]["save_dir"])
                / cfg_dict["logging"]["model_name"]
                / "adaptive_config.yaml"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_adaptive_config(adaptive_cfg, str(save_path), adaptive_trainer.current_tier)

            return train_simple_with_adaptive_fallback(
                adaptive_cfg, adaptive_trainer, args.no_fallback
            )

        except Exception as e:
            print(f"Adaptive configuration failed: {e}")
            print("Falling back to original configuration...")
            return train_simple_with_fallback(cfg_dict, args.no_fallback)
    else:
        return train_simple_with_fallback(cfg_dict, args.no_fallback)


def train_simple_with_fallback(cfg_dict: dict, no_fallback: bool = False):
    """Train simplified environment with fallback capability."""
    try:
        train_simple_main(cfg_dict)
        return 0
    except Exception as e:
        print(f"Simplified training failed: {e}")
        if no_fallback:
            return 1

        print("Attempting adaptive fallback configuration...")
        try:
            # Create adaptive trainer from failed config
            temp_config_path = "temp_simple_config.yaml"
            import yaml as _yaml

            with open(temp_config_path, "w") as f:
                _yaml.dump(cfg_dict, f, default_flow_style=False)

            adaptive_trainer = create_adaptive_trainer(temp_config_path, auto_detect=False)
            if adaptive_trainer is None:
                print("Adaptive fallback not available")
                return 1

            return train_simple_with_adaptive_fallback(cfg_dict, adaptive_trainer, no_fallback)

        except Exception as fallback_error:
            print(f"Adaptive fallback also failed: {fallback_error}")
            return 1
        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)


def train_simple_with_adaptive_fallback(
    base_cfg: dict, adaptive_trainer, no_fallback: bool = False
):
    """Train simplified environment with adaptive configuration and automatic fallback on failure."""
    current_cfg = base_cfg.copy()
    max_attempts = 1 if no_fallback else adaptive_trainer.max_fallback_attempts + 1

    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                # First attempt with current/adaptive config
                print(
                    f"Simplified training attempt {attempt + 1}/{max_attempts} with tier: {adaptive_trainer.current_tier}"
                )
            else:
                # Get fallback configuration
                fallback_cfg = adaptive_trainer.get_fallback_config()
                if fallback_cfg is None:
                    print("No more fallback configurations available")
                    return 1

                # Force simplified environment for fallback configs
                if "env" not in fallback_cfg:
                    fallback_cfg["env"] = {}
                fallback_cfg["env"]["environment_class"] = "SimplifiedMultiAgentEnv"

                current_cfg = fallback_cfg
                print(
                    f"Simplified training attempt {attempt + 1}/{max_attempts} with fallback tier: {adaptive_trainer.current_tier}"
                )

                # Save fallback configuration for reference
                save_path = (
                    Path(current_cfg["logging"]["save_dir"])
                    / current_cfg["logging"]["model_name"]
                    / f"fallback_config_attempt_{attempt}.yaml"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_adaptive_config(current_cfg, str(save_path), adaptive_trainer.current_tier)

            train_simple_main(current_cfg)
            print(
                f"Simplified training succeeded with configuration tier: {adaptive_trainer.current_tier}"
            )
            return 0

        except Exception as e:
            print(f"Simplified training attempt {attempt + 1} failed: {e}")
            if attempt + 1 >= max_attempts:
                print("All simplified training attempts failed")
                return 1
            else:
                print("Attempting with lower tier configuration...")

    return 1


if __name__ == "__main__":
    main()
