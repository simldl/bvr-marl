"""
Training entry point for BVR-MARL air combat reinforcement learning.

Uses BVRMultiAgentEnv with Ray RLlib's default PPO model (no custom
neural-network architecture). For the advanced custom-network pipeline,
install an extension package that provides it.

Entry point: bvr-train
"""

from __future__ import annotations

import argparse
import os
import platform
import tempfile
import warnings
from pathlib import Path

# Configure distributed-training and OpenMP settings before Ray/torch loads.
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

import ray  # noqa: E402
import torch  # noqa: E402
import yaml as _yaml  # noqa: E402
from ray import tune  # noqa: E402

from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv  # noqa: E402
from bvr_marl_core.rl.training.adaptive_config import (  # noqa: E402
    create_adaptive_trainer,
    save_adaptive_config,
)
from bvr_marl_core.rl.training.callbacks import (  # noqa: E402
    EpisodeMetricsCallback,
    ProgressCallback,
    SmartCheckpointCallback,
)
from bvr_marl_core.rl.training.checkpoint_utils import (  # noqa: E402
    process_checkpoint_resume,
    process_weight_loading,
)
from bvr_marl_core.rl.training.config_builder import build_ppo_config  # noqa: E402
from bvr_marl_core.rl.training.runtime import (  # noqa: E402
    create_tuner,
    set_random_seeds,
    shared_policy_mapping_fn,
)
from bvr_marl_core.rl.utils import create_env_creator  # noqa: E402
from bvr_marl_core.utils import apply_overrides, load_config  # noqa: E402
from bvr_marl_core.utils.paths import core_project_root as project_root  # noqa: E402
from bvr_marl_core.utils.paths import rl_configs_root  # noqa: E402


def train_main(cfg: dict) -> None:
    """Core training loop using BVRMultiAgentEnv with default RLlib PPO."""
    seed = cfg.get("seed", 42)
    set_random_seeds(seed, announce=True)

    if torch.cuda.is_available() and cfg.get("num_gpus", 1) > 0:
        torch.cuda.set_device(0)

    if platform.system() == "Windows":
        _tmp_base = Path("C:/tmp")
        _tmp_base.mkdir(parents=True, exist_ok=True)
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="bvr_", dir=_tmp_base)).resolve())
    else:
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="bvr_ray_")).resolve())

    ray.init(
        _temp_dir=ray_tmp,
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={"env_vars": {"RADAR_FORCE_CPU": "1"}},
    )

    env_creator = create_env_creator(cfg)
    tune.register_env("BVRMultiAgentEnv", env_creator)

    model_name = cfg.get("logging", {}).get("model_name", "bvr_model")
    print("=" * 80)
    print("  PIPELINE     : Core training (default RLlib PPO, no custom network)")
    print("  ENVIRONMENT  : BVRMultiAgentEnv (full radar/AWACS/gun/countermeasures)")
    print(f"  MODEL NAME   : {model_name}")
    print(f"  AGENTS/TEAM  : {cfg['env'].get('num_agents_per_team', '?')}")
    print("=" * 80)

    spec_env: BVRMultiAgentEnv = env_creator({})
    spec_env.reset()
    obs_spaces = spec_env.observation_space
    act_spaces = spec_env.action_space

    # multi_spec=None → use RLlib's default PPO network (no custom RLModule)
    ppo_config = build_ppo_config(
        cfg,
        obs_spaces,
        act_spaces,
        multi_spec=None,
        policy_mapping_fn=shared_policy_mapping_fn,
        episode_callback=EpisodeMetricsCallback,
        env_name="BVRMultiAgentEnv",
    )

    save_dir = cfg["logging"]["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(save_dir)

    _model_dir = Path(save_dir) / model_name
    _model_dir.mkdir(parents=True, exist_ok=True)
    with open(_model_dir / "train_config.yaml", "w", encoding="utf-8") as _f:
        _yaml.dump(cfg, _f, default_flow_style=False, sort_keys=False)

    total_iters = int(cfg["training"].get("steps", 10))
    experiment_dir, _ = process_checkpoint_resume(cfg.get("resume_checkpoint"), str(project_root()))

    load_weights_from = None
    if not experiment_dir:
        load_weights_from = process_weight_loading(
            cfg.get("load_weights_from_checkpoint"), str(project_root())
        )
        if load_weights_from:
            print("=" * 80)
            print(f"LOADING WEIGHTS FROM: {load_weights_from}")
            print("=" * 80)
        else:
            print("=" * 80)
            print("STARTING NEW TRAINING WITH RANDOM WEIGHTS")
            print(f"Target iterations: {total_iters}")
            print("=" * 80)

    checkpoint_frequency = max(1, int(0.05 * total_iters))
    callbacks_list = [
        ProgressCallback(total_iters),
        SmartCheckpointCallback(total_iters, save_dir),
    ]
    if load_weights_from:
        from bvr_marl_core.rl.training.callbacks import WeightLoadingCallback  # noqa: PLC0415

        callbacks_list.insert(0, WeightLoadingCallback(load_weights_from))

    run_config = tune.RunConfig(
        name=model_name,
        storage_path=save_dir,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=checkpoint_frequency,
            num_to_keep=None,
        ),
        log_to_file=True,
        verbose=0,
        callbacks=callbacks_list,
        stop={"training_iteration": total_iters},
    )

    create_tuner(ppo_config, run_config, experiment_dir).fit()
    ray.shutdown()


def main() -> int:
    """Entry point for ``bvr-train``."""
    parser = argparse.ArgumentParser(
        description="BVR-MARL training (default RLlib PPO, BVRMultiAgentEnv)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Config file path or bare name (default: basic)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Config directory (default: configs/training/)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        help="Key=value overrides applied on top of the config",
    )
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

    config_name = args.config or "basic"
    config_path = args.config_path

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
    if not config_file.exists():
        pkg_fallback = rl_configs_root() / config_name
        if pkg_fallback.exists():
            config_file = pkg_fallback

    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: config not found: {config_file}")
        return 1
    except Exception as exc:
        print(f"Error loading config: {exc}")
        return 1

    if args.overrides:
        apply_overrides(config, args.overrides)

    cfg_dict = config.to_dict()

    if args.adaptive:
        print("=" * 80)
        print("ADAPTIVE TRAINING MODE ENABLED")
        print("=" * 80)
        try:
            adaptive_trainer = create_adaptive_trainer(str(config_file), auto_detect=True)
            if adaptive_trainer is None:
                print("User declined adaptive configuration, proceeding with original config...")
                return _train_with_fallback(cfg_dict, args.no_fallback)

            adaptive_cfg = adaptive_trainer.get_adaptive_config()
            print(f"Using adaptive configuration tier: {adaptive_trainer.current_tier}")

            save_path = (
                Path(cfg_dict["logging"]["save_dir"])
                / cfg_dict["logging"]["model_name"]
                / "adaptive_config.yaml"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_adaptive_config(adaptive_cfg, str(save_path), adaptive_trainer.current_tier)

            return _train_with_adaptive_fallback(adaptive_cfg, adaptive_trainer, args.no_fallback)
        except Exception as exc:
            print(f"Adaptive configuration failed: {exc}")
            print("Falling back to original configuration...")
            return _train_with_fallback(cfg_dict, args.no_fallback)
    else:
        return _train_with_fallback(cfg_dict, args.no_fallback)


def _train_with_fallback(cfg_dict: dict, no_fallback: bool = False) -> int:
    """Train with optional adaptive fallback on failure."""
    try:
        train_main(cfg_dict)
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}")
        if no_fallback:
            return 1

        print("Attempting adaptive fallback configuration...")
        try:
            import tempfile

            import yaml as _yaml_fb

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                _yaml_fb.dump(cfg_dict, tmp, default_flow_style=False)
                tmp_path = tmp.name

            adaptive_trainer = create_adaptive_trainer(tmp_path, auto_detect=False)
            if adaptive_trainer is None:
                print("Adaptive fallback not available")
                return 1
            return _train_with_adaptive_fallback(cfg_dict, adaptive_trainer, no_fallback)
        except Exception as fallback_exc:
            print(f"Adaptive fallback also failed: {fallback_exc}")
            return 1
        finally:
            import os as _os

            if "_tmp_path" in dir() and _os.path.exists(tmp_path):  # type: ignore[name-defined]
                _os.unlink(tmp_path)


def _train_with_adaptive_fallback(
    base_cfg: dict, adaptive_trainer, no_fallback: bool = False
) -> int:
    """Train with adaptive configuration and automatic tier fallback on failure."""
    current_cfg = base_cfg.copy()
    max_attempts = 1 if no_fallback else adaptive_trainer.max_fallback_attempts + 1

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                fallback_cfg = adaptive_trainer.get_fallback_config()
                if fallback_cfg is None:
                    print("No more fallback configurations available")
                    return 1
                current_cfg = fallback_cfg
                save_path = (
                    Path(current_cfg["logging"]["save_dir"])
                    / current_cfg["logging"]["model_name"]
                    / f"fallback_config_attempt_{attempt}.yaml"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_adaptive_config(current_cfg, str(save_path), adaptive_trainer.current_tier)

            print(
                f"Training attempt {attempt + 1}/{max_attempts} "
                f"with tier: {adaptive_trainer.current_tier}"
            )
            train_main(current_cfg)
            print(f"Training succeeded with tier: {adaptive_trainer.current_tier}")
            return 0
        except Exception as exc:
            print(f"Training attempt {attempt + 1} failed: {exc}")
            if attempt + 1 >= max_attempts:
                print("All training attempts failed")
                return 1
            print("Attempting with lower tier configuration...")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
