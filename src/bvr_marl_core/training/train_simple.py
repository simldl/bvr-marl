"""
Training script for the lightweight SimplifiedMultiAgentEnv.

Observation: truth positions + missile warner (no radar pipeline).
Action space: [energy, lift_phi, lift_N, missile_trigger] (4-dim).
No AWACS, no gun, no countermeasures.  Uses default RLlib PPO — no custom
neural-network architecture.

Entry point: bvr-train-simple
"""

from __future__ import annotations

import argparse
import os
import platform
import tempfile
import warnings
from pathlib import Path

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

from bvr_marl_core.rl.environment.gym.simplified_env import SimplifiedMultiAgentEnv  # noqa: E402
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
from bvr_marl_core.rl.utils import create_simplified_env_creator  # noqa: E402
from bvr_marl_core.utils import apply_overrides, load_config  # noqa: E402
from bvr_marl_core.utils.paths import core_project_root as project_root  # noqa: E402
from bvr_marl_core.utils.paths import rl_configs_root  # noqa: E402


def policy_mapping_fn(agent_id, *_args, **_kwargs):
    """All agents share a single policy (symmetric self-play)."""
    return shared_policy_mapping_fn(agent_id, *_args, **_kwargs)


def setup_directories(cfg: dict) -> tuple[str, str]:
    """Return (save_dir, log_dir) as absolute paths."""
    save_dir = cfg["logging"]["save_dir"]
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(save_dir)
    log_dir = cfg["logging"].get("log_dir", save_dir)
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(log_dir)
    return save_dir, log_dir


def train_simple_main(cfg: dict) -> None:
    """Core simple training loop using SimplifiedMultiAgentEnv."""
    seed = cfg.get("seed", 42)
    set_random_seeds(seed)

    if torch.cuda.is_available() and cfg.get("num_gpus", 1) > 0:
        torch.cuda.set_device(0)

    if platform.system() == "Windows":
        _tmp_base = Path("C:/tmp")
        _tmp_base.mkdir(parents=True, exist_ok=True)
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="bvr_s_", dir=_tmp_base)).resolve())
    else:
        ray_tmp = str(Path(tempfile.mkdtemp(prefix="bvr_simple_ray_")).resolve())

    ray.init(
        _temp_dir=ray_tmp,
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={"env_vars": {"RADAR_FORCE_CPU": "1"}},
    )

    env_creator = create_simplified_env_creator(cfg)
    tune.register_env("SimplifiedMultiAgentEnv", env_creator)

    model_name = cfg.get("logging", {}).get("model_name", "bvr_simple")
    print("=" * 80)
    print("  PIPELINE     : Simple training (default RLlib PPO, no custom network)")
    print("  ENVIRONMENT  : SimplifiedMultiAgentEnv (truth obs, no radar)")
    print(f"  MODEL NAME   : {model_name}")
    print(f"  AGENTS/TEAM  : {cfg['env'].get('num_agents_per_team', '?')}")
    print("=" * 80)

    spec_env: SimplifiedMultiAgentEnv = env_creator({})
    spec_env.reset()
    obs_spaces = spec_env.observation_space
    act_spaces = spec_env.action_space

    ppo_config = build_ppo_config(
        cfg,
        obs_spaces,
        act_spaces,
        multi_spec=None,
        policy_mapping_fn=policy_mapping_fn,
        episode_callback=EpisodeMetricsCallback,
        env_name="SimplifiedMultiAgentEnv",
    )

    save_dir, _ = setup_directories(cfg)

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
            print(f"Loading weights from: {load_weights_from}")
        else:
            print(f"Starting new training. Target iterations: {total_iters}")

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
    """Entry point for ``bvr-train-simple``."""
    parser = argparse.ArgumentParser(
        description="BVR-MARL simple training (default RLlib PPO, SimplifiedMultiAgentEnv)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Config file path or bare name (default: basic)",
    )
    parser.add_argument("--config-path", type=str, help="Config directory")
    parser.add_argument("--overrides", nargs="*", help="Key=value overrides")
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

    try:
        train_simple_main(config.to_dict())
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
