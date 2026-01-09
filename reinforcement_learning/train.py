"""
Refactored training script for BVR air combat reinforcement learning.

This is the main entry point for training. The heavy lifting is delegated to
focused modules in the training/ directory.

Updated: 2025-12-10 11:18 - Fixed LSTM rollout shapes
"""

from __future__ import annotations
import os
import sys
import warnings
import platform

# ---------- Setup project path FIRST ----------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ray_tmp = os.path.abspath(os.path.join(project_root, "..", "ray_tmp"))
os.makedirs(ray_tmp, exist_ok=True)

# ---------- Setup environment ----------
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

# ---------- Now safe to import other modules ----------
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import ray
from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

# Import training modules
from reinforcement_learning.training.callbacks import (
    EpisodeMetricsCallback,
    ProgressCallback,
    SmartCheckpointCallback,
)
from reinforcement_learning.training.config_builder import (
    configure_automation_level,
    build_ppo_config,
)
from reinforcement_learning.training.checkpoint_utils import (
    process_checkpoint_resume,
    process_weight_loading,
)

# Environment and network imports
from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from reinforcement_learning.utils import create_env_creator
from reinforcement_learning.networks.rl_module.custom_multi_agent_model import CustomMultiAgentRLModule

# Import for random seed setting
import random
import numpy as np


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print(f"RANDOM SEED SET TO: {seed}")
    print("=" * 80)


def policy_mapping_fn(agent_id, *_args, **_kwargs):
    """All agents use the same policy for symmetric self-play."""
    return "shared_policy"


def create_rl_module_spec(cfg: DictConfig, obs_spaces, act_spaces):
    """Create RLModule specification for the network."""
    # Get representative agent (all agents have identical spaces)
    aid_agent = next(iter(obs_spaces.keys()))

    # Build model config
    model_config_dict = OmegaConf.to_container(cfg.model.model_config, resolve=True)

    # Configure automation level
    automation_level = cfg.model.get("automation_level", 1)
    use_wrapper = cfg.model.get("use_neural_wrapper", True)
    wrapper_config = OmegaConf.to_container(cfg.model.get("neural_wrapper_config", {}), resolve=True)

    configure_automation_level(automation_level, use_wrapper, model_config_dict, wrapper_config)

    # Create shared policy spec (self-play)
    shared_spec = RLModuleSpec(
        module_class=CustomMultiAgentRLModule,
        observation_space=obs_spaces[aid_agent],
        action_space=act_spaces[aid_agent],
        model_config=model_config_dict,
    )

    return MultiRLModuleSpec(rl_module_specs={"shared_policy": shared_spec})


def setup_directories(cfg: DictConfig) -> tuple[str, str]:
    """Setup and validate save/log directories."""
    save_dir = cfg.logging.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(project_root, "..", save_dir))

    log_dir = cfg.logging.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(os.path.join(project_root, "..", log_dir))

    return save_dir, log_dir


def create_tuner(cfg: DictConfig, ppo_config, run_config, experiment_dir):
    """Create Ray Tune Tuner for training."""
    if experiment_dir:
        # Resume from existing experiment
        return tune.Tuner.restore(
            path=experiment_dir,
            trainable="PPO",
            param_space=ppo_config.to_dict(),
            resume_unfinished=True,
            resume_errored=False,
        )
    else:
        # Start fresh training (or with loaded weights via PPO's restore mechanism)
        return tune.Tuner(
            trainable="PPO",
            param_space=ppo_config.to_dict(),
            run_config=run_config,
        )


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    """Main training entry point."""
    # Set random seeds
    seed = cfg.get("seed", 42)
    set_random_seeds(seed)

    # GPU setup
    if torch.cuda.is_available() and cfg.get("num_gpus", 1) > 0:
        torch.cuda.set_device(0)

    # Initialize Ray
    ray.init(
        _temp_dir=ray_tmp,  # <--- new
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "working_dir": project_root,
            "env_vars": {
                "PYTHONPATH": project_root,
                "RADAR_FORCE_CPU": "1",  # Force CPU for radars in workers to avoid CUDA conflicts
            }
        },
    )

    # Register environment and probe observation/action spaces
    env_creator = create_env_creator(cfg)
    tune.register_env("BVRMultiAgentEnv", env_creator)

    spec_env: BVRMultiAgentEnv = env_creator({})
    spec_env.reset()
    obs_spaces = spec_env.observation_space
    act_spaces = spec_env.action_space

    # Create RLModule specification
    multi_spec = create_rl_module_spec(cfg, obs_spaces, act_spaces)

    # Build PPO configuration
    ppo_config = build_ppo_config(
        cfg, obs_spaces, act_spaces, multi_spec,
        policy_mapping_fn, EpisodeMetricsCallback
    )

    # Setup directories
    save_dir, log_dir = setup_directories(cfg)

    # Handle checkpoint resumption
    total_training_iterations = int(cfg.training.get("steps", 10))
    experiment_dir, _ = process_checkpoint_resume(
        cfg.get("resume_checkpoint", None), project_root
    )

    # Handle weight loading from checkpoint (if not resuming full experiment)
    load_weights_from = None
    if not experiment_dir:
        load_weights_from = process_weight_loading(
            cfg.get("load_weights_from_checkpoint", None), project_root
        )

        if load_weights_from:
            print("=" * 80)
            print(f"LOADING WEIGHTS FROM CHECKPOINT: {load_weights_from}")
            print(f"Target iterations: {total_training_iterations}")
            print("=" * 80)
        else:
            print("=" * 80)
            print("STARTING NEW TRAINING WITH RANDOM WEIGHTS")
            print(f"Target iterations: {total_training_iterations}")
            print("=" * 80)

    # Setup callbacks
    progress_callback = ProgressCallback(total_training_iterations)
    checkpoint_callback = SmartCheckpointCallback(total_training_iterations, save_dir)

    # Add weight loading callback if needed
    callbacks_list = [progress_callback, checkpoint_callback]
    if load_weights_from:
        from reinforcement_learning.training.callbacks import WeightLoadingCallback
        weight_loader = WeightLoadingCallback(load_weights_from)
        callbacks_list.insert(0, weight_loader)  # Add first so it runs before other callbacks

    # Configure training run
    # Calculate checkpoint frequency (every 5% milestone)
    checkpoint_frequency = max(1, int(0.05 * total_training_iterations))

    print("=" * 80)
    print("CHECKPOINT CONFIGURATION")
    print(f"Checkpoint frequency: Every {checkpoint_frequency} iterations (~5%)")
    print(f"Checkpoints will be saved at: {save_dir}/{cfg.logging.model_name}")
    print(f"Retention policy: Keep all milestones (every 5%) + last 3 backups")
    print("=" * 80)

    run_config = tune.RunConfig(
        name=cfg.logging.model_name,
        storage_path=save_dir,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=checkpoint_frequency,  # Save every 5%
            num_to_keep=None,  # Let our callback decide what to keep
        ),
        log_to_file=True,
        verbose=0,
        callbacks=callbacks_list,
        stop={"training_iteration": total_training_iterations},
    )

    # Create and run tuner
    tuner = create_tuner(cfg, ppo_config, run_config, experiment_dir)
    result = tuner.fit()

    ray.shutdown()


if __name__ == "__main__":
    main()
