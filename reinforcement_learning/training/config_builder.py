"""PPO configuration builder for BVR combat training."""

from typing import Dict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from omegaconf import DictConfig, OmegaConf


def print_automation_info(automation_level: int, wrapped_action_dim: int, active_indices: list,
                          wrapper_config: dict, enable_target_selection: bool, enable_gun: bool,
                          enable_countermeasures: bool):
    """Print information about automation level and action space configuration."""
    print("=" * 80)
    print(f"AUTOMATION LEVEL {automation_level} - Energy + Lift-Vector Control")
    print("=" * 80)
    print(f"  Network controls {wrapped_action_dim} actions:")
    print(f"    [0] Ps - Specific Energy Rate (climb/dive/accelerate)")
    print(f"    [1] n  - Normal Load Factor (turn intensity/g-load)")
    print(f"    [2] phi - Bank Angle (turn direction/roll)")
    if 3 in active_indices:
        print(f"    [3] TARGET SELECTION - choose which enemy to engage")
    print(f"    [4] Missile firing")
    print(f"  Automation handles {10 - wrapped_action_dim} actions:")
    if enable_target_selection:
        print(f"    [3] Target selection")
    if enable_gun:
        print(f"    [5] Gun firing")
    if enable_countermeasures:
        print(f"    [6-9] Countermeasures: flares, chaff, ECM, decoys")
    print(f"  Automation behavior: {wrapper_config.get('automation_level', 'balanced')}")
    print("=" * 80)


def configure_automation_level(automation_level: int, use_wrapper: bool, model_config_dict: dict,
                                wrapper_config: dict) -> tuple:
    """
    Configure action space based on automation level.

    Returns:
        Tuple of (wrapped_action_dim, active_indices, enable_target_selection,
                  enable_gun, enable_countermeasures)
    """
    if automation_level == 1:
        # Level 1: Basic flight control (1v1 training)
        wrapped_action_dim = 4
        active_indices = [0, 1, 2, 4]
        enable_target_selection = True
        enable_gun = True
        enable_countermeasures = True
    elif automation_level == 2:
        # Level 2: Add targeting control (multi-agent training)
        wrapped_action_dim = 5
        active_indices = [0, 1, 2, 3, 4]
        enable_target_selection = False  # Network controls it
        enable_gun = True
        enable_countermeasures = True
    elif automation_level == 3:
        # Level 3: Full control
        wrapped_action_dim = 10
        active_indices = list(range(10))
        enable_target_selection = False
        enable_gun = False
        enable_countermeasures = False
    else:
        raise ValueError(f"Invalid automation_level: {automation_level}. Must be 1, 2, or 3.")

    # Update model config
    if use_wrapper and automation_level < 3:
        model_config_dict.update({
            "use_neural_wrapper": True,
            "wrapped_action_dim": wrapped_action_dim,
            "full_action_dim": 10,
            "active_indices": active_indices,
            "action_dim": wrapped_action_dim,
        })
        print_automation_info(automation_level, wrapped_action_dim, active_indices,
                            wrapper_config, enable_target_selection, enable_gun, enable_countermeasures)
    else:
        # Full control mode
        model_config_dict.update({
            "use_neural_wrapper": False,
            "wrapped_action_dim": 10,
            "full_action_dim": 10,
            "active_indices": list(range(10)),
            "action_dim": 10,
        })
        print("=" * 80)
        print("AUTOMATION LEVEL 3 - FULL CONTROL MODE")
        print("=" * 80)
        print("  Network controls all 10 actions directly")
        print("=" * 80)

    return wrapped_action_dim, active_indices, enable_target_selection, enable_gun, enable_countermeasures


def build_ppo_config(cfg: DictConfig, obs_spaces: Dict, act_spaces: Dict,
                     multi_spec: MultiRLModuleSpec, policy_mapping_fn,
                     episode_callback) -> PPOConfig:
    """Build PPO configuration from hydra config."""
    seed = cfg.get("seed", 42)

    ppo_config = (
        PPOConfig()
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env="BVRMultiAgentEnv", env_config=cfg.env, clip_actions=True)
        .debugging(seed=seed)
        .rl_module(rl_module_spec=multi_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .env_runners(
            num_env_runners=cfg.get("num_env_runners", 30),
            num_envs_per_env_runner=cfg.get("num_envs_per_env_runner", 1),
            rollout_fragment_length=cfg.get("rollout_fragment_length", "auto"),
            batch_mode="truncate_episodes",
            sample_timeout_s=600.0,
        )
        .callbacks(episode_callback)
        .learners(
            num_learners=0,
            num_gpus_per_learner=1,
        )
        .resources(
            num_gpus=1,
        )
        .training(
            gamma=cfg.training.get("gamma", 0.995),
            lambda_=cfg.training.get("lambda", 0.95),
            use_gae=cfg.training.get("use_gae", True),
            use_critic=True,
            train_batch_size_per_learner=cfg.training.get("train_batch_size", 4096),
            minibatch_size=cfg.training.get("sgd_minibatch_size", 128),
            num_epochs=cfg.training.get("num_epochs", 1),
            vf_loss_coeff=cfg.training.get("vf_loss_coeff", 0.5),
            entropy_coeff=cfg.training.get("entropy_coef", 0.01),
            kl_coeff=cfg.training.get("kl_coeff", 0.2),
            clip_param=cfg.training.get("clip_param", 0.2),
            grad_clip=cfg.training.get("grad_clip", None),
            grad_clip_by="global_norm",
        )
    )

    return ppo_config
