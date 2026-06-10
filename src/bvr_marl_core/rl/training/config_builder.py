"""PPO configuration builder for BVR combat training."""

import platform
import warnings

from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def print_automation_info(
    automation_level: int,
    wrapped_action_dim: int,
    active_indices: list,
    wrapper_config: dict,
    enable_target_selection: bool,
    enable_gun: bool,
    enable_countermeasures: bool,
):
    """Print information about automation level and action space configuration."""
    print("=" * 80)
    print(f"AUTOMATION LEVEL {automation_level} - Energy + Lift-Vector Control")
    print("=" * 80)
    print(f"  Network controls {wrapped_action_dim} actions:")
    print("    [0] Ps - Specific Energy Rate (climb/dive/accelerate)")
    print("    [1] n  - Normal Load Factor (turn intensity/g-load)")
    print("    [2] phi - Bank Angle (turn direction/roll)")
    if 3 in active_indices:
        print("    [3] TARGET SELECTION - choose which enemy to engage")
    print("    [4] Missile firing")
    print(f"  Automation handles {10 - wrapped_action_dim} actions:")
    if enable_target_selection:
        print("    [3] Target selection")
    if enable_gun:
        print("    [5] Gun firing")
    if enable_countermeasures:
        print("    [6-9] Countermeasures: flares, chaff, ECM, decoys")
    print(f"  Automation behavior: {wrapper_config.get('automation_level', 'balanced')}")
    print("=" * 80)


def configure_automation_level(
    automation_level: int, use_wrapper: bool, model_config_dict: dict, wrapper_config: dict
) -> tuple:
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
        model_config_dict.update(
            {
                "use_neural_wrapper": True,
                "wrapped_action_dim": wrapped_action_dim,
                "full_action_dim": 10,
                "active_indices": active_indices,
                "action_dim": wrapped_action_dim,
            }
        )
        print_automation_info(
            automation_level,
            wrapped_action_dim,
            active_indices,
            wrapper_config,
            enable_target_selection,
            enable_gun,
            enable_countermeasures,
        )
    else:
        # Full control mode
        model_config_dict.update(
            {
                "use_neural_wrapper": False,
                "wrapped_action_dim": 10,
                "full_action_dim": 10,
                "active_indices": list(range(10)),
                "action_dim": 10,
            }
        )
        print("=" * 80)
        print("AUTOMATION LEVEL 3 - FULL CONTROL MODE")
        print("=" * 80)
        print("  Network controls all 10 actions directly")
        print("=" * 80)

    return (
        wrapped_action_dim,
        active_indices,
        enable_target_selection,
        enable_gun,
        enable_countermeasures,
    )


def _resolve_num_learners(num_learners: int) -> int:
    """Return 0 on Windows: Ray Train uses NCCL for distributed learners, which is Linux-only."""
    if platform.system() == "Windows" and num_learners > 0:
        warnings.warn(
            f"num_learners={num_learners} is not supported on Windows (NCCL unavailable). "
            "Falling back to num_learners=0 (local learner in driver process).",
            stacklevel=3,
        )
        return 0
    return num_learners


def policy_ids_from_cfg(cfg: dict) -> set[str]:
    """Return the policy IDs implied by ``training.multi_agent``."""
    ma_cfg = cfg.get("training", {}).get("multi_agent", {})
    mode = ma_cfg.get("policy_mode", "shared")

    if mode == "shared":
        return {ma_cfg.get("shared_policy_id", "shared_policy")}

    if mode == "team_separate":
        return {
            ma_cfg.get("attacker_policy_id", "attacker_policy"),
            ma_cfg.get("defender_policy_id", "defender_policy"),
        }

    raise ValueError(f"Unknown policy_mode: {mode}")


def policies_to_train_from_cfg(cfg: dict) -> list[str]:
    """Return trainable policy IDs, honoring optional side-freezing flags."""
    ma_cfg = cfg.get("training", {}).get("multi_agent", {})
    mode = ma_cfg.get("policy_mode", "shared")

    if mode == "shared":
        return [ma_cfg.get("shared_policy_id", "shared_policy")]

    if mode == "team_separate":
        policies: list[str] = []
        if bool(ma_cfg.get("train_attacker", True)):
            policies.append(ma_cfg.get("attacker_policy_id", "attacker_policy"))
        if bool(ma_cfg.get("train_defender", True)):
            policies.append(ma_cfg.get("defender_policy_id", "defender_policy"))
        return policies

    raise ValueError(f"Unknown policy_mode: {mode}")


def build_ppo_config(
    cfg: dict,
    obs_spaces: dict,
    act_spaces: dict,
    multi_spec: MultiRLModuleSpec | None,
    policy_mapping_fn,
    episode_callback,
    env_name: str | type = "BVRMultiAgentEnv",
) -> PPOConfig:
    """Build PPO configuration from hydra config.

    Args:
        multi_spec: Custom RLModule spec for the shared policy.  Pass ``None``
                    to use RLlib's default PPO models (no custom network).
        env_name: Registered Ray environment name or importable env class.
                  Defaults to ``"BVRMultiAgentEnv"``. Pass an env class when
                  you want RLlib workers to reconstruct the env without relying
                  on Tune's string registry.
    """
    seed = cfg.get("seed", 42)

    # Use the new API stack only when a custom RLModule spec is provided.
    # Without a custom spec, RLlib falls back to DefaultPPOTorchRLModule which
    # uses PPOCatalog  and PPOCatalog has no encoder for Dict observation spaces.
    # The old API stack has built-in preprocessors that flatten Dict obs to a Box.
    use_new_api = multi_spec is not None
    training_cfg = cfg.get("training", {})
    num_env_runners = cfg.get(
        "num_env_runners",
        training_cfg.get("num_env_runners", training_cfg.get("n_envs", 30)),
    )
    num_envs_per_env_runner = cfg.get(
        "num_envs_per_env_runner",
        training_cfg.get("num_envs_per_env_runner", 1),
    )
    rollout_fragment_length = cfg.get(
        "rollout_fragment_length",
        training_cfg.get("rollout_fragment_length", "auto"),
    )
    train_batch_size = training_cfg.get("train_batch_size", training_cfg.get("batch_size", 4096))
    minibatch_size = training_cfg.get("sgd_minibatch_size", 128)
    policies = policy_ids_from_cfg(cfg)
    policies_to_train = policies_to_train_from_cfg(cfg)

    ppo_config = (
        PPOConfig()
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=use_new_api,
            enable_env_runner_and_connector_v2=use_new_api,
        )
        .environment(
            env=env_name,
            env_config=cfg["env"],
            normalize_actions=False,
            clip_actions=False,
        )
        .debugging(seed=seed)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="truncate_episodes",
            sample_timeout_s=600.0,
        )
        .callbacks(episode_callback)
        .learners(
            num_learners=_resolve_num_learners(cfg.get("num_learners", 0)),
            num_gpus_per_learner=cfg.get("num_gpus", 1),
        )
        .resources(
            num_gpus=cfg.get("num_gpus", 1),
        )
        .training(
            lr=training_cfg.get("learning_rate", 0.0003),
            gamma=training_cfg.get("gamma", 0.995),
            lambda_=training_cfg.get("lambda", 0.95),
            use_gae=training_cfg.get("use_gae", True),
            use_critic=True,
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=training_cfg.get("num_epochs", 1),
            vf_loss_coeff=training_cfg.get("vf_loss_coeff", 0.5),
            entropy_coeff=training_cfg.get("entropy_coef", 0.01),
            kl_coeff=training_cfg.get("kl_coeff", 0.2),
            clip_param=training_cfg.get("clip_param", 0.2),
            grad_clip=training_cfg.get("grad_clip", None),
            grad_clip_by="global_norm",
        )
    )

    if multi_spec is not None:
        ppo_config = ppo_config.rl_module(rl_module_spec=multi_spec)

    return ppo_config
