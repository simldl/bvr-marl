"""PPO configuration builder for BVR combat training."""

from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.ppo import PPOConfig


def build_ppo_config(
    cfg: dict,
    obs_spaces: dict,
    act_spaces: dict,
    policy_mapping_fn=None,
    episode_callback=None,
    env_name: str = "BVRMultiAgentEnv",
) -> PPOConfig:
    """Build PPO configuration from hydra config.

    Args:
        env_name: Registered Ray environment name. Defaults to "BVRMultiAgentEnv".
                  Pass "SimplifiedMultiAgentEnv" when using train_simplified.py.
    """
    seed = cfg.get("seed", 42)

    # Try to use legacy API stack for compatibility if new stack fails
    use_new_api = cfg.get("use_new_api_stack", True)

    ppo_config_builder = (
        PPOConfig()
        .framework("torch")
        .environment(env=env_name, env_config=cfg["env"], clip_actions=True)
        .debugging(seed=seed)
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
            num_gpus_per_learner=cfg.get("num_gpus", 1),
        )
        .resources(
            num_gpus=cfg.get("num_gpus", 1),
        )
        .training(
            lr=cfg["training"].get("learning_rate", 0.0003),
            gamma=cfg["training"].get("gamma", 0.995),
            lambda_=cfg["training"].get("lambda", 0.95),
            use_gae=cfg["training"].get("use_gae", True),
            use_critic=True,
            train_batch_size_per_learner=cfg["training"].get("train_batch_size", 4096),
            minibatch_size=cfg["training"].get("sgd_minibatch_size", 128),
            num_epochs=cfg["training"].get("num_epochs", 1),
            vf_loss_coeff=cfg["training"].get("vf_loss_coeff", 0.5),
            entropy_coeff=cfg["training"].get("entropy_coef", 0.01),
            kl_coeff=cfg["training"].get("kl_coeff", 0.2),
            clip_param=cfg["training"].get("clip_param", 0.2),
            grad_clip=cfg["training"].get("grad_clip", None),
            grad_clip_by="global_norm",
        )
    )

    # Configure API stack based on compatibility needs
    if use_new_api:
        ppo_config_builder = ppo_config_builder.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    else:
        ppo_config_builder = ppo_config_builder.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

    return ppo_config_builder
