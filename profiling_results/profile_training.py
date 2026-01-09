"""
Profile a complete training run including RL overhead.

This profiles the full training pipeline:
- Environment simulation (reset, step)
- Policy network inference (forward passes)
- Value network inference
- PPO loss computation
- Gradient backpropagation
- Optimizer updates
- Ray worker communication

Usage:
    python profile_training.py --iterations 100
"""

import argparse
import cProfile
import pstats
from pathlib import Path

def profile_training(num_iterations: int = 100, config_name: str = "train_config_v4_aggressive"):
    """
    Profile a complete training run.

    Args:
        num_iterations: Number of training iterations to profile
        config_name: Name of the training configuration to use
    """
    print("="*80)
    print("TRAINING PROFILING")
    print("="*80)
    print(f"Configuration: {config_name}")
    print(f"Iterations: {num_iterations}")
    print("="*80)
    print()

    # Import here to avoid overhead in profiling setup
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray import tune
    import yaml
    import sys

    # Find project root and add to path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Load config
    config_path = project_root / "reinforcement_learning" / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Looked in: {config_path.absolute()}")
        return

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Import and register the environment
    from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
    from reinforcement_learning.rl_utils.rl_utils import create_env_creator
    from reinforcement_learning.networks.custom_multi_agent_model import CustomMultiAgentRLModule

    env_creator = create_env_creator(config_dict.get("env_config", {}))
    tune.register_env("BVRMultiAgentEnv", env_creator)
    print("Environment registered successfully")

    # Get observation and action spaces from the environment
    spec_env = env_creator({})
    spec_env.reset()
    obs_spaces = spec_env.observation_space
    act_spaces = spec_env.action_space

    # Get a representative agent ID (all agents have identical spaces)
    aid_agent = next(iter(obs_spaces.keys()))

    # Get model config from config dict
    model_config_dict = config_dict.get("model", {})

    # Create multi-agent RLModule spec (same as training)
    shared_spec = RLModuleSpec(
        module_class=CustomMultiAgentRLModule,
        observation_space=obs_spaces[aid_agent],
        action_space=act_spaces[aid_agent],
        model_config=model_config_dict,
    )
    multi_spec = MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": shared_spec,
        }
    )

    # Policy mapping function (same as training)
    def policy_mapping_fn(agent_id, *_args, **_kwargs):
        return "shared_policy"

    # Create PPO algorithm with minimal resources for profiling
    # Configuration matches the actual training setup
    print("Creating PPO algorithm...")
    ppo_config = (
        PPOConfig()
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="BVRMultiAgentEnv",
            env_config=config_dict.get("env_config", {}),
            clip_actions=True
        )
        .rl_module(rl_module_spec=multi_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .env_runners(
            num_env_runners=2,  # Minimal workers for profiling (vs 30 in production)
            rollout_fragment_length=128,
            batch_mode="complete_episodes",
            sample_timeout_s=600.0,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,  # CPU for profiling
        )
        .resources(
            num_gpus=0,  # CPU for profiling (vs 1 in production)
        )
        .training(
            gamma=config_dict.get("gamma", 0.995),
            lambda_=config_dict.get("lambda", 0.95),
            use_gae=config_dict.get("use_gae", True),
            use_critic=True,
            train_batch_size_per_learner=config_dict.get("train_batch_size", 4096),
            minibatch_size=config_dict.get("sgd_minibatch_size", 128),
            num_epochs=config_dict.get("num_epochs", 1),
            vf_loss_coeff=config_dict.get("vf_loss_coeff", 0.5),
            entropy_coeff=config_dict.get("entropy_coef", 0.01),
            kl_coeff=config_dict.get("kl_coeff", 0.2),
            clip_param=config_dict.get("clip_param", 0.2),
            grad_clip=config_dict.get("grad_clip", None),
            grad_clip_by="global_norm",
        )
        .evaluation(
            evaluation_interval=None,  # Disable evaluation during profiling
        )
    )

    print("Building algorithm...")
    algo = ppo_config.build()

    print(f"Algorithm created successfully")
    print(f"Number of workers: 2")
    print("="*80)
    print()

    print("Starting profiling...")
    print(f"Running {num_iterations} training iterations...")
    print()

    # Profile the training loop
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        for i in range(num_iterations):
            result = algo.train()

            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{num_iterations} completed")
                print(f"  Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
                print(f"  Training iteration time: {result.get('time_this_iter_s', 'N/A'):.2f}s")

    finally:
        profiler.disable()
        algo.stop()

    print()
    print("="*80)
    print("PROFILING RESULTS")
    print("="*80)
    print()

    # Save profiling stats
    output_dir = Path("profiling_results")
    output_dir.mkdir(exist_ok=True)

    stats_file = output_dir / "training_profile.stats"
    profiler.dump_stats(str(stats_file))

    # Generate report
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')

    # Print top functions by cumulative time
    print("Top 50 functions by CUMULATIVE time:")
    print("-" * 80)
    stats.print_stats(50)

    print("\n")
    print("Top 50 functions by TOTAL (self) time:")
    print("-" * 80)
    stats.sort_stats('tottime')
    stats.print_stats(50)

    print("\n")
    print("Caller/Callee relationships for top 20 functions:")
    print("-" * 80)
    stats.sort_stats('cumulative')
    stats.print_callers(20)

    # Save detailed report to file
    report_file = output_dir / "training_profile_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING PROFILING REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write("="*80 + "\n\n")

        f.write("Top 100 functions by CUMULATIVE time:\n")
        f.write("-" * 80 + "\n")
        stats_obj = pstats.Stats(str(stats_file), stream=f)
        stats_obj.strip_dirs()
        stats_obj.sort_stats('cumulative')
        stats_obj.print_stats(100)

        f.write("\n\n")
        f.write("Top 100 functions by TOTAL (self) time:\n")
        f.write("-" * 80 + "\n")
        stats_obj.sort_stats('tottime')
        stats_obj.print_stats(100)

        f.write("\n\n")
        f.write("Caller/Callee relationships for top 50 functions:\n")
        f.write("-" * 80 + "\n")
        stats_obj.sort_stats('cumulative')
        stats_obj.print_callers(50)

    print()
    print("="*80)
    print(f"Detailed profiling stats saved to: {stats_file}")
    print(f"Full profiling report saved to: {report_file}")
    print("="*80)
    print("PROFILING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile complete training run")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations to profile (default: 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_config_v4_aggressive",
        help="Training configuration name (default: train_config_v4_aggressive)"
    )

    args = parser.parse_args()

    profile_training(args.iterations, args.config)
