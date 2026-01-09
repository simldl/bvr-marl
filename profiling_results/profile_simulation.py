"""
Profiling script for air-to-air RL simulation.
Uses cProfile to identify performance bottlenecks in the simulation.
"""
from __future__ import annotations
import cProfile
import pstats
import io
import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Environment setup for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Import after path setup
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


def profile_simulation_steps(num_episodes=3, steps_per_episode=100, config_name="train_config_v4_aggressive"):
    """
    Profile the simulation by running multiple episodes and steps.

    Args:
        num_episodes: Number of episodes to run
        steps_per_episode: Number of steps per episode
        config_name: Name of the config file to use (without .yaml extension)
    """
    print("=" * 80)
    print("SIMULATION PROFILING")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print("=" * 80)

    # Load config using Hydra
    hydra.initialize(version_base=None, config_path="../reinforcement_learning/configs")
    cfg = hydra.compose(config_name=config_name)

    # Create environment
    from reinforcement_learning.utils.env_creator import create_env_creator
    env_creator = create_env_creator(cfg)
    env = env_creator({})

    # Get the base environment (unwrap if wrapped)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    print("Environment created successfully")
    print(f"Number of agents: {len(base_env.all_agent_ids)}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.keys())}")
    print("=" * 80)
    print("\nStarting profiling...\n")

    # Create profiler
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run simulation
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in base_env.all_agent_ids}

        for step in range(steps_per_episode):
            # Sample random actions for each agent
            actions = {
                agent_id: env.action_space[agent_id].sample()
                for agent_id in base_env.all_agent_ids
            }

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            # Accumulate rewards
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward

            # Check if episode is done
            if all(terminateds.values()) or all(truncateds.values()):
                print(f"Episode {episode + 1}/{num_episodes} ended at step {step + 1}")
                break
        else:
            print(f"Episode {episode + 1}/{num_episodes} completed {steps_per_episode} steps")

        # Print episode summary
        avg_reward = np.mean(list(episode_reward.values()))
        print(f"  Average reward: {avg_reward:.2f}")

    # Stop profiling
    profiler.disable()

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    # Create string buffer for stats output
    s = io.StringIO()

    # Sort by cumulative time
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    print("\nTop 50 functions by CUMULATIVE time:")
    print("-" * 80)
    ps.print_stats(50)
    print(s.getvalue())

    # Reset buffer
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')

    print("\n" + "=" * 80)
    print("Top 50 functions by TOTAL (self) time:")
    print("-" * 80)
    ps.print_stats(50)
    print(s.getvalue())

    # Reset buffer for caller/callee analysis
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    print("\n" + "=" * 80)
    print("Caller/Callee relationships for top 20 functions:")
    print("-" * 80)
    ps.print_callers(20)
    print(s.getvalue())

    # Save detailed stats to file
    output_dir = project_root / "profiling_results"
    output_dir.mkdir(exist_ok=True)

    stats_file = output_dir / "simulation_profile.stats"
    profiler.dump_stats(str(stats_file))
    print(f"\nDetailed profiling stats saved to: {stats_file}")

    # Save human-readable report
    report_file = output_dir / "simulation_profile_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIMULATION PROFILING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Steps per episode: {steps_per_episode}\n")
        f.write("=" * 80 + "\n\n")

        # Cumulative time stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        f.write("Top 100 functions by CUMULATIVE time:\n")
        f.write("-" * 80 + "\n")
        ps.print_stats(100)
        f.write(s.getvalue())
        f.write("\n\n")

        # Total time stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('tottime')
        f.write("Top 100 functions by TOTAL (self) time:\n")
        f.write("-" * 80 + "\n")
        ps.print_stats(100)
        f.write(s.getvalue())
        f.write("\n\n")

        # Callers
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        f.write("Caller/Callee relationships:\n")
        f.write("-" * 80 + "\n")
        ps.print_callers(50)
        f.write(s.getvalue())

    print(f"Full profiling report saved to: {report_file}")

    # Cleanup
    env.close()
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

    return profiler


def profile_specific_components():
    """
    Profile specific components of the simulation in isolation.
    """
    print("\n" + "=" * 80)
    print("COMPONENT-SPECIFIC PROFILING")
    print("=" * 80)

    # Load config
    hydra.initialize(version_base=None, config_path="../reinforcement_learning/configs")
    cfg = hydra.compose(config_name="train_config_v4_aggressive")

    from reinforcement_learning.utils.env_creator import create_env_creator
    env_creator = create_env_creator(cfg)
    env = env_creator({})

    # Get the base environment (unwrap if wrapped)
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    # Reset to get initial state
    obs, info = env.reset()

    # Sample actions
    actions = {agent_id: env.action_space[agent_id].sample() for agent_id in base_env.all_agent_ids}

    print("\n1. Profiling environment RESET:")
    print("-" * 80)
    profiler_reset = cProfile.Profile()
    profiler_reset.enable()
    for _ in range(10):
        env.reset()
    profiler_reset.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler_reset, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    print("\n2. Profiling environment STEP:")
    print("-" * 80)
    profiler_step = cProfile.Profile()
    profiler_step.enable()
    for _ in range(50):
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        # Resample actions
        actions = {agent_id: env.action_space[agent_id].sample() for agent_id in base_env.all_agent_ids}
    profiler_step.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler_step, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    # Cleanup
    env.close()
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    print("\n" + "=" * 80)
    print("COMPONENT PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile air-to-air RL simulation")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--config", type=str, default="train_config_v4_aggressive",
                        help="Config file name (without .yaml)")
    parser.add_argument("--component", action="store_true",
                        help="Run component-specific profiling")

    args = parser.parse_args()

    try:
        # Full simulation profiling
        profile_simulation_steps(
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            config_name=args.config
        )

        # Component profiling if requested
        if args.component:
            profile_specific_components()

    except Exception as e:
        print(f"\nERROR during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
