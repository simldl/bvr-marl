"""
Profiling script for BVR-MARL simulation.
Uses cProfile to identify performance bottlenecks in the simulation.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys

import numpy as np

from bvr_marl_core.utils.paths import core_project_root as _project_root

# Environment setup for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def _load_env(config_name: str):
    """Load config and create environment, returning (env, agent_ids)."""

    from bvr_marl_core.rl.utils import create_env_creator
    from bvr_marl_core.utils import load_config
    from bvr_marl_core.utils.paths import rl_configs_root

    top_level = _project_root() / "configs" / "training" / f"{config_name}.yaml"
    pkg_default = rl_configs_root() / f"{config_name}.yaml"
    config_file = top_level if top_level.exists() else pkg_default
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    cfg = load_config(config_file).to_dict()
    env_creator = create_env_creator(cfg)
    env = env_creator({})
    agent_ids = list(env.action_space.keys())
    return env, agent_ids


def profile_simulation_steps(num_episodes=3, steps_per_episode=100, config_name="basic"):
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

    env, agent_ids = _load_env(config_name)

    print("Environment created successfully")
    print(f"Number of agents: {len(agent_ids)}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.keys())}")
    print("=" * 80)
    print("\nStarting profiling...\n")

    profiler = cProfile.Profile()
    profiler.enable()

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in agent_ids}

        for step in range(steps_per_episode):
            actions = {agent_id: env.action_space[agent_id].sample() for agent_id in agent_ids}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward

            if all(terminateds.values()) or all(truncateds.values()):
                print(f"Episode {episode + 1}/{num_episodes} ended at step {step + 1}")
                break
        else:
            print(f"Episode {episode + 1}/{num_episodes} completed {steps_per_episode} steps")

        avg_reward = np.mean(list(episode_reward.values()))
        print(f"  Average reward: {avg_reward:.2f}")

    profiler.disable()

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    print("\nTop 50 functions by CUMULATIVE time:")
    print("-" * 80)
    ps.print_stats(50)
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("tottime")
    print("\n" + "=" * 80)
    print("Top 50 functions by TOTAL (self) time:")
    print("-" * 80)
    ps.print_stats(50)
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    print("\n" + "=" * 80)
    print("Caller/Callee relationships for top 20 functions:")
    print("-" * 80)
    ps.print_callers(20)
    print(s.getvalue())

    output_dir = _project_root() / "profiling_results"
    output_dir.mkdir(exist_ok=True)

    stats_file = output_dir / "simulation_profile.stats"
    profiler.dump_stats(str(stats_file))
    print(f"\nDetailed profiling stats saved to: {stats_file}")

    report_file = output_dir / "simulation_profile_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SIMULATION PROFILING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Steps per episode: {steps_per_episode}\n")
        f.write("=" * 80 + "\n\n")

        for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "TOTAL (self)")]:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.strip_dirs()
            ps.sort_stats(sort_key)
            f.write(f"Top 100 functions by {label} time:\n")
            f.write("-" * 80 + "\n")
            ps.print_stats(100)
            f.write(s.getvalue())
            f.write("\n\n")

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats("cumulative")
        f.write("Caller/Callee relationships:\n")
        f.write("-" * 80 + "\n")
        ps.print_callers(50)
        f.write(s.getvalue())

    print(f"Full profiling report saved to: {report_file}")

    env.close()

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

    return profiler


def profile_specific_components(config_name="basic"):
    """Profile specific components of the simulation in isolation."""
    print("\n" + "=" * 80)
    print("COMPONENT-SPECIFIC PROFILING")
    print("=" * 80)

    env, agent_ids = _load_env(config_name)
    obs, info = env.reset()
    actions = {agent_id: env.action_space[agent_id].sample() for agent_id in agent_ids}

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
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("\n2. Profiling environment STEP:")
    print("-" * 80)
    profiler_step = cProfile.Profile()
    profiler_step.enable()
    for _ in range(50):
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        actions = {agent_id: env.action_space[agent_id].sample() for agent_id in agent_ids}
    profiler_step.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler_step, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    env.close()

    print("\n" + "=" * 80)
    print("COMPONENT PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile BVR-MARL simulation")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument(
        "--config", type=str, default="basic", help="Config file name (without .yaml)"
    )
    parser.add_argument("--component", action="store_true", help="Run component-specific profiling")

    args = parser.parse_args()

    try:
        profile_simulation_steps(
            num_episodes=args.episodes, steps_per_episode=args.steps, config_name=args.config
        )
        if args.component:
            profile_specific_components(config_name=args.config)
    except Exception as e:
        print(f"\nERROR during profiling: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
