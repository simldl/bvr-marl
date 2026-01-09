"""
Analyze training profiling results and identify bottlenecks.

This script categorizes time spent in different parts of the training pipeline:
- Environment simulation
- Policy network inference
- Value network inference
- Loss computation
- Gradient computation
- Optimizer updates
- Ray/communication overhead

Usage:
    python analyze_training_profile.py
"""

import pstats
from pathlib import Path


def categorize_function(func_name: str, file_name: str) -> str:
    """Categorize a function based on its name and file."""

    # Environment simulation
    if any(x in file_name.lower() for x in ['simulator', 'bvr_multi_agent_env', 'aircraft', 'missile']):
        if 'step' in func_name or 'update' in func_name or 'do_tick' in func_name:
            return "Environment Simulation"

    # Policy/Value network
    if any(x in file_name.lower() for x in ['torch', 'model', 'network', 'policy']):
        if 'forward' in func_name:
            return "Neural Network Forward"
        if 'backward' in func_name:
            return "Gradient Computation"

    # Optimizer
    if 'optim' in file_name.lower() or 'optimizer' in func_name.lower():
        return "Optimizer Updates"

    # Loss computation
    if 'loss' in func_name.lower() or 'ppo' in file_name.lower():
        if any(x in func_name.lower() for x in ['loss', 'advantage', 'value_function']):
            return "Loss Computation"

    # Ray/RLlib overhead
    if any(x in file_name.lower() for x in ['ray', 'rllib', 'worker', 'rollout']):
        return "Ray/RLlib Overhead"

    # Observation/Action processing
    if any(x in func_name.lower() for x in ['observation', 'obs']):
        return "Observation Processing"
    if any(x in func_name.lower() for x in ['action', 'apply']):
        return "Action Processing"

    # Reward calculation
    if 'reward' in func_name.lower():
        return "Reward Calculation"

    return "Other"


def analyze_training_profile():
    """Analyze training profiling results."""

    stats_file = Path("profiling_results/training_profile.stats")

    if not stats_file.exists():
        print(f"Error: Profiling stats file not found: {stats_file}")
        print("Please run profile_training.py first.")
        return

    print("="*80)
    print("TRAINING PROFILING ANALYSIS - BOTTLENECK DETECTION")
    print("="*80)
    print()

    # Load stats
    stats = pstats.Stats(str(stats_file))
    stats.strip_dirs()

    total_time = stats.total_tt
    print(f"Total profiled time: {total_time:.2f} seconds")
    print("="*80)
    print()

    # Categorize time by component
    category_time = {}

    for func, data in stats.stats.items():
        file_name = func[0]
        func_name = func[2]
        cumtime = data[3]

        category = categorize_function(func_name, file_name)
        category_time[category] = category_time.get(category, 0) + cumtime

    # Print time breakdown
    print("TIME BREAKDOWN BY CATEGORY:")
    print("-" * 80)

    sorted_categories = sorted(category_time.items(), key=lambda x: x[1], reverse=True)
    for category, time in sorted_categories:
        percentage = (time / total_time) * 100
        print(f"{category:30s} {time:8.3f}s  ({percentage:5.1f}%)")

    print()
    print("="*80)
    print()

    # Find top bottlenecks
    print("TOP 20 BOTTLENECKS (by cumulative time):")
    print("-" * 80)
    print(f"{'Function':<50} {'File':<30} {'Cum%':>6} {'Cum(s)':>8} {'Self(s)':>8} {'Calls':>8}")
    print("-" * 116)

    stats.sort_stats('cumulative')

    bottlenecks = []
    for func, data in list(stats.stats.items())[:20]:
        file_name = func[0]
        line_num = func[1]
        func_name = func[2]

        ncalls = data[0]
        tottime = data[2]
        cumtime = data[3]

        cum_percent = (cumtime / total_time) * 100

        bottlenecks.append({
            'func': func_name,
            'file': file_name,
            'cum_percent': cum_percent,
            'cumtime': cumtime,
            'tottime': tottime,
            'ncalls': ncalls
        })

        print(f"{func_name[:50]:<50} {file_name[-30:]:<30} {cum_percent:5.1f}% {cumtime:8.3f} {tottime:8.3f} {ncalls:8}")

    print()
    print("="*80)
    print()

    # Find functions by self time
    print("TOP 20 FUNCTIONS BY SELF TIME (excluding subfunctions):")
    print("-" * 80)
    print(f"{'Function':<50} {'File':<30} {'Self%':>6} {'Self(s)':>8} {'Calls':>8}")
    print("-" * 108)

    stats.sort_stats('tottime')

    for func, data in list(stats.stats.items())[:20]:
        file_name = func[0]
        func_name = func[2]

        ncalls = data[0]
        tottime = data[2]

        self_percent = (tottime / total_time) * 100

        print(f"{func_name[:50]:<50} {file_name[-30:]:<30} {self_percent:5.1f}% {tottime:8.3f} {ncalls:8}")

    print()
    print("="*80)
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)
    print()

    # Analyze category breakdown
    env_time = category_time.get("Environment Simulation", 0)
    nn_time = category_time.get("Neural Network Forward", 0)
    grad_time = category_time.get("Gradient Computation", 0)
    opt_time = category_time.get("Optimizer Updates", 0)
    ray_time = category_time.get("Ray/RLlib Overhead", 0)

    if env_time > total_time * 0.4:
        print("⚠ HIGH environment simulation time detected:")
        print(f"  - Environment: {env_time:.2f}s ({env_time/total_time*100:.1f}%)")
        print("  - This is expected, but can be optimized further")
        print("  - See profiling_results/OPTIMIZATION_SUMMARY.md for environment optimizations")
        print()

    if nn_time > total_time * 0.2:
        print("⚠ HIGH neural network forward pass time:")
        print(f"  - Forward passes: {nn_time:.2f}s ({nn_time/total_time*100:.1f}%)")
        print("  - Consider: Smaller network, batch inference, model quantization")
        print()

    if grad_time > total_time * 0.15:
        print("⚠ HIGH gradient computation time:")
        print(f"  - Gradients: {grad_time:.2f}s ({grad_time/total_time*100:.1f}%)")
        print("  - Consider: Mixed precision training, gradient accumulation")
        print()

    if ray_time > total_time * 0.15:
        print("⚠ HIGH Ray/RLlib overhead:")
        print(f"  - Ray overhead: {ray_time:.2f}s ({ray_time/total_time*100:.1f}%)")
        print("  - Consider: Increase rollout_fragment_length, reduce num_workers")
        print()

    print("="*80)
    print()
    print("For detailed line-by-line profiling, consider using:")
    print("  - line_profiler: pip install line_profiler")
    print("  - Run: kernprof -l -v your_script.py")
    print("="*80)


if __name__ == "__main__":
    analyze_training_profile()
