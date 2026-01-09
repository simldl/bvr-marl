"""
Analyze profiling results and generate insights about performance bottlenecks.
"""
import pstats
import sys
from pathlib import Path


def analyze_profile_stats(stats_file="profiling_results/simulation_profile.stats"):
    """
    Load and analyze profiling statistics, highlighting key bottlenecks.
    """
    stats_path = Path(stats_file)

    if not stats_path.exists():
        print(f"ERROR: Stats file not found at {stats_path}")
        print("Please run profile_simulation.py first to generate profiling data.")
        return

    print("=" * 80)
    print("PROFILING ANALYSIS - BOTTLENECK DETECTION")
    print("=" * 80)

    # Load stats
    stats = pstats.Stats(str(stats_path))
    stats.strip_dirs()

    # Get total runtime
    total_time = 0.0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        total_time = max(total_time, ct)

    print(f"\nTotal profiled time: {total_time:.2f} seconds")
    print("=" * 80)

    # Analyze cumulative time (where time is spent including subfunctions)
    print("\nTOP 20 BOTTLENECKS (by cumulative time):")
    print("-" * 80)
    stats.sort_stats('cumulative')

    # Get top functions
    top_funcs = []
    for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
        filename, line, func_name = func
        pct = (ct / total_time * 100) if total_time > 0 else 0
        top_funcs.append((func_name, filename, ct, tt, pct, nc))

    # Print formatted table
    print(f"{'Function':<50} {'File':<30} {'Cum%':<8} {'Cum(s)':<10} {'Self(s)':<10} {'Calls':<10}")
    print("-" * 120)

    for func_name, filename, cum_time, self_time, pct, ncalls in top_funcs:
        # Truncate long names
        func_name = func_name[:48] if len(func_name) > 48 else func_name
        filename = filename[-28:] if len(filename) > 28 else filename

        print(f"{func_name:<50} {filename:<30} {pct:>6.1f}%  {cum_time:>9.3f}  {self_time:>9.3f}  {ncalls:>9}")

    # Analyze self time (time spent in function itself, not subfunctions)
    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY SELF TIME (excluding subfunctions):")
    print("-" * 80)
    stats.sort_stats('tottime')

    top_self_funcs = []
    for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
        filename, line, func_name = func
        pct = (tt / total_time * 100) if total_time > 0 else 0
        top_self_funcs.append((func_name, filename, tt, pct, nc))

    print(f"{'Function':<50} {'File':<30} {'Self%':<8} {'Self(s)':<10} {'Calls':<10}")
    print("-" * 120)

    for func_name, filename, self_time, pct, ncalls in top_self_funcs:
        func_name = func_name[:48] if len(func_name) > 48 else func_name
        filename = filename[-28:] if len(filename) > 28 else filename

        print(f"{func_name:<50} {filename:<30} {pct:>6.1f}%  {self_time:>9.3f}  {ncalls:>9}")

    # Identify expensive calls (high average time per call)
    print("\n" + "=" * 80)
    print("TOP 20 SLOWEST FUNCTIONS (by average time per call):")
    print("-" * 80)

    avg_time_funcs = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if nc > 0:  # Avoid division by zero
            avg_time = ct / nc
            if avg_time > 0.001:  # Only consider functions taking >1ms on average
                filename, line, func_name = func
                avg_time_funcs.append((func_name, filename, avg_time, nc, ct))

    avg_time_funcs.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Function':<50} {'File':<30} {'Avg(ms)':<10} {'Calls':<10} {'Total(s)':<10}")
    print("-" * 120)

    for func_name, filename, avg_time, ncalls, total_time in avg_time_funcs[:20]:
        func_name = func_name[:48] if len(func_name) > 48 else func_name
        filename = filename[-28:] if len(filename) > 28 else filename

        print(f"{func_name:<50} {filename:<30} {avg_time*1000:>9.2f}  {ncalls:>9}  {total_time:>9.3f}")

    # Category analysis
    print("\n" + "=" * 80)
    print("TIME BREAKDOWN BY CATEGORY:")
    print("-" * 80)

    categories = {
        "Simulation Physics": [],
        "Observation Processing": [],
        "Action Processing": [],
        "Reward Calculation": [],
        "Environment Management": [],
        "Numpy Operations": [],
        "Python Overhead": [],
        "Other": []
    }

    keywords = {
        "Simulation Physics": ["aircraft", "missile", "radar", "update", "physics", "dynamics", "geodetic", "kinematics"],
        "Observation Processing": ["observation", "obs", "build_obs", "get_obs", "normalize"],
        "Action Processing": ["action", "process_action", "apply_action", "command"],
        "Reward Calculation": ["reward", "calculate_reward", "compute_reward"],
        "Environment Management": ["reset", "step", "spawn", "init", "close"],
        "Numpy Operations": ["numpy", "ndarray", "array", "dot", "matmul", "linalg"],
        "Python Overhead": ["__", "isinstance", "getattr", "setattr", "dict", "list"]
    }

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func
        categorized = False

        for category, kws in keywords.items():
            if any(kw in func_name.lower() or kw in filename.lower() for kw in kws):
                categories[category].append(tt)
                categorized = True
                break

        if not categorized:
            categories["Other"].append(tt)

    category_totals = {cat: sum(times) for cat, times in categories.items()}
    category_totals = dict(sorted(category_totals.items(), key=lambda x: x[1], reverse=True))

    total_categorized = sum(category_totals.values())

    for category, time in category_totals.items():
        pct = (time / total_categorized * 100) if total_categorized > 0 else 0
        print(f"{category:<30} {time:>10.3f}s  ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)

    # Generate recommendations based on findings
    print("\nBased on the profiling data, consider the following optimizations:\n")

    # Check for numpy inefficiencies
    numpy_time = category_totals.get("Numpy Operations", 0)
    if numpy_time > total_time * 0.1:
        print("• HIGH numpy operation time detected:")
        print("  - Consider using in-place operations where possible")
        print("  - Vectorize operations instead of loops")
        print("  - Use contiguous arrays (np.ascontiguousarray)")

    # Check for observation processing
    obs_time = category_totals.get("Observation Processing", 0)
    if obs_time > total_time * 0.1:
        print("\n• HIGH observation processing time:")
        print("  - Consider caching computed observations")
        print("  - Optimize normalization routines")
        print("  - Reduce observation complexity if possible")

    # Check for physics simulation
    sim_time = category_totals.get("Simulation Physics", 0)
    if sim_time > total_time * 0.3:
        print("\n• Physics simulation is the main bottleneck:")
        print("  - This is expected for high-fidelity simulations")
        print("  - Consider using Numba JIT compilation for hot loops")
        print("  - Profile individual physics components separately")

    # Check for Python overhead
    python_time = category_totals.get("Python Overhead", 0)
    if python_time > total_time * 0.15:
        print("\n• HIGH Python overhead detected:")
        print("  - Consider using __slots__ for frequently created classes")
        print("  - Reduce dynamic attribute access")
        print("  - Use local variables instead of repeated lookups")

    print("\n" + "=" * 80)
    print("For detailed line-by-line profiling, consider using:")
    print("  - line_profiler: pip install line_profiler")
    print("  - Run: kernprof -l -v your_script.py")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation profiling results")
    parser.add_argument("--stats", type=str, default="profiling_results/simulation_profile.stats",
                        help="Path to .stats file")

    args = parser.parse_args()

    try:
        analyze_profile_stats(args.stats)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
