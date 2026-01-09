"""
Export TensorBoard metrics to high-quality plots.

This script reads TensorBoard event files and creates publication-quality
plots for episode returns, missiles fired, kills, and deaths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard is not installed!")
    print("Install with: pip install tensorboard")
    exit(1)


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 24,
        'font.family': 'sans-serif',
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 34,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 4,
        'lines.markersize': 10,
    })
    mplstyle.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def load_tensorboard_data(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load data from TensorBoard event files.

    Args:
        logdir: Path to directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    metrics = {}

    # Get all available scalar tags
    available_tags = ea.Tags()['scalars']
    print(f"\nAvailable metrics in TensorBoard:")
    for tag in sorted(available_tags):
        print(f"  - {tag}")

    # Extract relevant metrics - need to add ray/tune/ prefix
    metric_mappings = {
        'episode_return_mean': ['ray/tune/env_runners/episode_return_mean', 'episode_reward_mean'],
        'agent_return_A0': ['ray/tune/env_runners/agent_episode_returns_mean/A0'],
        'agent_return_A1': ['ray/tune/env_runners/agent_episode_returns_mean/A1'],
        'agent_return_B0': ['ray/tune/env_runners/agent_episode_returns_mean/B0'],
        'agent_return_B1': ['ray/tune/env_runners/agent_episode_returns_mean/B1'],
        'team_a_missiles': ['ray/tune/env_runners/team_a_missiles_fired'],
        'team_b_missiles': ['ray/tune/env_runners/team_b_missiles_fired'],
        'total_missiles': ['ray/tune/env_runners/total_missiles_fired'],
        'team_a_kills': ['ray/tune/env_runners/team_a_kills'],
        'team_b_kills': ['ray/tune/env_runners/team_b_kills'],
        'total_kills': ['ray/tune/env_runners/total_kills'],
        'team_a_deaths': ['ray/tune/env_runners/team_a_deaths'],
        'team_b_deaths': ['ray/tune/env_runners/team_b_deaths'],
        'total_deaths': ['ray/tune/env_runners/total_deaths'],
    }

    for metric_name, possible_tags in metric_mappings.items():
        for tag in possible_tags:
            if tag in available_tags:
                events = ea.Scalars(tag)
                metrics[metric_name] = [(e.step, e.value) for e in events]
                print(f"[OK] Loaded {metric_name}: {len(events)} data points")
                break
        else:
            print(f"[SKIP] Could not find metric: {metric_name}")

    return metrics


def smooth_data(data: List[Tuple[int, float]], window_size: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply moving average smoothing to data.

    Args:
        data: List of (step, value) tuples
        window_size: Size of smoothing window

    Returns:
        Tuple of (steps, raw_values, smoothed_values)
    """
    if not data:
        return np.array([]), np.array([]), np.array([])

    steps = np.array([d[0] for d in data])
    values = np.array([d[1] for d in data])

    # Apply moving average
    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
    steps_smoothed = steps[window_size - 1:]

    return steps, values, smoothed, steps_smoothed


def plot_episode_returns(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Plot episode returns over training."""
    if 'episode_return_mean' not in metrics:
        print("Skipping episode returns plot - no data")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    steps, raw, smoothed, steps_smoothed = smooth_data(metrics['episode_return_mean'], smoothing)

    # Plot raw data with low alpha
    ax.plot(steps, raw, alpha=0.2, color='steelblue', label='Raw')

    # Plot smoothed data
    ax.plot(steps_smoothed, smoothed, color='darkblue', linewidth=2.5, label=f'Smoothed (window={smoothing})')

    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero Line')

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Return over Training', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'episode_returns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_agent_returns(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Plot individual agent returns over training."""
    agent_metrics = {k: v for k, v in metrics.items() if k.startswith('agent_return_')}

    if not agent_metrics:
        print("Skipping agent returns plot - no data")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each agent
    colors = {
        'agent_return_A0': '#0066CC',  # Team A - Dark Blue
        'agent_return_A1': '#66B3FF',  # Team A - Light Blue
        'agent_return_B0': '#CC0000',  # Team B - Dark Red
        'agent_return_B1': '#FF6666',  # Team B - Light Red
    }

    for metric_key, data in agent_metrics.items():
        agent_id = metric_key.replace('agent_return_', '')
        steps, raw, smoothed, steps_smoothed = smooth_data(data, smoothing)

        color = colors.get(metric_key, 'gray')
        ax.plot(steps_smoothed, smoothed, color=color, linewidth=2.5,
                label=f'Agent {agent_id}', marker='o', markersize=3,
                markevery=max(1, len(steps_smoothed)//30))

    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero Line')

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Episode Return')
    ax.set_title('Individual Agent Returns over Training', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'agent_returns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_missiles_fired(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Plot missiles fired over training."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'team_a': '#0066CC', 'team_b': '#CC0000', 'total': '#000000'}

    for metric_key, color_key in [('team_a_missiles', 'team_a'),
                                    ('team_b_missiles', 'team_b'),
                                    ('total_missiles', 'total')]:
        if metric_key in metrics:
            steps, raw, smoothed, steps_smoothed = smooth_data(metrics[metric_key], smoothing)

            label = metric_key.replace('_', ' ').replace('missiles', '').title()
            ax.plot(steps_smoothed, smoothed, color=colors[color_key],
                   linewidth=2.5, label=label, marker='o', markersize=4, markevery=max(1, len(steps_smoothed)//30))

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Missiles Fired per Episode')
    ax.set_title('Missiles Fired over Training', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / 'missiles_fired.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_kills(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Plot kills over training."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'team_a': '#0066CC', 'team_b': '#CC0000', 'total': '#000000'}

    for metric_key, color_key in [('team_a_kills', 'team_a'),
                                    ('team_b_kills', 'team_b'),
                                    ('total_kills', 'total')]:
        if metric_key in metrics:
            steps, raw, smoothed, steps_smoothed = smooth_data(metrics[metric_key], smoothing)

            label = metric_key.replace('_', ' ').title()
            ax.plot(steps_smoothed, smoothed, color=colors[color_key],
                   linewidth=2.5, label=label, marker='s', markersize=4, markevery=max(1, len(steps_smoothed)//30))

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Kills per Episode')
    ax.set_title('Kills over Training', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / 'kills.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_deaths(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Plot deaths over training."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'team_a': '#0066CC', 'team_b': '#CC0000', 'total': '#000000'}

    for metric_key, color_key in [('team_a_deaths', 'team_a'),
                                    ('team_b_deaths', 'team_b'),
                                    ('total_deaths', 'total')]:
        if metric_key in metrics:
            steps, raw, smoothed, steps_smoothed = smooth_data(metrics[metric_key], smoothing)

            label = metric_key.replace('_', ' ').title()
            ax.plot(steps_smoothed, smoothed, color=colors[color_key],
                   linewidth=2.5, label=label, marker='^', markersize=4, markevery=max(1, len(steps_smoothed)//30))

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Deaths per Episode')
    ax.set_title('Deaths over Training', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / 'deaths.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_combined_metrics(metrics: Dict, output_dir: Path, smoothing: int = 10):
    """Create a combined 2x2 subplot figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Episode Returns
    ax1 = fig.add_subplot(gs[0, :])
    if 'episode_return_mean' in metrics:
        steps, raw, smoothed, steps_smoothed = smooth_data(metrics['episode_return_mean'], smoothing)
        ax1.plot(steps, raw, alpha=0.2, color='steelblue')
        ax1.plot(steps_smoothed, smoothed, color='darkblue', linewidth=2.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Episode Return')
        ax1.set_title('Episode Return', fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # 2. Missiles Fired
    ax2 = fig.add_subplot(gs[1, 0])
    for metric_key, color, label in [('team_a_missiles', '#0066CC', 'Team A'),
                                      ('team_b_missiles', '#CC0000', 'Team B')]:
        if metric_key in metrics:
            steps, raw, smoothed, steps_smoothed = smooth_data(metrics[metric_key], smoothing)
            ax2.plot(steps_smoothed, smoothed, color=color, linewidth=2, label=label)
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Missiles Fired')
    ax2.set_title('Missiles Fired', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 3. Kills
    ax3 = fig.add_subplot(gs[1, 1])
    for metric_key, color, label in [('team_a_kills', '#0066CC', 'Team A'),
                                      ('team_b_kills', '#CC0000', 'Team B')]:
        if metric_key in metrics:
            steps, raw, smoothed, steps_smoothed = smooth_data(metrics[metric_key], smoothing)
            ax3.plot(steps_smoothed, smoothed, color=color, linewidth=2, label=label)
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Kills per Episode')
    ax3.set_title('Kills', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    fig.suptitle('Training Metrics Summary', fontsize=28, fontweight='bold', y=0.995)

    output_path = output_dir / 'combined_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard metrics to plots')
    parser.add_argument('--logdir', type=str,
                       default=r'C:\Users\rl_main_simon\Code\air_to_air_rl\reinforcement_learning\models\bvr_v4_aggressive\PPO_BVRMultiAgentEnv_1c568_00000_0_2025-11-13_15-00-28',
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default='./tensorboard_exports',
                       help='Output directory for plots')
    parser.add_argument('--smoothing', type=int, default=20,
                       help='Smoothing window size')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (in addition to saving)')

    args = parser.parse_args()

    # Setup
    setup_plot_style()

    # Resolve paths
    script_dir = Path(__file__).parent
    logdir = Path(args.logdir)
    if not logdir.is_absolute():
        logdir = (script_dir / logdir).resolve()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TensorBoard Plot Exporter")
    print(f"{'='*60}")
    print(f"Log directory: {logdir}")
    print(f"Output directory: {output_dir}")
    print(f"Smoothing window: {args.smoothing}")
    print(f"{'='*60}\n")

    # Check if logdir exists
    if not logdir.exists():
        print(f"ERROR: Log directory not found: {logdir}")
        return

    # Load data
    print("Loading TensorBoard data...")
    metrics = load_tensorboard_data(str(logdir))

    if not metrics:
        print("\nERROR: No metrics found in TensorBoard logs!")
        return

    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}\n")

    # Generate individual plots
    plot_episode_returns(metrics, output_dir, args.smoothing)
    plot_agent_returns(metrics, output_dir, args.smoothing)
    plot_missiles_fired(metrics, output_dir, args.smoothing)
    plot_kills(metrics, output_dir, args.smoothing)
    plot_deaths(metrics, output_dir, args.smoothing)

    # Generate combined plot
    plot_combined_metrics(metrics, output_dir, args.smoothing)

    print(f"\n{'='*60}")
    print(f"[SUCCESS] All plots saved to: {output_dir}")
    print(f"{'='*60}\n")

    if args.show:
        print("Opening plots...")
        import webbrowser
        for img in output_dir.glob('*.png'):
            webbrowser.open(str(img))


if __name__ == '__main__':
    main()
