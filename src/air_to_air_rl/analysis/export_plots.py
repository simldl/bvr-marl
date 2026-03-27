"""
Export TensorBoard metrics to high-quality plots.

This script reads TensorBoard event files and creates publication-quality
plots for episode returns, missiles fired, kills, and deaths.

"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard is not installed!")
    print("Install with: pip install tensorboard")
    exit(1)


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "font.size": 24,
            "font.family": "sans-serif",
            "axes.labelsize": 28,
            "axes.titlesize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 34,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 4,
            "lines.markersize": 10,
        }
    )
    # Use fallback if seaborn style is not available
    if "seaborn-v0_8-darkgrid" in plt.style.available:
        mplstyle.use("seaborn-v0_8-darkgrid")
    else:
        plt.style.use("default")


def load_tensorboard_data(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """
    Load data from TensorBoard event files.

    Args:
        logdir: Directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))

    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {logdir}")

    print(f"Found {len(event_files)} event files")

    # Load data from event files
    data = {}
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={
                    event_accumulator.SCALARS: 0,  # Load all scalars
                },
            )
            ea.Reload()

            # Extract scalar data
            for scalar_tag in ea.Tags()["scalars"]:
                scalar_events = ea.Scalars(scalar_tag)
                values = [(se.step, se.value) for se in scalar_events]

                if scalar_tag not in data:
                    data[scalar_tag] = []
                data[scalar_tag].extend(values)

        except Exception as e:
            print(f"Warning: Could not process {event_file}: {e}")

    # Sort data by step for each metric
    for metric in data:
        data[metric] = sorted(data[metric], key=lambda x: x[0])

    return data


def smooth_curve(values: list[float], smoothing: float = 0.9) -> list[float]:
    """Apply exponential smoothing to a curve."""
    if not values:
        return []

    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + value * (1 - smoothing))

    return smoothed


def create_metric_plot(
    data: dict[str, list[tuple[int, float]]],
    metric_keys: list[str],
    title: str,
    ylabel: str,
    output_path: str,
    smoothing: float = 0.9,
):
    """Create and save a plot for specific metrics."""
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, metric_key in enumerate(metric_keys):
        # Find matching metrics (partial key matching)
        matching_metrics = [k for k in data.keys() if metric_key.lower() in k.lower()]

        if not matching_metrics:
            print(f"Warning: No data found for metric containing '{metric_key}'")
            continue

        for j, full_key in enumerate(matching_metrics):
            if not data[full_key]:
                continue

            steps, values = zip(*data[full_key])

            # Apply smoothing
            smoothed_values = smooth_curve(list(values), smoothing)

            color = colors[(i + j) % len(colors)]
            label = full_key.split("/")[-1] if "/" in full_key else full_key

            plt.plot(steps, smoothed_values, label=label, color=color)

    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def export_training_plots(logdir: str, output_dir: str = "exported_plots"):
    """Export all training plots from TensorBoard logs."""

    # Setup plot styling
    setup_plot_style()

    # Load TensorBoard data
    print(f"Loading TensorBoard data from: {logdir}")
    data = load_tensorboard_data(logdir)

    if not data:
        print("No data found in TensorBoard logs")
        return

    print(f"Found metrics: {list(data.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define metrics to plot
    plot_configs = [
        {
            "metrics": ["episode_return", "return", "reward"],
            "title": "Episode Returns During Training",
            "ylabel": "Episode Return",
            "filename": "episode_returns.png",
        },
        {
            "metrics": ["missiles_fired", "missile"],
            "title": "Missiles Fired Per Episode",
            "ylabel": "Missiles Fired",
            "filename": "missiles_fired.png",
        },
        {
            "metrics": ["kills", "kill"],
            "title": "Kills Per Episode",
            "ylabel": "Kills",
            "filename": "kills.png",
        },
        {
            "metrics": ["deaths", "death"],
            "title": "Deaths Per Episode",
            "ylabel": "Deaths",
            "filename": "deaths.png",
        },
        {
            "metrics": ["win_rate", "victory"],
            "title": "Win Rate",
            "ylabel": "Win Rate",
            "filename": "win_rate.png",
        },
    ]

    # Generate plots
    for config in plot_configs:
        create_metric_plot(
            data=data,
            metric_keys=config["metrics"],
            title=config["title"],
            ylabel=config["ylabel"],
            output_path=output_path / config["filename"],
        )

    print(f"All plots exported to: {output_path}")


def main():
    """Main entry point for air2air-export-plots command."""
    parser = argparse.ArgumentParser(description="Export TensorBoard metrics to high-quality plots")
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        required=True,
        help="Directory containing TensorBoard event files",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="exported_plots",
        help="Output directory for exported plots (default: exported_plots)",
    )
    parser.add_argument(
        "--smoothing",
        "-s",
        type=float,
        default=0.9,
        help="Exponential smoothing factor (0.0 = no smoothing, 0.99 = heavy smoothing)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        print(f"Error: Log directory does not exist: {args.logdir}")
        return 1

    export_training_plots(args.logdir, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
