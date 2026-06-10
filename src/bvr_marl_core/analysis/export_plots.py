"""
Export TensorBoard metrics to high-quality multi-run plots.

Reads TensorBoard event files and creates publication-quality plots for
mean return, kills, deaths, missiles fired, and win rate.  Supports
multiple runs on the same graph so training runs can be compared visually.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from bvr_marl_core.analysis.plotting import (
    TACTICAL_DASHBOARD_TAGS,
    TAG_LABELS,
    load_tensorboard_scalars,
    plot_tactical_dashboard,
    plot_training_dashboard,
)
from bvr_marl_core.analysis.plotting.plot_tensorboard_metric import normalize_tag

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard is not installed!")
    print("Install with: pip install tensorboard")
    exit(1)


# Priority-ordered search patterns per metric.  The first exact match wins;
# if none are exact, the first partial (case-insensitive) match wins.
_METRIC_PATTERNS: dict[str, list[str]] = {
    "mean_return": [
        "env_runners/episode_return_mean",
        "episode_return_mean",
        "episode_reward_mean",
        "episode_return",
        "return",
        "reward",
    ],
    "kills": ["total_kills", "team_a_kills", "kills", "kill"],
    "deaths": ["total_deaths", "team_a_deaths", "deaths", "death"],
    "missiles_fired": [
        "total_missiles_fired",
        "team_a_missiles_fired",
        "missiles_fired",
        "missile",
    ],
    "win_rate": [
        "tactical/true_win_rate",
        "true_kill_win_rate",
        "true_win_rate",
        "kill_win_rate",
        "tactical/win_rate",
        "outcome_win",
        "win_rate",
        "victory",
    ],
}

_STRICT_WIN_RATE_PATTERNS = {
    "tactical/true_win_rate",
    "true_kill_win_rate",
    "true_win_rate",
    "kill_win_rate",
}

_PLOT_CONFIGS = [
    {
        "key": "mean_return",
        "title": "Mean Episode Return",
        "ylabel": "Mean Return",
        "filename": "mean_return.png",
    },
    {
        "key": "kills",
        "title": "Total Kills Per Episode",
        "ylabel": "Kills",
        "filename": "kills.png",
    },
    {
        "key": "deaths",
        "title": "Total Deaths Per Episode",
        "ylabel": "Deaths",
        "filename": "deaths.png",
    },
    {
        "key": "missiles_fired",
        "title": "Total Missiles Fired Per Episode",
        "ylabel": "Missiles Fired",
        "filename": "missiles_fired.png",
    },
    {
        "key": "win_rate",
        "title": "Training True Kill Win Rate",
        "ylabel": "True Kill Win Rate",
        "fallback_title": "Training Kill-Advantage Rate (Legacy Proxy)",
        "fallback_ylabel": "Kill-Advantage Rate",
        "mixed_title": "Training Win Rate (Mixed True/Proxy)",
        "strict_patterns": _STRICT_WIN_RATE_PATTERNS,
        "filename": "win_rate.png",
    },
]


def _safe_filename(tag: str) -> str:
    """Convert a metric tag to a safe PNG filename component."""
    for ch in r"/\: *?\"<>|":
        tag = tag.replace(ch, "_")
    return tag


def _tactical_plot_configs() -> list[dict]:
    """Build multi-run comparison configs for the shared tactical metric catalog.

    Titles/ylabels come from the shared ``TAG_LABELS`` so the metric catalog is
    defined in exactly one place. Pattern matching relies on the partial,
    case-insensitive match in ``_find_metric_key`` to resolve the namespaced
    ``ray/tune/env_runners/<tag>`` form that campaign logs emit.
    """
    configs: list[dict] = []
    for tag in TACTICAL_DASHBOARD_TAGS:
        labels = TAG_LABELS.get(tag, {}) or TAG_LABELS.get(normalize_tag(tag), {})
        configs.append(
            {
                "key": tag,
                "patterns": [tag],
                "title": labels.get("title", tag),
                "ylabel": labels.get("ylabel", tag),
                "filename": f"tactical__{_safe_filename(tag.split('/')[-1])}.png",
            }
        )
    return configs


_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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
    if "seaborn-v0_8-darkgrid" in plt.style.available:
        mplstyle.use("seaborn-v0_8-darkgrid")
    else:
        plt.style.use("default")


def load_tensorboard_data(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """
    Load all scalar data from TensorBoard event files under logdir.

    Returns a dict mapping metric tag -> sorted list of (step, value) tuples.
    """
    event_files = []
    for root, _dirs, files in os.walk(logdir):
        for fname in files:
            if fname.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, fname))

    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {logdir}")

    print(f"  Found {len(event_files)} event file(s)")

    data: dict[str, list[tuple[int, float]]] = {}
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={event_accumulator.SCALARS: 0},
            )
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                events = ea.Scalars(tag)
                data.setdefault(tag, []).extend((se.step, se.value) for se in events)
        except Exception as e:
            print(f"  Warning: could not process {event_file}: {e}")

    for tag in data:
        data[tag] = sorted(data[tag], key=lambda x: x[0])

    return data


def smooth_curve(values: list[float], smoothing: float = 0.9) -> list[float]:
    """Apply exponential moving-average smoothing."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(smoothed[-1] * smoothing + v * (1 - smoothing))
    return smoothed


def _expand_run_dir(run_name: str, logdir: str) -> dict[str, str]:
    """
    Return a {display_name: path} mapping for a given run directory.

    If the directory contains event files directly, returns {run_name: logdir}.
    If it has no direct event files but its immediate subdirectories do, returns
    one entry per subdirectory so each sub-run becomes its own plot line.
    This handles Ray Tune experiment directories (each trial is a subdir) and
    manually organized run collections.
    """
    root = Path(logdir)

    # Event files directly at root → single run, keep the supplied name
    if list(root.glob("events.out.tfevents.*")):
        return {run_name: logdir}

    # Scan immediate children for sub-runs
    sub_runs: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and list(child.glob("**/events.out.tfevents.*")):
            sub_runs[child.name] = str(child)

    if not sub_runs:
        # No sub-runs found at all — pass through and let load_tensorboard_data report the error
        return {run_name: logdir}

    if len(sub_runs) == 1:
        # One sub-run: use the parent's name so the legend stays clean
        return {run_name: next(iter(sub_runs.values()))}

    # Multiple sub-runs: label each with its subdirectory name
    return sub_runs


def _find_metric_key(data: dict[str, list], patterns: list[str]) -> str | None:
    """Return the highest-priority data key that matches one of the patterns."""
    # Exact match first (in pattern priority order)
    for pattern in patterns:
        if pattern in data:
            return pattern
    # Partial case-insensitive match (in pattern priority order)
    for pattern in patterns:
        for key in data:
            if pattern.lower() in key.lower():
                return key
    return None


def _matches_any_pattern(key: str, patterns: set[str]) -> bool:
    key_lower = key.lower()
    return any(pattern.lower() in key_lower for pattern in patterns)


_MAX_LABEL_LEN = 30


def _truncate(name: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Truncate a run name to max_len chars, appending '…' if cut."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def create_multi_run_plot(
    run_data: dict[str, dict[str, list[tuple[int, float]]]],
    patterns: list[str],
    title: str,
    ylabel: str,
    output_path: str,
    smoothing: float = 0.9,
    fallback_title: str | None = None,
    fallback_ylabel: str | None = None,
    mixed_title: str | None = None,
    strict_patterns: set[str] | None = None,
):
    """Create and save a plot with one line per run."""
    plt.figure(figsize=(12, 8))
    plotted = False
    selected_keys: list[str] = []

    for run_idx, (run_name, data) in enumerate(run_data.items()):
        key = _find_metric_key(data, patterns)
        if key is None:
            print(f"  [{run_name}] no data found for pattern '{patterns[0]}'")
            continue
        if not data[key]:
            continue

        steps, values = zip(*data[key])
        smoothed = smooth_curve(list(values), smoothing)
        plt.plot(
            steps,
            smoothed,
            label=_truncate(run_name),
            color=_COLORS[run_idx % len(_COLORS)],
        )
        plotted = True
        selected_keys.append(key)

    if not plotted:
        print(f"  Skipping '{title}' — no matching data in any run")
        plt.close()
        return

    actual_title = title
    actual_ylabel = ylabel
    if strict_patterns is not None:
        strict_count = sum(_matches_any_pattern(key, strict_patterns) for key in selected_keys)
        if strict_count == 0:
            actual_title = fallback_title or title
            actual_ylabel = fallback_ylabel or ylabel
            print(
                "  Win-rate plot used the legacy kill-advantage proxy; "
                "strict true kill-win data was not found."
            )
        elif strict_count < len(selected_keys):
            actual_title = mixed_title or title
            print("  Warning: win-rate plot mixes strict true kill-win and legacy proxy metrics.")

    plt.title(actual_title)
    plt.xlabel("Training Step")
    plt.ylabel(actual_ylabel)
    plt.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=14,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.5,
    )
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def export_training_plots(
    run_dirs: dict[str, str],
    output_dir: str = "exported_plots",
    smoothing: float = 0.9,
):
    """Load TensorBoard data for each run and export all configured plots."""
    setup_plot_style()

    # Expand each entry: a dir with no direct event files but sub-run dirs
    # becomes multiple entries (one per sub-run) rather than a merged blob.
    expanded: dict[str, str] = {}
    for run_name, logdir in run_dirs.items():
        expanded.update(_expand_run_dir(run_name, logdir))

    run_data: dict[str, dict] = {}
    for run_name, logdir in expanded.items():
        print(f"\nLoading '{run_name}' from {logdir}")
        try:
            data = load_tensorboard_data(logdir)
            if data:
                run_data[run_name] = data
            else:
                print("  (no scalar data found)")
        except ValueError as e:
            print(f"  Warning: {e}")

    if not run_data:
        print("No data loaded — nothing to plot")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Overview comparison plots followed by the shared tactical metric catalog.
    all_configs = _PLOT_CONFIGS + _tactical_plot_configs()
    print(f"\nExporting {len(all_configs)} comparison plots to {out}")
    for cfg in all_configs:
        create_multi_run_plot(
            run_data=run_data,
            patterns=cfg.get("patterns") or _METRIC_PATTERNS[cfg["key"]],
            title=cfg["title"],
            ylabel=cfg["ylabel"],
            output_path=str(out / cfg["filename"]),
            smoothing=smoothing,
            fallback_title=cfg.get("fallback_title"),
            fallback_ylabel=cfg.get("fallback_ylabel"),
            mixed_title=cfg.get("mixed_title"),
            strict_patterns=cfg.get("strict_patterns"),
        )

    _export_dashboards(expanded, out, smoothing)

    print(f"\nDone — all plots in: {out}")


def _export_dashboards(expanded: dict[str, str], out: Path, smoothing: float) -> None:
    """Export the shared training and tactical dashboards for *expanded* runs.

    Reuses the same dashboard renderers as the extension plot exporter via the
    shared ``bvr_marl_core.analysis.plotting`` package, so there is one
    implementation of the metric/label catalog and the dashboards.
    """
    import pandas as pd

    frames = []
    for run_name, logdir in expanded.items():
        df = load_tensorboard_scalars(logdir)
        if df.empty:
            continue
        df = df.copy()
        df["run"] = run_name  # one legend entry per run, ignore internal sub-paths
        frames.append(df)

    if not frames:
        return

    df_all = pd.concat(frames, ignore_index=True)
    print("\nExporting training + tactical dashboards")
    try:
        plot_training_dashboard(
            df=df_all, output_dir=out, smoothing=smoothing, show_raw=False, formats=["png", "pdf"]
        )
        plot_tactical_dashboard(
            df=df_all, output_dir=out, smoothing=smoothing, show_raw=False, formats=["png", "pdf"]
        )
    except Exception as exc:  # noqa: BLE001 — dashboards are best-effort extras
        print(f"  Warning: dashboard export failed: {exc}")


def main():
    """Main entry point for the bvr-export-plots command."""
    parser = argparse.ArgumentParser(description="Export TensorBoard metrics to high-quality plots")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--logdir",
        "-l",
        type=str,
        help="Single log directory (backward-compatible; run label = directory name)",
    )
    src.add_argument(
        "--runs",
        "-r",
        nargs="+",
        metavar="RUN",
        help="One or more run names to plot together (resolved against --models-dir)",
    )
    src.add_argument(
        "--run-paths",
        nargs="+",
        metavar="NAME:PATH",
        help="Explicit name:path pairs, e.g. --run-paths run1:/abs/path/run1 run2:/abs/path/run2",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Root models directory used when --runs is specified (default: models)",
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
        help="Exponential smoothing factor: 0=none, 0.99=heavy (default: 0.9)",
    )

    args = parser.parse_args()

    if args.logdir:
        if not os.path.exists(args.logdir):
            print(f"Error: directory does not exist: {args.logdir}")
            return 1
        run_dirs = {Path(args.logdir).name: args.logdir}
    elif args.run_paths:
        run_dirs = {}
        for entry in args.run_paths:
            if ":" not in entry:
                print(f"Error: --run-paths entries must be NAME:PATH, got: {entry!r}")
                return 1
            name, path = entry.split(":", 1)
            if not os.path.exists(path):
                print(f"Warning: path not found for '{name}': {path}")
            else:
                run_dirs[name] = path
        if not run_dirs:
            print("Error: none of the specified run paths were found")
            return 1
    else:
        models_dir = Path(args.models_dir)
        run_dirs = {}
        for run_name in args.runs:
            run_path = models_dir / run_name
            if not run_path.exists():
                print(f"Warning: run not found — {run_path}")
            else:
                run_dirs[run_name] = str(run_path)
        if not run_dirs:
            print("Error: none of the specified runs were found")
            return 1

    export_training_plots(run_dirs, args.output_dir, args.smoothing)
    return 0


if __name__ == "__main__":
    exit(main())
