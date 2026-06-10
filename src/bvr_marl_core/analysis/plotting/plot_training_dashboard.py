import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_tensorboard_metric import (
    _run_color_map,
    _smooth,
    labels_for_tag,
    normalize_tag,
)
from .unibw_style import set_unibw_style

# Each panel is a list of alternative canonical tags; the first one present in
# the loaded data wins. This lets the same dashboard work across the SB3-style
# logs (rollout/…, custom_metrics/…) and the RLlib-native campaign logs
# (episode_return_mean, tactical/…).
DASHBOARD_TAGS = [
    ["rollout/ep_rew_mean", "episode_return_mean"],
    ["custom_metrics/kills_mean", "tactical/team_a_kills"],
    ["custom_metrics/active_missiles_mean", "tactical/team_a_missiles_fired"],
    ["train/entropy_loss", "reward/total"],
]

# Tactical-quality overview, ordered to mirror the curriculum "Metrics to Track"
# checklist: engagement → shot quality → terminal effect → survivability → timing.
# Tags are canonical (namespace-stripped); they are resolved against whatever
# the loaded run actually emits, and panels with no data are hidden. The list
# intentionally spans both the per-episode ``env_runners/tactical/*`` metrics
# (present in all campaigns) and the per-iteration metrics that newer runs add,
# so the dashboard stays useful for old and new logs alike.
TACTICAL_DASHBOARD_TAGS = [
    # Engagement
    "tactical/team_a_missiles_fired",
    "tactical/team_a_missile_episode_rate",
    "tactical/passivity_rate",
    # Shot quality
    "tactical/valid_shot_rate",
    "tactical/invalid_shot_rate",
    "tactical/shot_efficiency",
    # Terminal effect
    "tactical/team_a_kills",
    "tactical/team_a_kill_episode_rate",
    "tactical/episodes_with_any_missile_kill_rate",
    "tactical/win_rate",
    "tactical/true_win_rate",
    "tactical/kill_ratio",
    # Survivability
    "tactical/survival_rate",
    "tactical/team_a_boundary_deaths_per_agent",
    "tactical/boundary_death_rate",
    "tactical/timeout_rate",
    # Timing
    "tactical/avg_time_to_first_shot_s",
    "tactical/avg_time_to_first_kill_s",
]


def _resolve_present_tag(available: set[str], canonical: str | list[str]) -> str | None:
    """Map a canonical tag (or list of alternatives) to the tag present in data.

    Campaign logs namespace tags under ``ray/tune/env_runners/…``; this matches
    a bare canonical tag (e.g. ``tactical/valid_shot_rate``) against the
    namespace-stripped form of whatever is present. When *canonical* is a list,
    the first alternative that resolves wins. Returns None when none are present.
    """
    candidates = [canonical] if isinstance(canonical, str) else list(canonical)
    for candidate in candidates:
        if candidate in available:
            return candidate
        for tag in available:
            if normalize_tag(tag) == candidate:
                return tag
    return None


def _plot_tag_on_ax(
    ax,
    df: pd.DataFrame,
    tag: str,
    color_map: dict[str, str],
    smoothing: float,
    show_raw: bool,
) -> bool:
    """Draw every run's curve for *tag* onto *ax*.

    Returns False (and leaves *ax* untouched) when no data exists for the tag.
    """
    tag_df = df[df["tag"] == tag]
    if tag_df.empty:
        return False

    labels = labels_for_tag(tag)
    for run, run_df in tag_df.groupby("run"):
        run_df = run_df.sort_values("step")
        color = color_map[run]

        if show_raw:
            ax.plot(run_df["step"], run_df["value"], color=color, alpha=0.18, linewidth=1.0)

        ax.plot(
            run_df["step"],
            _smooth(run_df["value"], smoothing),
            color=color,
            label=run,
        )

    ax.set_title(labels.get("title", tag))
    ax.set_xlabel(labels.get("xlabel", "Training step"))
    ax.set_ylabel(labels.get("ylabel", tag))
    ax.legend()
    return True


def plot_training_dashboard(
    df: pd.DataFrame,
    output_dir: str | Path,
    smoothing: float = 0.85,
    show_raw: bool = True,
    formats: list[str] | None = None,
) -> None:
    """Export a 2×2 overview dashboard of key RL metrics.

    Writes training_dashboard.<fmt> for each format in *formats*
    (default: svg and pdf) into output_dir.
    """
    if formats is None:
        formats = ["svg", "pdf"]

    set_unibw_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = df["run"].unique().tolist() if not df.empty else []
    color_map = _run_color_map(runs)
    available = set(df["tag"].unique().tolist()) if not df.empty else set()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, tag in zip(axes_flat, DASHBOARD_TAGS):
        present = _resolve_present_tag(available, tag)
        if present is None or not _plot_tag_on_ax(ax, df, present, color_map, smoothing, show_raw):
            ax.set_visible(False)

    fig.tight_layout()

    for fmt in formats:
        out = output_dir / f"training_dashboard.{fmt}"
        fig.savefig(out)
        print(f"Exported {out}")

    plt.close(fig)


def plot_tactical_dashboard(
    df: pd.DataFrame,
    output_dir: str | Path,
    smoothing: float = 0.85,
    show_raw: bool = True,
    formats: list[str] | None = None,
    tags: list[str] | None = None,
    ncols: int = 3,
) -> list[str]:
    """Export a grid dashboard of tactical missile-employment metrics.

    Mirrors :func:`plot_training_dashboard` but covers the curriculum
    "Metrics to Track" checklist (engagement, shot quality, terminal effect,
    survivability, and timing). Tags with no data are hidden so the dashboard
    works for any curriculum stage. Returns the list of files written.
    """
    if formats is None:
        formats = ["svg", "pdf"]

    tags = tags if tags is not None else TACTICAL_DASHBOARD_TAGS

    set_unibw_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = df["run"].unique().tolist() if not df.empty else []
    color_map = _run_color_map(runs)
    available = set(df["tag"].unique().tolist()) if not df.empty else set()

    # Only lay out panels for tags that actually have data, so the grid stays
    # compact for early stages that emit a subset of the metrics. Canonical tags
    # are resolved against the namespaced tags present in campaign logs.
    present_tags = [
        present for tag in tags if (present := _resolve_present_tag(available, tag)) is not None
    ]
    if not present_tags:
        print("No tactical metric data found — skipping tactical dashboard.")
        return []

    ncols = max(1, ncols)
    nrows = math.ceil(len(present_tags) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, tag in zip(axes_flat, present_tags):
        _plot_tag_on_ax(ax, df, tag, color_map, smoothing, show_raw)

    for ax in axes_flat[len(present_tags) :]:
        ax.set_visible(False)

    fig.suptitle("Tactical Engagement Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    written: list[str] = []
    for fmt in formats:
        out = output_dir / f"tactical_dashboard.{fmt}"
        fig.savefig(out)
        written.append(str(out))
        print(f"Exported {out}")

    plt.close(fig)
    return written
