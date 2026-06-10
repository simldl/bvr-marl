from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .unibw_style import RUN_COLOR_ORDER, set_unibw_style

TAG_LABELS: dict[str, dict[str, str]] = {
    "rollout/ep_rew_mean": {
        "title": "Training Performance",
        "ylabel": "Mean episode return",
        "xlabel": "Training step",
    },
    "train/value_loss": {
        "title": "Value-Function Loss",
        "ylabel": "Loss",
        "xlabel": "Training step",
    },
    "train/policy_gradient_loss": {
        "title": "Policy-Gradient Loss",
        "ylabel": "Loss",
        "xlabel": "Training step",
    },
    "train/entropy_loss": {
        "title": "Policy Entropy",
        "ylabel": "Entropy loss",
        "xlabel": "Training step",
    },
    "train/approx_kl": {
        "title": "Approximate KL Divergence",
        "ylabel": "Approx. KL",
        "xlabel": "Training step",
    },
    "custom_metrics/kills_mean": {
        "title": "Kills per Episode",
        "ylabel": "Mean kills",
        "xlabel": "Training step",
    },
    "custom_metrics/active_missiles_mean": {
        "title": "Active Missiles",
        "ylabel": "Mean active missiles",
        "xlabel": "Training step",
    },
    "custom_metrics/tipi_mean": {
        "title": "Theoretical Instantaneous Probability of Intercept",
        "ylabel": "Mean TIPI",
        "xlabel": "Training step",
    },
    "custom_metrics/survival_time_mean": {
        "title": "Survival Time",
        "ylabel": "Mean survival time",
        "xlabel": "Training step",
    },
    "tactical/team_a_boundary_deaths_per_agent": {
        "title": "Team A Boundary Deaths per Agent",
        "ylabel": "Deaths per agent",
        "xlabel": "Training step",
    },
    "tactical/team_a_real_deaths_per_agent": {
        "title": "Team A Shot-Down Deaths per Agent",
        "ylabel": "Deaths per agent",
        "xlabel": "Training step",
    },
    "tactical/team_b_boundary_deaths_per_agent": {
        "title": "Team B Boundary Deaths per Agent",
        "ylabel": "Deaths per agent",
        "xlabel": "Training step",
    },
    "tactical/team_b_real_deaths_per_agent": {
        "title": "Team B Shot-Down Deaths per Agent",
        "ylabel": "Deaths per agent",
        "xlabel": "Training step",
    },
    # ── Engagement / missile-employment quality ───────────────────────────────
    "tactical/team_a_missiles_fired": {
        "title": "Team A Missiles Fired",
        "ylabel": "Mean missiles per episode",
        "xlabel": "Training step",
    },
    "tactical/team_b_missiles_fired": {
        "title": "Team B Missiles Fired",
        "ylabel": "Mean missiles per episode",
        "xlabel": "Training step",
    },
    "tactical/team_a_kills": {
        "title": "Team A Kills",
        "ylabel": "Mean kills per episode",
        "xlabel": "Training step",
    },
    "tactical/team_b_kills": {
        "title": "Team B Kills",
        "ylabel": "Mean kills per episode",
        "xlabel": "Training step",
    },
    "tactical/valid_shot_rate": {
        "title": "Valid Shot Rate",
        "ylabel": "Valid / attempted shots",
        "xlabel": "Training step",
    },
    "tactical/invalid_shot_rate": {
        "title": "Invalid Shot Rate",
        "ylabel": "Invalid / attempted shots",
        "xlabel": "Training step",
    },
    "tactical/shot_efficiency": {
        "title": "Shot Efficiency",
        "ylabel": "Kills per missile fired",
        "xlabel": "Training step",
    },
    "tactical/passivity_rate": {
        "title": "Passivity Rate",
        "ylabel": "Fraction of episodes never firing",
        "xlabel": "Training step",
    },
    "tactical/lock_rate": {
        "title": "Lock Rate",
        "ylabel": "Mean lock rate",
        "xlabel": "Training step",
    },
    "tactical/fov_rate": {
        "title": "Field-of-View Rate",
        "ylabel": "Mean FOV rate",
        "xlabel": "Training step",
    },
    # ── Episode-level outcome rates ───────────────────────────────────────────
    "tactical/win_rate": {
        "title": "Win Rate (Kill Advantage)",
        "ylabel": "Fraction of episodes won",
        "xlabel": "Training step",
    },
    "tactical/true_win_rate": {
        "title": "True Win Rate (Full Missile Kill)",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    "tactical/kill_ratio": {
        "title": "Kill Ratio",
        "ylabel": "Team A kills / Team B kills",
        "xlabel": "Training step",
    },
    "tactical/survival_rate": {
        "title": "Team A Survival Rate",
        "ylabel": "Fraction surviving",
        "xlabel": "Training step",
    },
    "tactical/episodes_with_any_missile_kill_rate": {
        "title": "Episodes with Any Missile Kill",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    "tactical/team_a_kill_episode_rate": {
        "title": "Team A Kill-Episode Rate",
        "ylabel": "Fraction of episodes with ≥1 kill",
        "xlabel": "Training step",
    },
    "tactical/team_b_kill_episode_rate": {
        "title": "Team B Kill-Episode Rate",
        "ylabel": "Fraction of episodes with ≥1 kill",
        "xlabel": "Training step",
    },
    "tactical/team_a_missile_episode_rate": {
        "title": "Team A Missile-Episode Rate",
        "ylabel": "Fraction of episodes firing ≥1 missile",
        "xlabel": "Training step",
    },
    "tactical/team_b_missile_episode_rate": {
        "title": "Team B Missile-Episode Rate",
        "ylabel": "Fraction of episodes firing ≥1 missile",
        "xlabel": "Training step",
    },
    "tactical/team_a_boundary_death_episode_rate": {
        "title": "Team A Boundary-Death Episode Rate",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    "tactical/team_b_boundary_death_episode_rate": {
        "title": "Team B Boundary-Death Episode Rate",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    "tactical/boundary_death_rate": {
        "title": "Boundary-Death Rate",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    "tactical/timeout_rate": {
        "title": "Timeout Rate",
        "ylabel": "Fraction of episodes timed out",
        "xlabel": "Training step",
    },
    "tactical/line_objective_success_rate": {
        "title": "Line-Objective Success Rate",
        "ylabel": "Fraction of episodes",
        "xlabel": "Training step",
    },
    # ── Engagement timing ─────────────────────────────────────────────────────
    "tactical/avg_engagement_duration_s": {
        "title": "Average Engagement Duration",
        "ylabel": "Seconds",
        "xlabel": "Training step",
    },
    "tactical/avg_time_to_first_shot_s": {
        "title": "Average Time to First Shot",
        "ylabel": "Seconds",
        "xlabel": "Training step",
    },
    "tactical/avg_time_to_first_kill_s": {
        "title": "Average Time to First Kill",
        "ylabel": "Seconds",
        "xlabel": "Training step",
    },
    "tactical/team_a_alive_count": {
        "title": "Team A Alive at Episode End",
        "ylabel": "Mean aircraft alive",
        "xlabel": "Training step",
    },
    "tactical/team_b_alive_count": {
        "title": "Team B Alive at Episode End",
        "ylabel": "Mean aircraft alive",
        "xlabel": "Training step",
    },
    # ── RLlib-native training curves (ray/tune/env_runners + learners) ────────
    "episode_return_mean": {
        "title": "Training Performance",
        "ylabel": "Mean episode return",
        "xlabel": "Training step",
    },
    "episode_return_max": {
        "title": "Episode Return (Max)",
        "ylabel": "Max episode return",
        "xlabel": "Training step",
    },
    "episode_return_min": {
        "title": "Episode Return (Min)",
        "ylabel": "Min episode return",
        "xlabel": "Training step",
    },
    "episode_len_mean": {
        "title": "Episode Length",
        "ylabel": "Mean steps per episode",
        "xlabel": "Training step",
    },
    # ── Reward composition (ray/tune/env_runners/reward/*) ────────────────────
    # These per-component mean contributions are the primary diagnostic for
    # dense-reward exploitation: watch terminal terms (kills) vs dense posture
    # terms (tracking, heading_alignment) as a fraction of reward/total.
    "reward/total": {
        "title": "Reward — Total",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/kills": {
        "title": "Reward — Kills (terminal)",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/destruction": {
        "title": "Reward — Destruction (terminal)",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/passivity": {
        "title": "Reward — Passivity Penalty",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/tracking": {
        "title": "Reward — Tracking (dense)",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/heading_alignment": {
        "title": "Reward — Heading Alignment (dense)",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/survival": {
        "title": "Reward — Survival",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/evasion": {
        "title": "Reward — Evasion",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/boundary_violation": {
        "title": "Reward — Boundary Violation Penalty",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/boundary_approach": {
        "title": "Reward — Boundary Approach Penalty",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/center_zone_control": {
        "title": "Reward — Center-Zone Control",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/lift_balance": {
        "title": "Reward — Lift Balance",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    "reward/orbit_penalty": {
        "title": "Reward — Orbit Penalty",
        "ylabel": "Mean reward",
        "xlabel": "Training step",
    },
    # ── Curriculum progression ────────────────────────────────────────────────
    "curriculum/stage_win_rate": {
        "title": "Curriculum Stage Win Rate",
        "ylabel": "Win rate",
        "xlabel": "Training step",
    },
    "curriculum/stage_stability": {
        "title": "Curriculum Stage Stability",
        "ylabel": "Stability score",
        "xlabel": "Training step",
    },
    "curriculum/promotion_score": {
        "title": "Curriculum Promotion Score",
        "ylabel": "Promotion score",
        "xlabel": "Training step",
    },
    "curriculum/regression_score": {
        "title": "Curriculum Regression Score",
        "ylabel": "Regression score",
        "xlabel": "Training step",
    },
}


# RLlib writes scalars under namespaced TensorBoard tags such as
# ``ray/tune/env_runners/tactical/valid_shot_rate``. The TAG_LABELS keys above
# use the bare ``tactical/...`` form, so we strip these known prefixes before
# looking a tag up. Order matters: the most specific prefix is tried first.
RLLIB_TAG_PREFIXES = ("ray/tune/env_runners/", "ray/tune/")


def normalize_tag(tag: str) -> str:
    """Strip RLlib TensorBoard namespace prefixes so tags match TAG_LABELS keys."""
    for prefix in RLLIB_TAG_PREFIXES:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return tag


def labels_for_tag(tag: str) -> dict[str, str]:
    """Return the label dict for *tag*, tolerant of RLlib namespace prefixes."""
    return TAG_LABELS.get(tag) or TAG_LABELS.get(normalize_tag(tag), {})


def has_known_label(tag: str) -> bool:
    """True when *tag* (raw or namespace-stripped) has a human-readable label."""
    return tag in TAG_LABELS or normalize_tag(tag) in TAG_LABELS


def _smooth(values: pd.Series, smoothing: float) -> pd.Series:
    if smoothing <= 0:
        return values
    return values.ewm(alpha=1.0 - smoothing, adjust=False).mean()


def _run_color_map(runs: list[str]) -> dict[str, str]:
    """Assign colors deterministically by sorting run names."""
    return {run: RUN_COLOR_ORDER[i % len(RUN_COLOR_ORDER)] for i, run in enumerate(sorted(runs))}


def plot_metric(
    df: pd.DataFrame,
    tag: str,
    output_path: str | Path,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    smoothing: float = 0.85,
    show_raw: bool = True,
) -> None:
    """Export a single TensorBoard metric as a presentation-ready figure.

    Saves to output_path exactly as given (extension determines format).
    """
    set_unibw_style()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_df = df[df["tag"] == tag].copy()
    if metric_df.empty:
        raise ValueError(f"No TensorBoard scalar found for tag: {tag!r}")

    labels = labels_for_tag(tag)
    title = title or labels.get("title", tag)
    xlabel = xlabel or labels.get("xlabel", "Training step")
    ylabel = ylabel or labels.get("ylabel", tag)

    runs = metric_df["run"].unique().tolist()
    color_map = _run_color_map(runs)

    fig, ax = plt.subplots()

    for run, run_df in metric_df.groupby("run"):
        run_df = run_df.sort_values("step")
        color = color_map[run]

        if show_raw:
            ax.plot(
                run_df["step"],
                run_df["value"],
                color=color,
                alpha=0.18,
                linewidth=1.0,
            )

        ax.plot(
            run_df["step"],
            _smooth(run_df["value"], smoothing),
            color=color,
            label=run,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
