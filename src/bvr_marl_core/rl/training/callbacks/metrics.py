"""Episode metrics callback for logging custom statistics to TensorBoard."""

from collections.abc import Callable

from ray.rllib.callbacks.callbacks import RLlibCallback

# Info keys that gate each optional metric group. The environment writes a group
# as a unit, so the presence of its lead key implies the rest of the group.
_TEAM_GATE = "team_a_missiles_fired"
_NORMALIZED_GATE = "team_a_mean_kills_per_agent"
_SHOTS_GATE = "team_a_valid_missile_shots"
_BOUNDARY_GATE = "team_a_boundary_deaths"

# Info keys forwarded verbatim when present, rather than renamed or combined.
_PASSTHROUGH_PREFIXES = ("line_objective/", "episode/")

_TEAM_KEYS_REQUIRED = (
    "team_a_missiles_fired",
    "team_b_missiles_fired",
    "team_a_kills",
    "team_b_kills",
    "team_a_deaths",
    "team_b_deaths",
)
_TEAM_KEYS_OPTIONAL = (
    "outcome_win",
    "team_a_missile_kills",
    "team_b_missile_kills",
    "team_a_alive_count",
    "team_b_alive_count",
)
_NORMALIZED_KEYS = (
    "team_a_mean_kills_per_agent",
    "team_b_mean_kills_per_agent",
    "team_a_mean_deaths_per_agent",
    "team_b_mean_deaths_per_agent",
)
_SHOT_KEYS = (
    "team_a_valid_missile_shots",
    "team_b_valid_missile_shots",
    "team_a_vetoed_missile_shots",
    "team_b_vetoed_missile_shots",
)
_ENVELOPE_KEYS = (
    "team_a_in_envelope_missile_shots",
    "team_b_in_envelope_missile_shots",
    "team_a_out_of_envelope_missile_shots",
    "team_b_out_of_envelope_missile_shots",
)
_SENSOR_RATE_KEYS = (
    "team_a_lock_rate",
    "team_b_lock_rate",
    "team_a_fov_rate",
    "team_b_fov_rate",
)


def _metric_sink(metrics_logger, worker) -> Callable[[str, object], None] | None:
    """Resolve where metrics go, or None when this runner cannot record any.

    The new API stack logs through ``metrics_logger``; the legacy stack appends to
    ``worker.custom_metrics``.
    """
    if metrics_logger is not None:

        def log_new_stack(key: str, value: object) -> None:
            metrics_logger.log_value(key=key, value=value, reduce="mean", window=1)

        return log_new_stack

    if worker is not None and hasattr(worker, "custom_metrics"):

        def log_legacy_stack(key: str, value: object) -> None:
            worker.custom_metrics.setdefault(key, []).append(value)

        return log_legacy_stack

    return None


def _representative_agent_info(episode_info: dict) -> dict:
    """The last info dict of any one agent.

    Team statistics are mirrored into every agent's info, so one agent is enough.
    Values arrive either as a single dict or as one dict per timestep.
    """
    agent_infos = {}
    for agent_id, info_value in episode_info.items():
        if isinstance(info_value, list):
            agent_infos[agent_id] = info_value[-1] if info_value else {}
        else:
            agent_infos[agent_id] = info_value
    return next(iter(agent_infos.values())) if agent_infos else {}


def _positive_int(info: dict, *keys: str) -> int | None:
    """First key that reads as a positive integer, else None."""
    for key in keys:
        try:
            value = int(info.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _inferred_team_size(info: dict, prefix: str) -> int:
    """Agents on a side, from an explicit count or back-solved from a per-agent mean."""
    configured = _positive_int(info, f"{prefix}_team_size", "team_size", "agents_per_side")
    if configured is not None:
        return configured
    for total_key, mean_key in (
        (f"{prefix}_deaths", f"{prefix}_mean_deaths_per_agent"),
        (f"{prefix}_kills", f"{prefix}_mean_kills_per_agent"),
    ):
        total = info.get(total_key)
        mean = info.get(mean_key)
        # Both halves are required: a mean without its raw total cannot be
        # back-solved, and dividing by it would raise inside the callback and
        # take down the episode-end hook mid-training.
        if mean and total is not None:
            try:
                return max(round(total / mean), 1)
            except (TypeError, ZeroDivisionError):
                continue
    return 1


def _team_metrics(info: dict) -> dict:
    """Per-episode team totals."""
    if _TEAM_GATE not in info:
        return {}
    metrics = {key: info[key] for key in _TEAM_KEYS_REQUIRED}
    metrics.update({key: info.get(key, 0) for key in _TEAM_KEYS_OPTIONAL})
    return metrics


def _normalized_metrics(info: dict) -> dict:
    """Kills and deaths already divided by team size."""
    if _NORMALIZED_GATE not in info:
        return {}
    return {key: info[key] for key in _NORMALIZED_KEYS}


def _shot_metrics(info: dict) -> dict:
    """Missile-firing diagnostics, plus envelope and sensor-discipline rates."""
    if _SHOTS_GATE not in info:
        return {}
    metrics = {key: info[key] for key in _SHOT_KEYS}
    metrics.update({key: info[key] for key in _ENVELOPE_KEYS if key in info})
    metrics.update({key: info[key] for key in _SENSOR_RATE_KEYS if key in info})
    return metrics


def _boundary_metrics(info: dict) -> dict:
    """Deaths caused by leaving the playable area."""
    if _BOUNDARY_GATE not in info:
        return {}
    team_a = info["team_a_boundary_deaths"]
    team_b = info["team_b_boundary_deaths"]
    return {
        "team_a_boundary_deaths": team_a,
        "team_b_boundary_deaths": team_b,
        "total_boundary_deaths": team_a + team_b,
    }


def _passthrough_metrics(info: dict) -> dict:
    """Namespaced keys the environment logs directly."""
    return {key: value for key, value in info.items() if str(key).startswith(_PASSTHROUGH_PREFIXES)}


def _aggregate_metrics(info: dict) -> dict:
    """Cross-team totals and team-size-weighted averages."""
    team_a_size = _inferred_team_size(info, "team_a")
    team_b_size = _inferred_team_size(info, "team_b")
    total_agents = max(team_a_size + team_b_size, 1)
    a_share = team_a_size / total_agents
    b_share = team_b_size / total_agents

    return {
        "total_missiles_fired": info.get("team_a_missiles_fired", 0)
        + info.get("team_b_missiles_fired", 0),
        "total_kills": info.get("team_a_kills", 0) + info.get("team_b_kills", 0),
        "total_deaths": info.get("team_a_deaths", 0) + info.get("team_b_deaths", 0),
        # A clean sweep: team A's missiles accounted for every opposing agent.
        "true_kill_win_rate": int(info.get("team_a_missile_kills", 0) >= team_b_size),
        "total_mean_kills_per_agent": info.get("team_a_mean_kills_per_agent", 0) * a_share
        + info.get("team_b_mean_kills_per_agent", 0) * b_share,
        "total_mean_deaths_per_agent": info.get("team_a_mean_deaths_per_agent", 0) * a_share
        + info.get("team_b_mean_deaths_per_agent", 0) * b_share,
    }


def _episode_metrics(info: dict) -> dict:
    """Every metric derived from one agent's end-of-episode info."""
    metrics: dict = {}
    metrics.update(_team_metrics(info))
    metrics.update(_normalized_metrics(info))
    metrics.update(_shot_metrics(info))
    metrics.update(_boundary_metrics(info))
    metrics.update(_passthrough_metrics(info))
    metrics.update(_aggregate_metrics(info))
    return metrics


class EpisodeMetricsCallback(RLlibCallback):
    """
    RLlib callback to extract custom episode metrics and log them.
    This makes missiles_fired, kills, deaths, and boundary violations visible in TensorBoard.

    Note: Each metric is logged PER EPISODE. With multiple parallel workers,
    TensorBoard will show the average across all episodes completed in that iteration.
    """

    def on_episode_end(
        self,
        *,
        episode=None,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index=None,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        """Extract this episode's custom metrics from agent info and log them.

        Compatible with both the legacy and new RLlib API stacks.
        """
        sink = _metric_sink(metrics_logger, worker)
        if sink is None:
            return

        episode_info = episode.get_infos()
        if not episode_info:
            return

        info = _representative_agent_info(episode_info)
        for metric_name, metric_value in _episode_metrics(info).items():
            sink(metric_name, metric_value)
