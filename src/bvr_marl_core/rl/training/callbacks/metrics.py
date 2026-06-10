"""Episode metrics callback for logging custom statistics to TensorBoard."""

from ray.rllib.callbacks.callbacks import RLlibCallback


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
        """Called when an episode ends - extract our custom metrics from episode info.

        Compatible with both legacy and new API stacks.
        """
        # Handle legacy API stack parameters
        if worker is not None and metrics_logger is None:
            # Legacy API stack - worker has custom_metrics
            if hasattr(worker, "custom_metrics"):

                def metrics_logger_func(key, value, **kw):
                    worker.custom_metrics.setdefault(key, []).append(value)
            else:
                return
        elif metrics_logger is None:
            return
        else:

            def metrics_logger_func(key, value, **kw):
                metrics_logger.log_value(key=key, value=value, **kw)

        # Get the episode info from the last step
        # In multi-agent, we need to get info from any agent
        episode_info = episode.get_infos()

        if episode_info:
            # episode_info values can be either dicts or lists of dicts (one per timestep)
            # We need to get the last info for each agent
            agent_infos = {}
            for agent_id, info_value in episode_info.items():
                # If it's a list, get the last element; otherwise use as-is
                if isinstance(info_value, list):
                    agent_infos[agent_id] = info_value[-1] if info_value else {}
                else:
                    agent_infos[agent_id] = info_value

            # Get info from first agent (all agents have same team stats in their info)
            first_agent_info = next(iter(agent_infos.values())) if agent_infos else {}

            # Build metrics dictionary to log
            metrics_to_log = {}

            # Extract team-level metrics (these are already per-episode totals)
            if "team_a_missiles_fired" in first_agent_info:
                metrics_to_log["team_a_missiles_fired"] = first_agent_info["team_a_missiles_fired"]
                metrics_to_log["team_b_missiles_fired"] = first_agent_info["team_b_missiles_fired"]
                metrics_to_log["team_a_kills"] = first_agent_info["team_a_kills"]
                metrics_to_log["team_b_kills"] = first_agent_info["team_b_kills"]
                metrics_to_log["team_a_deaths"] = first_agent_info["team_a_deaths"]
                metrics_to_log["team_b_deaths"] = first_agent_info["team_b_deaths"]
                metrics_to_log["outcome_win"] = first_agent_info.get("outcome_win", 0)
                metrics_to_log["team_a_missile_kills"] = first_agent_info.get(
                    "team_a_missile_kills", 0
                )
                metrics_to_log["team_b_missile_kills"] = first_agent_info.get(
                    "team_b_missile_kills", 0
                )
                metrics_to_log["team_a_alive_count"] = first_agent_info.get("team_a_alive_count", 0)
                metrics_to_log["team_b_alive_count"] = first_agent_info.get("team_b_alive_count", 0)

            # Extract normalized (per-agent) metrics
            if "team_a_mean_kills_per_agent" in first_agent_info:
                metrics_to_log["team_a_mean_kills_per_agent"] = first_agent_info[
                    "team_a_mean_kills_per_agent"
                ]
                metrics_to_log["team_b_mean_kills_per_agent"] = first_agent_info[
                    "team_b_mean_kills_per_agent"
                ]
                metrics_to_log["team_a_mean_deaths_per_agent"] = first_agent_info[
                    "team_a_mean_deaths_per_agent"
                ]
                metrics_to_log["team_b_mean_deaths_per_agent"] = first_agent_info[
                    "team_b_mean_deaths_per_agent"
                ]

            # Extract diagnostic metrics for missile firing
            if "team_a_valid_missile_shots" in first_agent_info:
                metrics_to_log["team_a_valid_missile_shots"] = first_agent_info[
                    "team_a_valid_missile_shots"
                ]
                metrics_to_log["team_b_valid_missile_shots"] = first_agent_info[
                    "team_b_valid_missile_shots"
                ]
                metrics_to_log["team_a_vetoed_missile_shots"] = first_agent_info[
                    "team_a_vetoed_missile_shots"
                ]
                metrics_to_log["team_b_vetoed_missile_shots"] = first_agent_info[
                    "team_b_vetoed_missile_shots"
                ]

                # Read pre-computed sensor discipline rates from environment info
                if "team_a_lock_rate" in first_agent_info:
                    metrics_to_log["team_a_lock_rate"] = first_agent_info["team_a_lock_rate"]
                if "team_b_lock_rate" in first_agent_info:
                    metrics_to_log["team_b_lock_rate"] = first_agent_info["team_b_lock_rate"]
                if "team_a_fov_rate" in first_agent_info:
                    metrics_to_log["team_a_fov_rate"] = first_agent_info["team_a_fov_rate"]
                if "team_b_fov_rate" in first_agent_info:
                    metrics_to_log["team_b_fov_rate"] = first_agent_info["team_b_fov_rate"]

            # Extract boundary death metrics
            if "team_a_boundary_deaths" in first_agent_info:
                metrics_to_log["team_a_boundary_deaths"] = first_agent_info[
                    "team_a_boundary_deaths"
                ]
                metrics_to_log["team_b_boundary_deaths"] = first_agent_info[
                    "team_b_boundary_deaths"
                ]
                metrics_to_log["total_boundary_deaths"] = (
                    first_agent_info["team_a_boundary_deaths"]
                    + first_agent_info["team_b_boundary_deaths"]
                )

            for key, value in first_agent_info.items():
                if str(key).startswith(("line_objective/", "episode/")):
                    metrics_to_log[key] = value

            # Compute total metrics from team-level statistics
            metrics_to_log["total_missiles_fired"] = first_agent_info.get(
                "team_a_missiles_fired", 0
            ) + first_agent_info.get("team_b_missiles_fired", 0)
            metrics_to_log["total_kills"] = first_agent_info.get(
                "team_a_kills", 0
            ) + first_agent_info.get("team_b_kills", 0)
            metrics_to_log["total_deaths"] = first_agent_info.get(
                "team_a_deaths", 0
            ) + first_agent_info.get("team_b_deaths", 0)

            # Compute total normalized metrics (average across both teams)
            def positive_int_info(*keys: str) -> int | None:
                for key in keys:
                    try:
                        value = int(first_agent_info.get(key))
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        return value
                return None

            def inferred_team_size(prefix: str) -> int:
                configured = positive_int_info(
                    f"{prefix}_team_size", "team_size", "agents_per_side"
                )
                if configured is not None:
                    return configured
                for total_key, mean_key in (
                    (f"{prefix}_deaths", f"{prefix}_mean_deaths_per_agent"),
                    (f"{prefix}_kills", f"{prefix}_mean_kills_per_agent"),
                ):
                    total = first_agent_info.get(total_key)
                    mean = first_agent_info.get(mean_key)
                    if mean:
                        return max(round(total / mean), 1)
                return 1

            team_a_size = inferred_team_size("team_a")
            team_b_size = inferred_team_size("team_b")
            metrics_to_log["true_kill_win_rate"] = int(
                first_agent_info.get("team_a_missile_kills", 0) >= team_b_size
            )
            total_agents = max(team_a_size + team_b_size, 1)
            metrics_to_log["total_mean_kills_per_agent"] = first_agent_info.get(
                "team_a_mean_kills_per_agent", 0
            ) * (team_a_size / total_agents) + first_agent_info.get(
                "team_b_mean_kills_per_agent", 0
            ) * (team_b_size / total_agents)
            metrics_to_log["total_mean_deaths_per_agent"] = first_agent_info.get(
                "team_a_mean_deaths_per_agent", 0
            ) * (team_a_size / total_agents) + first_agent_info.get(
                "team_b_mean_deaths_per_agent", 0
            ) * (team_b_size / total_agents)

            # Log all metrics
            for metric_name, metric_value in metrics_to_log.items():
                if metrics_logger is not None:
                    # New API stack
                    metrics_logger.log_value(
                        key=metric_name,
                        value=metric_value,
                        reduce="mean",
                        window=1,
                    )
                else:
                    # Legacy API stack - use custom_metrics
                    metrics_logger_func(metric_name, metric_value)
