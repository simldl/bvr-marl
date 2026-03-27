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
                # For legacy API, we'll use worker.custom_metrics instead of metrics_logger
                def metrics_logger_func(key, value, **kw):
                    worker.custom_metrics.setdefault(key, []).append(value)
            else:
                # No metrics logging available
                return
        elif metrics_logger is None:
            # No metrics logging available in either API
            return
        else:
            # New API stack - use metrics_logger directly
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

                # Compute lock success rates (avoid division by zero)
                team_a_steps = first_agent_info.get("team_a_steps", 1)
                team_b_steps = first_agent_info.get("team_b_steps", 1)
                metrics_to_log["team_a_lock_ok_rate"] = first_agent_info[
                    "team_a_lock_ok_count"
                ] / max(team_a_steps, 1)
                metrics_to_log["team_b_lock_ok_rate"] = first_agent_info[
                    "team_b_lock_ok_count"
                ] / max(team_b_steps, 1)
                metrics_to_log["team_a_fov_ok_rate"] = first_agent_info[
                    "team_a_fov_ok_count"
                ] / max(team_a_steps, 1)
                metrics_to_log["team_b_fov_ok_rate"] = first_agent_info[
                    "team_b_fov_ok_count"
                ] / max(team_b_steps, 1)

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
            metrics_to_log["total_mean_kills_per_agent"] = first_agent_info.get(
                "team_a_mean_kills_per_agent", 0
            ) + first_agent_info.get("team_b_mean_kills_per_agent", 0)
            metrics_to_log["total_mean_deaths_per_agent"] = first_agent_info.get(
                "team_a_mean_deaths_per_agent", 0
            ) + first_agent_info.get("team_b_mean_deaths_per_agent", 0)

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
