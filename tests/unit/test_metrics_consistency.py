from __future__ import annotations

from types import SimpleNamespace

from bvr_marl_core.rl.environment.gym.gym_components.termination import TerminationChecker
from bvr_marl_core.rl.training.callbacks.metrics import EpisodeMetricsCallback


class _Episode:
    def __init__(self, info: dict):
        self._info = info

    def get_infos(self) -> dict:
        return {"A0": self._info}


class _MetricsLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def log_value(self, *, key: str, value: float, **_kwargs) -> None:
        self.values[key] = value


def test_total_mean_per_agent_is_average_not_sum() -> None:
    info = {
        "team_a_missiles_fired": 2,
        "team_b_missiles_fired": 4,
        "team_a_kills": 2,
        "team_b_kills": 0,
        "team_a_deaths": 0,
        "team_b_deaths": 2,
        "team_a_mean_kills_per_agent": 1.0,
        "team_b_mean_kills_per_agent": 0.0,
        "team_a_mean_deaths_per_agent": 0.0,
        "team_b_mean_deaths_per_agent": 1.0,
    }
    logger = _MetricsLogger()

    EpisodeMetricsCallback().on_episode_end(episode=_Episode(info), metrics_logger=logger)

    assert logger.values["total_mean_kills_per_agent"] == 0.5
    assert logger.values["total_mean_deaths_per_agent"] == 0.5


def test_episode_info_emits_team_b_fov_rate() -> None:
    checker = TerminationChecker(simulator=SimpleNamespace(active_units={}), config=None)
    checker.config = SimpleNamespace(agent_ids=["A0"], opponent_ids=["B0"])
    tracker = SimpleNamespace(
        episode_missiles_fired={"A0": 0, "B0": 0},
        episode_kills={"A0": 0, "B0": 0},
        episode_deaths={"A0": 0, "B0": 0},
        episode_valid_missile_shots={"A0": 0, "B0": 0},
        episode_vetoed_missile_shots={"A0": 0, "B0": 0},
        boundary_violators=set(),
        episode_steps_count={"A0": 2, "B0": 4},
        episode_lock_ok_count={"A0": 1, "B0": 2},
        episode_fov_ok_count={"A0": 1, "B0": 3},
        episode_first_shot_time={},
        episode_first_kill_time={},
    )

    info = checker.compute_episode_info(
        end_reasons=[],
        agents_alive=True,
        opponents_alive=True,
        current_step=1,
        state_tracker=tracker,
        agent_to_unit_id={},
    )

    assert info["team_a_fov_rate"] == 0.5
    assert info["team_b_fov_rate"] == 0.75


def test_per_agent_means_without_raw_totals_do_not_crash_episode_end() -> None:
    """Team size is back-solved from a mean only when the raw total is also present.

    An info dict carrying ``*_mean_deaths_per_agent`` but no ``*_deaths`` used to
    divide ``None`` by the mean, raising TypeError inside the RLlib episode-end
    hook and taking down the callback mid-training.
    """
    info = {
        "team_a_mean_kills_per_agent": 1.0,
        "team_b_mean_kills_per_agent": 0.5,
        "team_a_mean_deaths_per_agent": 0.5,
        "team_b_mean_deaths_per_agent": 1.0,
    }
    logger = _MetricsLogger()

    EpisodeMetricsCallback().on_episode_end(episode=_Episode(info), metrics_logger=logger)

    # Falls back to a team size of 1 per side rather than raising.
    assert logger.values["total_mean_kills_per_agent"] == 0.75
    assert logger.values["total_mean_deaths_per_agent"] == 0.75
