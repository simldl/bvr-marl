"""
Episode metrics collection for BVR MARL study.

Tracks key metrics for statistical analysis of training and evaluation episodes:
- Time-based metrics (detection, engagement, survival)
- Combat effectiveness metrics (shots fired, kills, accuracy)
- Tactical metrics (boundary violations, formation)
- Support metrics (AWACS support time)

All metrics are collected per-episode and can be exported to CSV for analysis.
"""

import csv
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """
    Metrics for a single episode.

    All times are in simulation seconds.
    All counts are integers.
    All ratios are floats in [0, 1].
    """

    # Episode identification
    episode_id: int = 0
    scenario_name: str = ""
    geometry_regime: str = ""
    seed: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    agent_aircraft_type: str = ""
    opponent_aircraft_type: str = ""
    agent_missile_type: str = ""
    opponent_missile_type: str = ""
    num_agents: int = 0
    num_opponents: int = 0
    datalink_mode: str = "full"

    # Episode outcome
    winner: str = ""  # "agent", "opponent", "draw", "timeout"
    termination_reason: str = ""
    episode_duration_s: float = 0.0
    episode_steps: int = 0

    # Time-based metrics (per team, first event)
    agent_time_to_first_detection_s: float = -1.0  # -1 = never detected
    opponent_time_to_first_detection_s: float = -1.0
    agent_time_to_first_shot_s: float = -1.0  # -1 = never fired
    opponent_time_to_first_shot_s: float = -1.0
    agent_time_to_first_kill_s: float = -1.0  # -1 = never killed
    opponent_time_to_first_kill_s: float = -1.0

    # Combat metrics (per team totals)
    agent_missiles_fired: int = 0
    opponent_missiles_fired: int = 0
    agent_missiles_hit: int = 0
    opponent_missiles_hit: int = 0
    agent_kills: int = 0
    opponent_kills: int = 0

    # Derived combat metrics
    agent_shots_per_kill: float = 0.0
    opponent_shots_per_kill: float = 0.0
    agent_missile_accuracy: float = 0.0  # hits / fired
    opponent_missile_accuracy: float = 0.0

    # Survival metrics
    agent_survivors: int = 0
    opponent_survivors: int = 0
    agent_total_survival_time_s: float = 0.0  # Sum of all agent survival times
    opponent_total_survival_time_s: float = 0.0
    agent_last_man_standing_time_s: float = 0.0  # Time when last agent died
    opponent_last_man_standing_time_s: float = 0.0

    # Tactical metrics
    agent_boundary_violations: int = 0
    opponent_boundary_violations: int = 0
    agent_countermeasures_used: int = 0
    opponent_countermeasures_used: int = 0

    # Support metrics (for AWACS scenarios)
    agent_awacs_support_time_s: float = 0.0  # Time agents had AWACS radar coverage
    opponent_awacs_support_time_s: float = 0.0
    agent_awacs_detections: int = 0  # Detections contributed by AWACS
    opponent_awacs_detections: int = 0

    # Engagement geometry (initial conditions)
    initial_range_m: float = 0.0
    initial_aspect: str = ""
    initial_altitude_diff_m: float = 0.0

    # Cumulative reward (if applicable)
    agent_total_reward: float = 0.0
    opponent_total_reward: float = 0.0

    def compute_derived_metrics(self):
        """Compute derived metrics from raw counts."""
        # Shots per kill
        if self.agent_kills > 0:
            self.agent_shots_per_kill = self.agent_missiles_fired / self.agent_kills
        if self.opponent_kills > 0:
            self.opponent_shots_per_kill = self.opponent_missiles_fired / self.opponent_kills

        # Missile accuracy
        if self.agent_missiles_fired > 0:
            self.agent_missile_accuracy = self.agent_missiles_hit / self.agent_missiles_fired
        if self.opponent_missiles_fired > 0:
            self.opponent_missile_accuracy = (
                self.opponent_missiles_hit / self.opponent_missiles_fired
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class MetricsCollector:
    """
    Collects metrics across multiple episodes.

    Provides aggregation, filtering, and export capabilities.
    """

    def __init__(self, output_dir: str = "study_results"):
        """
        Initialize the metrics collector.

        Args:
            output_dir: Directory for CSV output files.
        """
        self.output_dir = output_dir
        self.episodes: list[EpisodeMetrics] = []
        self._current_episode: EpisodeMetrics | None = None
        self._episode_counter = 0

        # Runtime tracking state
        self._first_detection_recorded = {"agent": False, "opponent": False}
        self._first_shot_recorded = {"agent": False, "opponent": False}
        self._first_kill_recorded = {"agent": False, "opponent": False}
        self._unit_spawn_times: dict[int, float] = {}
        self._unit_death_times: dict[int, float] = {}

    def start_episode(
        self,
        scenario_name: str = "",
        geometry_regime: str = "",
        seed: int | None = None,
        agent_aircraft_type: str = "",
        opponent_aircraft_type: str = "",
        agent_missile_type: str = "",
        opponent_missile_type: str = "",
        num_agents: int = 2,
        num_opponents: int = 2,
        datalink_mode: str = "full",
    ) -> EpisodeMetrics:
        """
        Start tracking a new episode.

        Returns:
            The EpisodeMetrics object for the new episode.
        """
        self._episode_counter += 1
        self._current_episode = EpisodeMetrics(
            episode_id=self._episode_counter,
            scenario_name=scenario_name,
            geometry_regime=geometry_regime,
            seed=seed,
            agent_aircraft_type=agent_aircraft_type,
            opponent_aircraft_type=opponent_aircraft_type,
            agent_missile_type=agent_missile_type,
            opponent_missile_type=opponent_missile_type,
            num_agents=num_agents,
            num_opponents=num_opponents,
            datalink_mode=datalink_mode,
        )

        # Reset tracking state
        self._first_detection_recorded = {"agent": False, "opponent": False}
        self._first_shot_recorded = {"agent": False, "opponent": False}
        self._first_kill_recorded = {"agent": False, "opponent": False}
        self._unit_spawn_times.clear()
        self._unit_death_times.clear()

        return self._current_episode

    def record_detection(self, team: str, sim_time_s: float):
        """Record first detection time for a team."""
        if self._current_episode is None:
            return

        if team == "agent" and not self._first_detection_recorded["agent"]:
            self._current_episode.agent_time_to_first_detection_s = sim_time_s
            self._first_detection_recorded["agent"] = True
        elif team == "opponent" and not self._first_detection_recorded["opponent"]:
            self._current_episode.opponent_time_to_first_detection_s = sim_time_s
            self._first_detection_recorded["opponent"] = True

    def record_shot(self, team: str, sim_time_s: float):
        """Record a missile shot."""
        if self._current_episode is None:
            return

        if team == "agent":
            self._current_episode.agent_missiles_fired += 1
            if not self._first_shot_recorded["agent"]:
                self._current_episode.agent_time_to_first_shot_s = sim_time_s
                self._first_shot_recorded["agent"] = True
        elif team == "opponent":
            self._current_episode.opponent_missiles_fired += 1
            if not self._first_shot_recorded["opponent"]:
                self._current_episode.opponent_time_to_first_shot_s = sim_time_s
                self._first_shot_recorded["opponent"] = True

    def record_hit(self, team: str):
        """Record a missile hit."""
        if self._current_episode is None:
            return

        if team == "agent":
            self._current_episode.agent_missiles_hit += 1
        elif team == "opponent":
            self._current_episode.opponent_missiles_hit += 1

    def record_kill(self, killer_team: str, victim_unit_id: int, sim_time_s: float):
        """Record a kill."""
        if self._current_episode is None:
            return

        if killer_team == "agent":
            self._current_episode.agent_kills += 1
            if not self._first_kill_recorded["agent"]:
                self._current_episode.agent_time_to_first_kill_s = sim_time_s
                self._first_kill_recorded["agent"] = True
        elif killer_team == "opponent":
            self._current_episode.opponent_kills += 1
            if not self._first_kill_recorded["opponent"]:
                self._current_episode.opponent_time_to_first_kill_s = sim_time_s
                self._first_kill_recorded["opponent"] = True

        # Record death time for survival calculations
        self._unit_death_times[victim_unit_id] = sim_time_s

    def record_unit_spawn(self, unit_id: int, sim_time_s: float = 0.0):
        """Record when a unit spawns for survival time calculation."""
        self._unit_spawn_times[unit_id] = sim_time_s

    def record_boundary_violation(self, team: str):
        """Record a boundary violation."""
        if self._current_episode is None:
            return

        if team == "agent":
            self._current_episode.agent_boundary_violations += 1
        elif team == "opponent":
            self._current_episode.opponent_boundary_violations += 1

    def record_countermeasure(self, team: str):
        """Record countermeasure usage."""
        if self._current_episode is None:
            return

        if team == "agent":
            self._current_episode.agent_countermeasures_used += 1
        elif team == "opponent":
            self._current_episode.opponent_countermeasures_used += 1

    def record_reward(self, team: str, reward: float):
        """Accumulate reward for a team."""
        if self._current_episode is None:
            return

        if team == "agent":
            self._current_episode.agent_total_reward += reward
        elif team == "opponent":
            self._current_episode.opponent_total_reward += reward

    def end_episode(
        self,
        winner: str,
        termination_reason: str,
        episode_duration_s: float,
        episode_steps: int,
        agent_survivors: int,
        opponent_survivors: int,
    ):
        """
        End the current episode and finalize metrics.

        Args:
            winner: "agent", "opponent", "draw", or "timeout"
            termination_reason: Why the episode ended
            episode_duration_s: Total simulation time
            episode_steps: Total steps taken
            agent_survivors: Number of surviving agents
            opponent_survivors: Number of surviving opponents
        """
        if self._current_episode is None:
            return

        ep = self._current_episode
        ep.winner = winner
        ep.termination_reason = termination_reason
        ep.episode_duration_s = episode_duration_s
        ep.episode_steps = episode_steps
        ep.agent_survivors = agent_survivors
        ep.opponent_survivors = opponent_survivors

        # Calculate survival times
        total_agent_survival = 0.0
        total_opponent_survival = 0.0
        last_agent_death = 0.0
        last_opponent_death = 0.0

        for unit_id, spawn_time in self._unit_spawn_times.items():
            death_time = self._unit_death_times.get(unit_id, episode_duration_s)
            survival = death_time - spawn_time

            # Determine team (assuming unit IDs encode team info or we track it separately)
            # For now, use a simple heuristic - customize based on your ID scheme
            # Agents typically have lower IDs
            if unit_id < 100:  # Placeholder logic
                total_agent_survival += survival
                last_agent_death = max(last_agent_death, death_time)
            else:
                total_opponent_survival += survival
                last_opponent_death = max(last_opponent_death, death_time)

        ep.agent_total_survival_time_s = total_agent_survival
        ep.opponent_total_survival_time_s = total_opponent_survival
        ep.agent_last_man_standing_time_s = (
            last_agent_death if agent_survivors == 0 else episode_duration_s
        )
        ep.opponent_last_man_standing_time_s = (
            last_opponent_death if opponent_survivors == 0 else episode_duration_s
        )

        # Compute derived metrics
        ep.compute_derived_metrics()

        # Store episode
        self.episodes.append(ep)
        self._current_episode = None

    def get_current_episode(self) -> EpisodeMetrics | None:
        """Get the current episode being tracked."""
        return self._current_episode

    def get_episode(self, episode_id: int) -> EpisodeMetrics | None:
        """Get a specific episode by ID."""
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute aggregate statistics across all episodes.

        Returns:
            Dictionary with mean, std, min, max for key metrics.
        """
        if not self.episodes:
            return {}

        # Extract metric arrays
        agent_wins = sum(1 for ep in self.episodes if ep.winner == "agent")
        opponent_wins = sum(1 for ep in self.episodes if ep.winner == "opponent")
        draws = sum(1 for ep in self.episodes if ep.winner == "draw")

        durations = [ep.episode_duration_s for ep in self.episodes]
        agent_kills = [ep.agent_kills for ep in self.episodes]
        opponent_kills = [ep.opponent_kills for ep in self.episodes]
        agent_fired = [ep.agent_missiles_fired for ep in self.episodes]
        opponent_fired = [ep.opponent_missiles_fired for ep in self.episodes]

        return {
            "num_episodes": len(self.episodes),
            "agent_win_rate": agent_wins / len(self.episodes),
            "opponent_win_rate": opponent_wins / len(self.episodes),
            "draw_rate": draws / len(self.episodes),
            "mean_duration_s": np.mean(durations),
            "std_duration_s": np.std(durations),
            "mean_agent_kills": np.mean(agent_kills),
            "mean_opponent_kills": np.mean(opponent_kills),
            "mean_agent_missiles_fired": np.mean(agent_fired),
            "mean_opponent_missiles_fired": np.mean(opponent_fired),
        }

    def export_to_csv(self, filename: str = "study_results.csv", append: bool = True):
        """
        Export all episodes to a CSV file.

        Args:
            filename: Output filename (within output_dir).
            append: If True, append to existing file. If False, overwrite.
        """
        if not self.episodes:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        # Get field names from dataclass
        fieldnames = list(asdict(self.episodes[0]).keys())

        mode = "a" if append and os.path.exists(filepath) else "w"
        write_header = not (append and os.path.exists(filepath))

        with open(filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for ep in self.episodes:
                writer.writerow(ep.to_dict())

    def clear(self):
        """Clear all stored episodes."""
        self.episodes.clear()
        self._current_episode = None
