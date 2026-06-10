"""
State tracking for rewards and episode statistics.
"""

from __future__ import annotations

from collections import deque


class StateTracker:
    """Tracks state across steps for reward computation and episode statistics."""

    def __init__(self):
        # Position and missile tracking
        self.previous_positions: dict[str, tuple] = {}  # agent_id -> (lon, lat, alt)
        self.missiles_fired_last_step: dict[str, int] = {}  # agent_id -> count

        # Phase 2: Enhanced reward tracking
        self.prev_sqi: dict[str, float] = {}  # agent_id -> float
        self.no_lock_timer: dict[str, float] = {}  # agent_id -> float (seconds without lock)
        self.nez_steps: dict[str, int] = {}  # agent_id -> int (consecutive steps in NEZ)
        self.orbit_state: dict[
            str, dict
        ] = {}  # agent_id -> {dr_dt_history: deque, duration: float}
        self.time_with_lock: dict[str, float] = {}  # agent_id -> float (seconds with lock)
        self.had_lock_previous_step: dict[str, bool] = {}  # agent_id -> bool
        self.team_first_shot: dict[str, bool] = {}  # team -> bool
        self.prev_tactical_potential: dict[str, float] = {}  # agent_id -> float (V11 shaping)
        self.prev_energy_advantage: dict[str, float] = {}  # agent_id -> float (V11 shaping)

        # Episode statistics
        self.episode_missiles_fired: dict[str, int] = {}  # agent_id -> total missiles
        self.episode_kills: dict[str, int] = {}  # agent_id -> number of kills
        self.episode_missile_kills: dict[str, int] = {}  # agent_id -> kills credited to missiles
        self.episode_deaths: dict[str, int] = {}  # agent_id -> 0 or 1

        # Episode diagnostic metrics
        self.episode_valid_missile_shots: dict[str, int] = {}  # agent_id -> total valid shots
        self.episode_vetoed_missile_shots: dict[str, int] = {}  # agent_id -> total vetoed attempts
        self.episode_lock_ok_count: dict[
            str, int
        ] = {}  # agent_id -> count of steps with radar lock
        self.episode_fov_ok_count: dict[
            str, int
        ] = {}  # agent_id -> count of steps with target in FOV
        self.episode_steps_count: dict[str, int] = {}  # agent_id -> total steps for this agent

        # First-event timing (sim seconds; absent = event never occurred this episode)
        self.episode_first_shot_time: dict[str, float] = {}  # agent_id -> sim_time_s of 1st shot
        self.episode_first_kill_time: dict[str, float] = {}  # agent_id -> sim_time_s of 1st kill

        # Death cause tracking
        self.boundary_violators: set[str] = set()  # agent_ids that died from boundary violation
        self.kill_credited_victims: set[str] = set()
        """Victim agent IDs that have already produced a credited kill this episode."""

        # Event processing
        self.last_processed_event_count: int = 0

    def reset(self):
        """Reset all tracking state for new episode."""
        self.previous_positions.clear()
        self.missiles_fired_last_step.clear()

        self.prev_sqi.clear()
        self.no_lock_timer.clear()
        self.nez_steps.clear()
        self.orbit_state.clear()
        self.time_with_lock.clear()
        self.had_lock_previous_step.clear()
        self.team_first_shot.clear()
        self.prev_tactical_potential.clear()
        self.prev_energy_advantage.clear()

        self.episode_missiles_fired.clear()
        self.episode_kills.clear()
        self.episode_missile_kills.clear()
        self.episode_deaths.clear()

        self.episode_valid_missile_shots.clear()
        self.episode_vetoed_missile_shots.clear()
        self.episode_lock_ok_count.clear()
        self.episode_fov_ok_count.clear()
        self.episode_steps_count.clear()
        self.episode_first_shot_time.clear()
        self.episode_first_kill_time.clear()

        self.boundary_violators.clear()
        self.kill_credited_victims.clear()
        self.last_processed_event_count = 0

    def initialize_agent(self, agent_id: str, position: tuple = None):
        """Initialize tracking for a single agent."""
        if position:
            self.previous_positions[agent_id] = position
        self.missiles_fired_last_step[agent_id] = 0

        self.episode_missiles_fired[agent_id] = 0
        self.episode_kills[agent_id] = 0
        self.episode_missile_kills[agent_id] = 0
        self.episode_deaths[agent_id] = 0

        self.episode_valid_missile_shots[agent_id] = 0
        self.episode_vetoed_missile_shots[agent_id] = 0
        self.episode_lock_ok_count[agent_id] = 0
        self.episode_fov_ok_count[agent_id] = 0
        self.episode_steps_count[agent_id] = 0

    def update_position(self, agent_id: str, position: tuple):
        """Update agent position for next step."""
        self.previous_positions[agent_id] = position

    def record_kill(self, agent_id: str, sim_time_s: float | None = None):
        """Record a kill for an agent."""
        self.episode_kills[agent_id] = self.episode_kills.get(agent_id, 0) + 1
        if sim_time_s is not None and agent_id not in self.episode_first_kill_time:
            self.episode_first_kill_time[agent_id] = sim_time_s

    def record_missile_kill(self, agent_id: str, sim_time_s: float | None = None):
        """Record a missile-caused kill for an agent."""
        self.record_kill(agent_id, sim_time_s)
        self.episode_missile_kills[agent_id] = self.episode_missile_kills.get(agent_id, 0) + 1

    def record_death(self, agent_id: str):
        """Record a death for an agent."""
        self.episode_deaths[agent_id] = 1

    def record_boundary_violation(self, agent_id: str):
        """Record that an agent died from boundary violation."""
        self.boundary_violators.add(agent_id)

    def update_missiles_fired(self, agent_id: str, count: int, sim_time_s: float | None = None):
        """Update missiles fired this step."""
        self.missiles_fired_last_step[agent_id] = count
        self.episode_missiles_fired[agent_id] = self.episode_missiles_fired.get(agent_id, 0) + count
        if count > 0 and sim_time_s is not None and agent_id not in self.episode_first_shot_time:
            self.episode_first_shot_time[agent_id] = sim_time_s

    def update_diagnostic_metrics(
        self, agent_id: str, valid_shots: int, vetoed_shots: int, lock_ok: bool, fov_ok: bool
    ):
        """Update episode-level diagnostic metrics."""
        self.episode_valid_missile_shots[agent_id] = (
            self.episode_valid_missile_shots.get(agent_id, 0) + valid_shots
        )
        self.episode_vetoed_missile_shots[agent_id] = (
            self.episode_vetoed_missile_shots.get(agent_id, 0) + vetoed_shots
        )
        self.episode_lock_ok_count[agent_id] = self.episode_lock_ok_count.get(agent_id, 0) + (
            1 if lock_ok else 0
        )
        self.episode_fov_ok_count[agent_id] = self.episode_fov_ok_count.get(agent_id, 0) + (
            1 if fov_ok else 0
        )
        self.episode_steps_count[agent_id] = self.episode_steps_count.get(agent_id, 0) + 1
