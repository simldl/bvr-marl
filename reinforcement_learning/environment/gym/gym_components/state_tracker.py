"""
State tracking for rewards and episode statistics.
"""

from __future__ import annotations
from typing import Dict, Set
from collections import deque


class StateTracker:
    """Tracks state across steps for reward computation and episode statistics."""

    def __init__(self):
        # Position and missile tracking
        self.previous_positions: Dict[str, tuple] = {}  # agent_id -> (lon, lat, alt)
        self.missiles_fired_last_step: Dict[str, int] = {}  # agent_id -> count

        # Phase 2: Enhanced reward tracking
        self.prev_sqi: Dict[str, float] = {}  # agent_id -> float
        self.no_lock_timer: Dict[str, float] = {}  # agent_id -> float (seconds without lock)
        self.nez_steps: Dict[str, int] = {}  # agent_id -> int (consecutive steps in NEZ)
        self.orbit_state: Dict[str, dict] = {}  # agent_id -> {dr_dt_history: deque, duration: float}
        self.time_with_lock: Dict[str, float] = {}  # agent_id -> float (seconds with lock)
        self.had_lock_previous_step: Dict[str, bool] = {}  # agent_id -> bool
        self.team_first_shot: Dict[str, bool] = {}  # team -> bool
        self.prev_tactical_potential: Dict[str, float] = {}  # agent_id -> float (V11 shaping)
        self.prev_energy_advantage: Dict[str, float] = {}  # agent_id -> float (V11 shaping)

        # Episode statistics
        self.episode_missiles_fired: Dict[str, int] = {}  # agent_id -> total missiles
        self.episode_kills: Dict[str, int] = {}  # agent_id -> number of kills
        self.episode_deaths: Dict[str, int] = {}  # agent_id -> 0 or 1

        # Episode diagnostic metrics
        self.episode_valid_missile_shots: Dict[str, int] = {}  # agent_id -> total valid shots
        self.episode_vetoed_missile_shots: Dict[str, int] = {}  # agent_id -> total vetoed attempts
        self.episode_lock_ok_count: Dict[str, int] = {}  # agent_id -> count of steps with radar lock
        self.episode_fov_ok_count: Dict[str, int] = {}  # agent_id -> count of steps with target in FOV
        self.episode_steps_count: Dict[str, int] = {}  # agent_id -> total steps for this agent

        # Death cause tracking
        self.boundary_violators: Set[str] = set()  # agent_ids that died from boundary violation

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
        self.episode_deaths.clear()

        self.episode_valid_missile_shots.clear()
        self.episode_vetoed_missile_shots.clear()
        self.episode_lock_ok_count.clear()
        self.episode_fov_ok_count.clear()
        self.episode_steps_count.clear()

        self.boundary_violators.clear()
        self.last_processed_event_count = 0

    def initialize_agent(self, agent_id: str, position: tuple = None):
        """Initialize tracking for a single agent."""
        if position:
            self.previous_positions[agent_id] = position
        self.missiles_fired_last_step[agent_id] = 0

        self.episode_missiles_fired[agent_id] = 0
        self.episode_kills[agent_id] = 0
        self.episode_deaths[agent_id] = 0

        self.episode_valid_missile_shots[agent_id] = 0
        self.episode_vetoed_missile_shots[agent_id] = 0
        self.episode_lock_ok_count[agent_id] = 0
        self.episode_fov_ok_count[agent_id] = 0
        self.episode_steps_count[agent_id] = 0

    def update_position(self, agent_id: str, position: tuple):
        """Update agent position for next step."""
        self.previous_positions[agent_id] = position

    def record_kill(self, agent_id: str):
        """Record a kill for an agent."""
        self.episode_kills[agent_id] = self.episode_kills.get(agent_id, 0) + 1

    def record_death(self, agent_id: str):
        """Record a death for an agent."""
        self.episode_deaths[agent_id] = 1

    def record_boundary_violation(self, agent_id: str):
        """Record that an agent died from boundary violation."""
        self.boundary_violators.add(agent_id)

    def update_missiles_fired(self, agent_id: str, count: int):
        """Update missiles fired this step."""
        self.missiles_fired_last_step[agent_id] = count
        self.episode_missiles_fired[agent_id] = self.episode_missiles_fired.get(agent_id, 0) + count

    def update_diagnostic_metrics(self, agent_id: str, valid_shots: int, vetoed_shots: int,
                                   lock_ok: bool, fov_ok: bool):
        """Update episode-level diagnostic metrics."""
        self.episode_valid_missile_shots[agent_id] = self.episode_valid_missile_shots.get(agent_id, 0) + valid_shots
        self.episode_vetoed_missile_shots[agent_id] = self.episode_vetoed_missile_shots.get(agent_id, 0) + vetoed_shots
        self.episode_lock_ok_count[agent_id] = self.episode_lock_ok_count.get(agent_id, 0) + (1 if lock_ok else 0)
        self.episode_fov_ok_count[agent_id] = self.episode_fov_ok_count.get(agent_id, 0) + (1 if fov_ok else 0)
        self.episode_steps_count[agent_id] = self.episode_steps_count.get(agent_id, 0) + 1
