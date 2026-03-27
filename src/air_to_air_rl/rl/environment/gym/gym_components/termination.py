"""
Episode termination logic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air_to_air_rl.simulator.core.simulator import Simulator


class TerminationChecker:
    """Handles episode termination conditions."""

    def __init__(self, simulator: Simulator, config):
        self.simulator = simulator
        self.config = config

        # Episode timing
        self.episode_start_time = None
        self.simulation_time = 0.0

    def reset_timing(self, tick_secs: float):
        """Reset timing for new episode."""
        self.episode_start_time = time.time()
        self.simulation_time = 0.0
        self.simulator.tick_secs = tick_secs

    def update_simulation_time(self, tick_secs: float):
        """Update simulation time after a step."""
        self.simulation_time += tick_secs

    def check_termination(
        self,
        current_step: int,
        agent_to_unit_id: dict[str, int],
        agent_ids: list[str],
        opponent_ids: list[str],
    ) -> tuple[bool, bool, list[str], float]:
        """
        Check if episode should terminate.

        Returns:
            (terminated, truncated, end_reasons, elapsed_real_time)
        """
        # Check if any units are alive
        any_alive = any(uid in self.simulator.active_units for uid in agent_to_unit_id.values())

        agents_alive = any(
            agent_to_unit_id.get(aid) in self.simulator.active_units
            for aid in agent_ids
            if agent_to_unit_id.get(aid) is not None
        )
        opponents_alive = any(
            agent_to_unit_id.get(aid) in self.simulator.active_units
            for aid in opponent_ids
            if agent_to_unit_id.get(aid) is not None
        )

        # Check early termination conditions
        early_termination = False
        if self.config.early_term_all_enemies_dead and (not agents_alive or not opponents_alive):
            early_termination = True
        if self.config.early_term_mission_complete and (not agents_alive or not opponents_alive):
            early_termination = True

        if self.config.early_term_no_missiles and not early_termination:
            units_with_missile_info = []
            total_missiles = 0
            missiles_in_flight = 0
            for unit in self.simulator.active_units.values():
                if hasattr(unit, "remaining_missiles"):
                    units_with_missile_info.append(unit)
                    total_missiles += getattr(unit, "remaining_missiles", 0)
            for unit in self.simulator.active_units.values():
                if getattr(unit, "is_missile", False):
                    missiles_in_flight += 1
            if units_with_missile_info and total_missiles == 0 and missiles_in_flight == 0:
                early_termination = True

        # Check time limits
        time_exceeded = current_step >= self.config.max_steps

        real_time_exceeded = False
        elapsed_real_time = 0.0
        if self.config.max_real_time_s is not None and self.episode_start_time is not None:
            elapsed_real_time = time.time() - self.episode_start_time
            real_time_exceeded = elapsed_real_time > self.config.max_real_time_s

        # Build end reasons
        end_reasons = []
        if not any_alive:
            end_reasons.append("all_units_dead")
        if not agents_alive:
            end_reasons.append("all_agents_dead")
        if not opponents_alive:
            end_reasons.append("all_opponents_dead")
        if time_exceeded:
            end_reasons.append(f"max_steps_reached({current_step}/{self.config.max_steps})")
        if real_time_exceeded:
            end_reasons.append(f"real_time_limit({elapsed_real_time:.1f}s)")

        terminated = (not any_alive) or early_termination
        truncated = time_exceeded or real_time_exceeded

        return terminated, truncated, end_reasons, elapsed_real_time

    def compute_episode_info(
        self,
        end_reasons: list[str],
        agents_alive: bool,
        opponents_alive: bool,
        current_step: int,
        state_tracker,
    ) -> dict:
        """Compute episode-end information for logging."""
        # Compute team-level statistics
        total_missiles_fired_team_a = sum(
            state_tracker.episode_missiles_fired.get(aid, 0) for aid in self.config.agent_ids
        )
        total_missiles_fired_team_b = sum(
            state_tracker.episode_missiles_fired.get(aid, 0) for aid in self.config.opponent_ids
        )
        total_kills_team_a = sum(
            state_tracker.episode_kills.get(aid, 0) for aid in self.config.agent_ids
        )
        total_kills_team_b = sum(
            state_tracker.episode_kills.get(aid, 0) for aid in self.config.opponent_ids
        )
        total_deaths_team_a = sum(
            state_tracker.episode_deaths.get(aid, 0) for aid in self.config.agent_ids
        )
        total_deaths_team_b = sum(
            state_tracker.episode_deaths.get(aid, 0) for aid in self.config.opponent_ids
        )

        # Compute normalized statistics
        num_agents_team_a = len(self.config.agent_ids)
        num_agents_team_b = len(self.config.opponent_ids)
        mean_kills_per_agent_team_a = (
            total_kills_team_a / num_agents_team_a if num_agents_team_a > 0 else 0.0
        )
        mean_kills_per_agent_team_b = (
            total_kills_team_b / num_agents_team_b if num_agents_team_b > 0 else 0.0
        )
        mean_deaths_per_agent_team_a = (
            total_deaths_team_a / num_agents_team_a if num_agents_team_a > 0 else 0.0
        )
        mean_deaths_per_agent_team_b = (
            total_deaths_team_b / num_agents_team_b if num_agents_team_b > 0 else 0.0
        )

        # Compute diagnostic metrics
        sum(state_tracker.episode_valid_missile_shots.get(aid, 0) for aid in self.config.agent_ids)
        sum(
            state_tracker.episode_valid_missile_shots.get(aid, 0)
            for aid in self.config.opponent_ids
        )
        sum(state_tracker.episode_vetoed_missile_shots.get(aid, 0) for aid in self.config.agent_ids)
        sum(
            state_tracker.episode_vetoed_missile_shots.get(aid, 0)
            for aid in self.config.opponent_ids
        )

        # Count boundary deaths
        boundary_deaths_team_a = sum(
            1 for aid in self.config.agent_ids if aid in state_tracker.boundary_violators
        )
        boundary_deaths_team_b = sum(
            1 for aid in self.config.opponent_ids if aid in state_tracker.boundary_violators
        )

        return {
            "episode_end_reason": end_reasons,
            "simulation_time_s": self.simulation_time,
            "simulation_steps": current_step,
            "agents_survived": agents_alive,
            "opponents_survived": opponents_alive,
            "team_a_missiles_fired": total_missiles_fired_team_a,
            "team_b_missiles_fired": total_missiles_fired_team_b,
            "team_a_kills": total_kills_team_a,
            "team_b_kills": total_kills_team_b,
            "team_a_deaths": total_deaths_team_a,
            "team_b_deaths": total_deaths_team_b,
            "team_a_mean_kills_per_agent": mean_kills_per_agent_team_a,
            "team_b_mean_kills_per_agent": mean_kills_per_agent_team_b,
            "team_a_mean_deaths_per_agent": mean_deaths_per_agent_team_a,
            "team_b_mean_deaths_per_agent": mean_deaths_per_agent_team_b,
            "team_a_boundary_deaths": boundary_deaths_team_a,
            "team_b_boundary_deaths": boundary_deaths_team_b,
        }
