"""
Episode termination logic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bvr_marl_core.aircraft.systems.fire_veto import (
    WASTED_VETO_CATEGORIES,
    team_wasted_info_key,
)
from bvr_marl_core.domain.launch_geometry import (
    SENSOR_CONTACT_STEPS,
    SENSOR_DIAGNOSTIC_KEYS,
    SENSOR_NEAREST_CONTACT_KM,
)

if TYPE_CHECKING:
    from bvr_marl_core.simulator import Simulator


def _sensor_means(sensor_sums: dict, agent_ids, step_count: int) -> dict:
    """Per-step sensor means, with the right denominator for each key.

    Most keys are counts that exist on every step, so they average over all steps.
    `nearest_contact_km` is only DEFINED on steps where a track exists, so it averages
    over those steps instead -- dividing it by all steps biases it toward zero exactly
    in proportion to how often the agent holds nothing, which is the regime being
    diagnosed and would make the metric lie hardest when it matters most.
    """
    totals = {
        key: sum(sensor_sums.get(aid, {}).get(key, 0.0) for aid in agent_ids)
        for key in SENSOR_DIAGNOSTIC_KEYS
    }
    contact_steps = totals.get(SENSOR_CONTACT_STEPS, 0.0)
    out = {}
    for key, total in totals.items():
        if key == SENSOR_NEAREST_CONTACT_KM:
            out[key] = total / contact_steps if contact_steps > 0 else float("nan")
        elif key == SENSOR_CONTACT_STEPS:
            out[key] = total / max(step_count, 1)
        else:
            out[key] = total / max(step_count, 1)
    return out


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
        agent_to_unit_id: dict | None = None,
    ) -> dict:
        """Compute episode-end information for logging."""
        episode_missile_kills = getattr(state_tracker, "episode_missile_kills", {})
        episode_in_envelope_missile_shots = getattr(
            state_tracker, "episode_in_envelope_missile_shots", {}
        )
        episode_out_of_envelope_missile_shots = getattr(
            state_tracker, "episode_out_of_envelope_missile_shots", {}
        )

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
        total_missile_kills_team_a = sum(
            episode_missile_kills.get(aid, 0) for aid in self.config.agent_ids
        )
        total_missile_kills_team_b = sum(
            episode_missile_kills.get(aid, 0) for aid in self.config.opponent_ids
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
        alive_count_team_a = max(num_agents_team_a - total_deaths_team_a, 0)
        alive_count_team_b = max(num_agents_team_b - total_deaths_team_b, 0)

        # Compute diagnostic metrics
        valid_shots_team_a = sum(
            state_tracker.episode_valid_missile_shots.get(aid, 0) for aid in self.config.agent_ids
        )
        valid_shots_team_b = sum(
            state_tracker.episode_valid_missile_shots.get(aid, 0)
            for aid in self.config.opponent_ids
        )
        vetoed_shots_team_a = sum(
            state_tracker.episode_vetoed_missile_shots.get(aid, 0) for aid in self.config.agent_ids
        )
        vetoed_shots_team_b = sum(
            state_tracker.episode_vetoed_missile_shots.get(aid, 0)
            for aid in self.config.opponent_ids
        )
        episode_vetoed_suppressed = getattr(state_tracker, "episode_vetoed_missile_suppressed", {})
        episode_vetoed_wasted = getattr(state_tracker, "episode_vetoed_missile_wasted", {})
        episode_vetoed_no_target = getattr(state_tracker, "episode_vetoed_missile_no_target", {})
        no_target_shots_team_a = sum(
            episode_vetoed_no_target.get(aid, 0) for aid in self.config.agent_ids
        )
        no_target_shots_team_b = sum(
            episode_vetoed_no_target.get(aid, 0) for aid in self.config.opponent_ids
        )
        suppressed_shots_team_a = sum(
            episode_vetoed_suppressed.get(aid, 0) for aid in self.config.agent_ids
        )
        suppressed_shots_team_b = sum(
            episode_vetoed_suppressed.get(aid, 0) for aid in self.config.opponent_ids
        )
        wasted_shots_team_a = sum(
            episode_vetoed_wasted.get(aid, 0) for aid in self.config.agent_ids
        )
        wasted_shots_team_b = sum(
            episode_vetoed_wasted.get(aid, 0) for aid in self.config.opponent_ids
        )
        # `wasted` split by the gate that rejected the launch. Emitted per team and
        # per category so a run can be asked "which gate ate the shots?" without a
        # replay -- the collapsed counter can only say "not target selection".
        episode_vetoed_wasted_by_category = getattr(
            state_tracker, "episode_vetoed_missile_wasted_by_category", {}
        )
        wasted_by_category_team = {}
        for team, ids in (("a", self.config.agent_ids), ("b", self.config.opponent_ids)):
            for category in WASTED_VETO_CATEGORIES:
                wasted_by_category_team[team_wasted_info_key(team, category)] = sum(
                    episode_vetoed_wasted_by_category.get(aid, {}).get(category, 0) for aid in ids
                )
        in_envelope_shots_team_a = sum(
            episode_in_envelope_missile_shots.get(aid, 0) for aid in self.config.agent_ids
        )
        in_envelope_shots_team_b = sum(
            episode_in_envelope_missile_shots.get(aid, 0) for aid in self.config.opponent_ids
        )
        out_of_envelope_shots_team_a = sum(
            episode_out_of_envelope_missile_shots.get(aid, 0) for aid in self.config.agent_ids
        )
        out_of_envelope_shots_team_b = sum(
            episode_out_of_envelope_missile_shots.get(aid, 0) for aid in self.config.opponent_ids
        )

        # Count boundary deaths
        boundary_deaths_team_a = sum(
            1 for aid in self.config.agent_ids if aid in state_tracker.boundary_violators
        )
        boundary_deaths_team_b = sum(
            1 for aid in self.config.opponent_ids if aid in state_tracker.boundary_violators
        )

        # Sensor discipline rates (lock and FOV coverage)
        total_steps_a = max(
            sum(state_tracker.episode_steps_count.get(aid, 0) for aid in self.config.agent_ids), 1
        )
        total_steps_b = max(
            sum(state_tracker.episode_steps_count.get(aid, 0) for aid in self.config.opponent_ids),
            1,
        )
        team_a_lock_rate = (
            sum(state_tracker.episode_lock_ok_count.get(aid, 0) for aid in self.config.agent_ids)
            / total_steps_a
        )
        team_b_lock_rate = (
            sum(state_tracker.episode_lock_ok_count.get(aid, 0) for aid in self.config.opponent_ids)
            / total_steps_b
        )
        # Steps on which a viable firing solution existed, counted independently of
        # whether the trigger was pulled -- the opportunity denominator.
        # getattr-guarded like the other episode counters above: minimal trackers and
        # test doubles do not carry every field, and a missing counter must degrade to
        # "no opportunities recorded", not raise.
        episode_shot_opportunities = getattr(state_tracker, "episode_shot_opportunity_count", {})
        steps_denominator = sum(
            getattr(state_tracker, "episode_steps_count", {}).get(aid, 0)
            for aid in self.config.agent_ids
        )
        shot_opportunities_team_a = sum(
            episode_shot_opportunities.get(aid, 0) for aid in self.config.agent_ids
        )
        shot_opportunities_team_b = sum(
            episode_shot_opportunities.get(aid, 0) for aid in self.config.opponent_ids
        )
        # P(fire | can_fire): did the policy pull the trigger on the steps where every
        # launch gate had already passed? Same getattr-with-default treatment as the
        # denominator above, for the same reason.
        episode_fire_attempts_on_opportunity = getattr(
            state_tracker, "episode_fire_attempt_on_opportunity_count", {}
        )
        fire_attempts_on_opportunity_team_a = sum(
            episode_fire_attempts_on_opportunity.get(aid, 0) for aid in self.config.agent_ids
        )
        team_a_fov_rate = (
            sum(state_tracker.episode_fov_ok_count.get(aid, 0) for aid in self.config.agent_ids)
            / total_steps_a
        )
        team_b_fov_rate = (
            sum(state_tracker.episode_fov_ok_count.get(aid, 0) for aid in self.config.opponent_ids)
            / total_steps_b
        )

        # Ammo remaining at episode end (surviving team_a units only)
        team_a_ammo_remaining = 0
        if agent_to_unit_id:
            for aid in self.config.agent_ids:
                uid = agent_to_unit_id.get(aid)
                if uid is not None:
                    unit = self.simulator.active_units.get(uid)
                    if unit is not None:
                        team_a_ammo_remaining += getattr(unit, "remaining_missiles", 0)

        # First-event timing (-1.0 = never occurred this episode)
        first_shot_times_a = [
            state_tracker.episode_first_shot_time[aid]
            for aid in self.config.agent_ids
            if aid in state_tracker.episode_first_shot_time
        ]
        time_to_first_shot_s = min(first_shot_times_a) if first_shot_times_a else -1.0

        first_kill_times_a = [
            state_tracker.episode_first_kill_time[aid]
            for aid in self.config.agent_ids
            if aid in state_tracker.episode_first_kill_time
        ]
        time_to_first_kill_s = min(first_kill_times_a) if first_kill_times_a else -1.0

        return {
            "episode_end_reason": end_reasons,
            "simulation_time_s": self.simulation_time,
            "simulation_steps": current_step,
            "agents_survived": agents_alive,
            "opponents_survived": opponents_alive,
            "team_size": num_agents_team_a,
            "agents_per_side": num_agents_team_a,
            "team_a_team_size": num_agents_team_a,
            "team_b_team_size": num_agents_team_b,
            # Outcome
            "outcome_win": int(total_kills_team_a > total_kills_team_b),
            # Missile usage
            "team_a_missiles_fired": total_missiles_fired_team_a,
            "team_b_missiles_fired": total_missiles_fired_team_b,
            "team_a_valid_missile_shots": valid_shots_team_a,
            "team_b_valid_missile_shots": valid_shots_team_b,
            "team_a_vetoed_missile_shots": vetoed_shots_team_a,
            "team_b_vetoed_missile_shots": vetoed_shots_team_b,
            # Veto attribution: suppressed = doctrine/safety (no-target/cooldown/cap/
            # winchester); wasted = trigger pulled on an invalid geometry (FOV/range/lock).
            "team_a_vetoed_missile_suppressed": suppressed_shots_team_a,
            "team_b_vetoed_missile_suppressed": suppressed_shots_team_b,
            "team_a_vetoed_missile_wasted": wasted_shots_team_a,
            "team_b_vetoed_missile_wasted": wasted_shots_team_b,
            "team_a_vetoed_missile_no_target": no_target_shots_team_a,
            "team_b_vetoed_missile_no_target": no_target_shots_team_b,
            # ... and `wasted` itself split by gate: *_wasted_{fov,range,lock}.
            # Overlapping conditions, so each is bounded by (not equal to) the sum.
            **wasted_by_category_team,
            "team_a_in_envelope_missile_shots": in_envelope_shots_team_a,
            "team_b_in_envelope_missile_shots": in_envelope_shots_team_b,
            "team_a_out_of_envelope_missile_shots": out_of_envelope_shots_team_a,
            "team_b_out_of_envelope_missile_shots": out_of_envelope_shots_team_b,
            "team_a_never_fired": int(total_missiles_fired_team_a == 0),
            "team_a_ammo_remaining": team_a_ammo_remaining,
            # Kill/death tallies
            "team_a_kills": total_kills_team_a,
            "team_b_kills": total_kills_team_b,
            "team_a_missile_kills": total_missile_kills_team_a,
            "team_b_missile_kills": total_missile_kills_team_b,
            "team_a_deaths": total_deaths_team_a,
            "team_b_deaths": total_deaths_team_b,
            "team_a_alive_count": alive_count_team_a,
            "team_b_alive_count": alive_count_team_b,
            "team_a_mean_kills_per_agent": mean_kills_per_agent_team_a,
            "team_b_mean_kills_per_agent": mean_kills_per_agent_team_b,
            "team_a_mean_deaths_per_agent": mean_deaths_per_agent_team_a,
            "team_b_mean_deaths_per_agent": mean_deaths_per_agent_team_b,
            # Boundary
            "team_a_boundary_deaths": boundary_deaths_team_a,
            "team_b_boundary_deaths": boundary_deaths_team_b,
            "total_boundary_deaths": boundary_deaths_team_a + boundary_deaths_team_b,
            # Sensor discipline
            "team_a_lock_rate": team_a_lock_rate,
            "team_b_lock_rate": team_b_lock_rate,
            # Sensor-chain means over the episode, team-A only. Emitted through the
            # same path as shot_opportunities, which is the one proven to survive
            # the runner boundary.
            **_sensor_means(
                getattr(state_tracker, "episode_sensor_sums", {}),
                self.config.agent_ids,
                steps_denominator,
            ),
            # Missile launch geometry + terminal outcome, aggregated per episode by
            # the collector on state_tracker. Emitted here alongside the sensor means
            # because this is the path proven to survive the runner boundary.
            #
            # This emit was written once, lost to a `git stash drop` before it was
            # committed, and its absence then caused three campaigns (v19, v20, v21)
            # to write NaN for every missile column while the agent was firing. The
            # collector was working the whole time; nothing ever read it.
            **(
                state_tracker.missile_diagnostics.episode_metrics()
                if getattr(state_tracker, "missile_diagnostics", None) is not None
                else {}
            ),
            "team_a_shot_opportunities": shot_opportunities_team_a,
            "team_b_shot_opportunities": shot_opportunities_team_b,
            "team_a_fire_attempts_on_opportunity": fire_attempts_on_opportunity_team_a,
            # The ratio itself, so a reader never has to reconstruct it from two counters
            # that could be windowed differently. None (absent) when there were no
            # opportunities at all -- that is "never had a shot", which is a different
            # statement from "had shots and declined every one", and a 0.0 would erase
            # the distinction this metric exists to make.
            "team_a_fire_rate_when_feasible": (
                fire_attempts_on_opportunity_team_a / shot_opportunities_team_a
                if shot_opportunities_team_a > 0
                else None
            ),
            "team_a_fov_rate": team_a_fov_rate,
            "team_b_fov_rate": team_b_fov_rate,
            # Timing
            "time_to_first_shot_s": time_to_first_shot_s,
            "time_to_first_kill_s": time_to_first_kill_s,
        }
