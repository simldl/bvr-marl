"""
Step execution and reward computation logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from bvr_marl_core.simulator import UnitDestroyedEvent

from .observation_builder import ObservationInfoBuilder

if TYPE_CHECKING:
    from bvr_marl_core.simulator import Simulator


class StepProcessor:
    """Processes environment steps: actions, simulation, observations, rewards."""

    def __init__(
        self, simulator: Simulator, obs_builder, action_processor, reward_calculator, config
    ):
        self.simulator = simulator
        self.obs_builder = obs_builder
        self.action_processor = action_processor
        self.reward_calculator = reward_calculator
        self.config = config
        self.obs_info_builder = ObservationInfoBuilder(obs_builder, action_processor)

    def process_step(
        self,
        actions: dict[str, np.ndarray] | None,
        current_agents: list,
        agent_to_unit_id: dict[str, int],
        observation_space: dict,
        state_tracker,
        helpers,
        tick_secs: float,
        current_step: int = 0,
        sim_time_s: float = 0.0,
    ) -> tuple:
        """
        Process a single environment step.

        Returns:
            (obs, rewards, terminateds, truncateds, infos, kills_this_step)
        """
        # Track missiles fired BEFORE actions
        prev_missile_counts = self._track_missiles_before_actions(current_agents, agent_to_unit_id)

        # Apply actions to alive units
        self._apply_actions(actions, current_agents, agent_to_unit_id)

        # Track boundary violations BEFORE simulation step
        self._track_boundary_violations(current_agents, agent_to_unit_id, state_tracker)

        # Advance simulation and process events
        self.simulator.do_tick()
        kills_this_step = self._process_events(agent_to_unit_id, state_tracker, sim_time_s)

        # Update missiles fired
        self._update_missiles_fired(
            current_agents, agent_to_unit_id, prev_missile_counts, state_tracker, sim_time_s
        )

        # Build observations and compute rewards
        obs, rewards, terminateds, truncateds, infos = self._build_step_outputs(
            current_agents,
            agent_to_unit_id,
            observation_space,
            state_tracker,
            helpers,
            kills_this_step,
            tick_secs,
            current_step,
        )

        return obs, rewards, terminateds, truncateds, infos, kills_this_step

    def _track_missiles_before_actions(self, current_agents: list, agent_to_unit_id: dict) -> dict:
        """Track missile counts before actions are applied."""
        prev_missile_counts = {}
        for aid in current_agents:
            uid = agent_to_unit_id.get(aid)
            if uid is not None and uid in self.simulator.active_units:
                unit = self.simulator.active_units[uid]
                prev_missile_counts[aid] = len(getattr(unit, "missiles", []))
        return prev_missile_counts

    def _apply_actions(self, actions: dict | None, current_agents: list, agent_to_unit_id: dict):
        """Apply actions to alive units."""
        actions = actions or {}
        for aid in current_agents:
            uid = agent_to_unit_id.get(aid)
            if uid is None or uid not in self.simulator.active_units:
                continue
            act = actions.get(aid, np.zeros(10, dtype=np.float32))
            self.action_processor.apply(uid, np.clip(np.asarray(act, np.float32), 0.0, 1.0))

    def _track_boundary_violations(
        self, current_agents: list, agent_to_unit_id: dict, state_tracker
    ):
        """Track boundary violations before simulation step."""
        for aid in current_agents:
            uid = agent_to_unit_id.get(aid)
            if uid is not None and uid in self.simulator.active_units:
                unit = self.simulator.active_units[uid]
                if hasattr(unit, "boundary_violation_active") and unit.boundary_violation_active:
                    state_tracker.record_boundary_violation(aid)

    def _process_events(
        self, agent_to_unit_id: dict, state_tracker, sim_time_s: float = 0.0
    ) -> dict:
        """Process simulator events and track kills."""
        all_events = self.simulator.events
        new_events = all_events[state_tracker.last_processed_event_count :]
        state_tracker.last_processed_event_count = len(all_events)

        kills_this_step = {}
        for event in new_events:
            if isinstance(event, UnitDestroyedEvent):
                self._process_kill_event(
                    event, agent_to_unit_id, state_tracker, kills_this_step, sim_time_s
                )
        return kills_this_step

    def _process_kill_event(
        self,
        event,
        agent_to_unit_id: dict,
        state_tracker,
        kills_this_step: dict,
        sim_time_s: float = 0.0,
    ):
        """Process a single kill event."""
        killer_unit = event.unit_killer
        destroyed_unit = event.unit_destroyed
        if not killer_unit or not destroyed_unit:
            return

        # Find destroyed agent
        destroyed_agent_id = None
        for aid, uid in agent_to_unit_id.items():
            if uid == destroyed_unit.id:
                destroyed_agent_id = aid
                break
        if destroyed_agent_id is None:
            return
        if destroyed_agent_id in state_tracker.kill_credited_victims:
            return

        # Only count if not boundary violation
        died_from_boundary = destroyed_agent_id in state_tracker.boundary_violators
        if died_from_boundary:
            state_tracker.kill_credited_victims.add(destroyed_agent_id)
            return

        # Find killer agent
        killer_is_missile = bool(getattr(killer_unit, "is_missile", False))
        for aid, uid in agent_to_unit_id.items():
            if uid == killer_unit.id:
                state_tracker.record_kill(aid, sim_time_s)
                state_tracker.kill_credited_victims.add(destroyed_agent_id)
                kills_this_step[aid] = kills_this_step.get(aid, 0) + 1
                break
            elif (
                hasattr(killer_unit, "source")
                and hasattr(killer_unit.source, "id")
                and killer_unit.source.id == uid
            ):
                if killer_is_missile:
                    state_tracker.record_missile_kill(aid, sim_time_s)
                else:
                    state_tracker.record_kill(aid, sim_time_s)
                state_tracker.kill_credited_victims.add(destroyed_agent_id)
                kills_this_step[aid] = kills_this_step.get(aid, 0) + 1
                break

    def _update_missiles_fired(
        self,
        current_agents: list,
        agent_to_unit_id: dict,
        prev_missile_counts: dict,
        state_tracker,
        sim_time_s: float = 0.0,
    ):
        """Update missiles fired counts."""
        for aid in current_agents:
            uid = agent_to_unit_id.get(aid)
            if uid is not None and uid in self.simulator.active_units:
                unit = self.simulator.active_units[uid]
                cur = len(getattr(unit, "missiles", []))
                fired = max(0, cur - prev_missile_counts.get(aid, 0))
                state_tracker.update_missiles_fired(aid, fired, sim_time_s)
            else:
                state_tracker.missiles_fired_last_step[aid] = 0

    def _build_step_outputs(
        self,
        current_agents: list,
        agent_to_unit_id: dict,
        observation_space: dict,
        state_tracker,
        helpers,
        kills_this_step: dict,
        tick_secs: float,
        current_step: int,
    ) -> tuple:
        """Build observations, rewards, and info for all agents."""
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        for aid in current_agents:
            uid = agent_to_unit_id.get(aid)
            unit = self.simulator.active_units.get(uid) if uid is not None else None

            if unit is None:
                # Agent died this tick
                obs[aid] = self.obs_info_builder.build_terminal_observation(observation_space[aid])
                died_from_boundary = aid in state_tracker.boundary_violators
                reward_val, _, _, _ = self.reward_calculator.compute_total_reward(
                    None,
                    self.config.map_limits,
                    destroyed=True,
                    died_from_boundary_violation=died_from_boundary,
                )
                rewards[aid] = reward_val
                terminateds[aid] = True
                truncateds[aid] = False
                infos[aid] = {"died_from_boundary": died_from_boundary}
                state_tracker.record_death(aid)
            else:
                # Agent still alive
                obs[aid] = self.obs_info_builder.build_observation(uid, observation_space[aid])
                reward_val, new_sqi, new_tactical_potential, new_energy_advantage = (
                    self._compute_agent_reward(
                        aid,
                        uid,
                        unit,
                        state_tracker,
                        helpers,
                        kills_this_step,
                        tick_secs,
                        current_step,
                    )
                )
                rewards[aid] = reward_val

                # Update tracking state
                self.obs_info_builder.update_tracking_state(
                    aid,
                    uid,
                    unit,
                    state_tracker,
                    tick_secs,
                    new_sqi,
                    new_tactical_potential,
                    new_energy_advantage,
                )

                terminateds[aid] = False
                truncateds[aid] = False
                infos[aid] = self.obs_info_builder.build_agent_info(uid, state_tracker, aid)

        return obs, rewards, terminateds, truncateds, infos

    def _compute_agent_reward(
        self,
        aid: str,
        uid: int,
        unit,
        state_tracker,
        helpers,
        kills_this_step: dict,
        tick_secs: float,
        current_step: int,
    ) -> tuple:
        """Compute reward for a single agent."""
        enemies = helpers.get_enemies_for_agent(aid)
        targets = helpers.get_targets_for_agent(aid)
        incoming_missiles = helpers.get_incoming_missiles_for_agent(aid)
        previous_position = state_tracker.previous_positions.get(aid)
        missiles_fired = state_tracker.missiles_fired_last_step.get(aid, 0)
        previous_altitude = previous_position[2] if previous_position else None
        action_state = self.action_processor.agent_states.get(uid, {})

        return self.reward_calculator.compute_total_reward(
            unit,
            self.config.map_limits,
            enemy_kill_count=kills_this_step.get(aid, 0),
            enemies=enemies,
            targets=targets,
            incoming_missiles=incoming_missiles,
            previous_position=previous_position,
            missiles_fired_this_step=missiles_fired,
            previous_altitude=previous_altitude,
            action_state=action_state,
            prev_sqi=state_tracker.prev_sqi.get(aid, 0.0),
            nez_steps=state_tracker.nez_steps.get(aid, 0),
            no_lock_duration_s=state_tracker.no_lock_timer.get(aid, 0.0),
            orbit_state=state_tracker.orbit_state.get(aid, {}),
            tick_secs=tick_secs,
            time_with_lock_s=state_tracker.time_with_lock.get(aid, 0.0),
            had_lock_previous_step=state_tracker.had_lock_previous_step.get(aid, False),
            prev_tactical_potential=state_tracker.prev_tactical_potential.get(aid, 0.0),
            prev_energy_advantage=state_tracker.prev_energy_advantage.get(aid, 0.0),
            current_step=current_step,
        )
