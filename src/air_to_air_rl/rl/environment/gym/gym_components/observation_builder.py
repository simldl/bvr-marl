"""
Observation and info building for agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from air_to_air_rl.rl.environment.gym.env_helpers import coerce_obs_to_space

if TYPE_CHECKING:
    pass


class ObservationInfoBuilder:
    """Builds observations and info dictionaries for agents."""

    def __init__(self, obs_builder, action_processor):
        self.obs_builder = obs_builder
        self.action_processor = action_processor

    def build_observation(self, uid: int, observation_space) -> dict:
        """Build observation for an agent.

        Fast path: sub-builders already emit float32 arrays with correct shapes,
        so coercion and dtype conversion are skipped on the normal runtime path.
        """
        return self.obs_builder.build(uid)

    def build_terminal_observation(self, observation_space) -> dict:
        """Build zero-filled observation for a terminated agent (deterministic, no sample())."""
        from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

        out: dict = {}
        for k, subspace in observation_space.spaces.items():
            if isinstance(subspace, Box):
                out[k] = np.zeros(subspace.shape, dtype=subspace.dtype or np.float32)
            elif isinstance(subspace, Discrete):
                out[k] = np.zeros(1, dtype=np.int64)
            elif isinstance(subspace, (MultiBinary, MultiDiscrete)):
                out[k] = np.zeros(subspace.shape, dtype=np.int64)
            else:
                out[k] = np.zeros(getattr(subspace, "shape", (1,)), dtype=np.float32)
        return out

    def build_agent_info(self, uid: int, state_tracker, aid: str) -> dict:
        """Build info dictionary for an agent."""
        training_signals = self.action_processor.get_training_signals(uid)
        envelope_scalars = self.action_processor.get_envelope_scalars(uid)
        action_state = self.action_processor.agent_states.get(uid, {})

        # Update diagnostic metrics
        state_tracker.update_diagnostic_metrics(
            aid,
            training_signals.get("valid_missile_fires", 0),
            training_signals.get("vetoed_missile_attempts", 0),
            action_state.get("last_lock_ok", False),
            action_state.get("last_fov_ok", False),
        )

        return {
            "valid_missile_shots": training_signals.get("valid_missile_fires", 0),
            "vetoed_missile_shots": training_signals.get("vetoed_missile_attempts", 0),
            "valid_gun_shots": training_signals.get("valid_gun_fires", 0),
            "vetoed_gun_shots": training_signals.get("vetoed_gun_attempts", 0),
            "cooldown_left_missile_s": action_state.get("missile_cooldown_left_s", 0.0),
            "cooldown_left_gun_s": action_state.get("gun_cooldown_left_s", 0.0),
            "last_lock_ok": action_state.get("last_lock_ok", False),
            "last_fov_ok": action_state.get("last_fov_ok", False),
            "last_target_id": action_state.get("last_target_id", None),
            "s_factor": envelope_scalars.get("s_factor", 1.0),
            "c_factor": envelope_scalars.get("c_factor", 1.0),
            "trigger_latches": envelope_scalars.get("trigger_latches", [False] * 6),
            "Ps_min": envelope_scalars.get("Ps_min"),
            "Ps_max": envelope_scalars.get("Ps_max"),
            "Ps_range": envelope_scalars.get("Ps_range"),
            "n_max_env": envelope_scalars.get("n_max"),
            "phi_max_env_deg": envelope_scalars.get("phi_max_deg"),
        }

    def update_tracking_state(
        self,
        aid: str,
        uid: int,
        unit,
        state_tracker,
        tick_secs: float,
        new_sqi: float,
        new_tactical_potential: float,
        new_energy_advantage: float,
    ):
        """Update tracking state for an agent after reward computation."""
        # Update lock timers
        has_lock = hasattr(unit, "locked_targets") and len(unit.locked_targets) > 0
        state_tracker.had_lock_previous_step[aid] = has_lock
        state_tracker.prev_sqi[aid] = new_sqi
        state_tracker.prev_tactical_potential[aid] = new_tactical_potential
        state_tracker.prev_energy_advantage[aid] = new_energy_advantage

        if has_lock:
            state_tracker.no_lock_timer[aid] = 0.0
            state_tracker.time_with_lock[aid] = (
                state_tracker.time_with_lock.get(aid, 0.0) + tick_secs
            )
        else:
            state_tracker.no_lock_timer[aid] = state_tracker.no_lock_timer.get(aid, 0.0) + tick_secs
            state_tracker.time_with_lock[aid] = 0.0

        # Update NEZ steps
        current_zone = "R4"
        if hasattr(unit, "get_state"):
            state = unit.get_state() or {}
            current_zone = state.get("dlz_zone", "R4")
        if current_zone == "R2":
            state_tracker.nez_steps[aid] = state_tracker.nez_steps.get(aid, 0) + 1
        else:
            state_tracker.nez_steps[aid] = 0

        # Initialize orbit state if needed
        if aid not in state_tracker.orbit_state:
            state_tracker.orbit_state[aid] = {}

        # Update position tracking
        if hasattr(unit, "position"):
            state_tracker.update_position(
                aid,
                (
                    unit.position.lon,
                    unit.position.lat,
                    getattr(unit.position, "alt", 0.0),
                ),
            )
