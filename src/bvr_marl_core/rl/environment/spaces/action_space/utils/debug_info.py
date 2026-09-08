"""
Debug information collector for action processing.
Provides diagnostic data and training signals.
"""

from bvr_marl_core.aircraft.systems.fire_veto import (
    WASTED_VETO_CATEGORIES,
    wasted_category_key,
    wasted_signal_key,
)


class DebugInfoCollector:
    """Collect debug and training signal information."""

    def __init__(self):
        """Initialize debug info collector."""
        pass

    def init_training_signals(self) -> dict:
        """Initialize training signal counters for an agent."""
        signals = {
            "valid_missile_fires_this_step": 0,
            "vetoed_missile_attempts_this_step": 0,
            # Veto attribution (subtotals of vetoed_missile_attempts_this_step):
            #   suppressed = trigger pressed but firing was not permissible (weapon on
            #     cooldown, per-target saturation cap, winchester). These are
            #     doctrine/safety constraints, NOT a policy error.
            #   no_target  = trigger pressed while the target-selection action addressed
            #     an empty contact slot (or the explicit no-target bin). This IS a policy
            #     error, and it is tracked separately from `suppressed` because lumping
            #     it in there left the target-selection head with no learning signal at
            #     all: the head could sit on a permanently empty slot forever while every
            #     shot was silently discarded as a "doctrine" veto.
            #   wasted     = firing was permissible and the trigger was pressed, but the
            #     launch was rejected by geometry (out of FOV/range/lock). This IS a
            #     policy error (pulled the trigger on an invalid solution).
            "vetoed_missile_suppressed_this_step": 0,
            "vetoed_missile_no_target_this_step": 0,
            "vetoed_missile_wasted_this_step": 0,
            # Of the missiles that actually launched, how many were inside the
            # aero NEZ/DLZ envelope (R2/R3) vs out of envelope (too far/too close).
            "in_envelope_missile_fires_this_step": 0,
            "out_of_envelope_missile_fires_this_step": 0,
            "valid_gun_fires_this_step": 0,
            "vetoed_gun_attempts_this_step": 0,
            # A viable firing solution existed on this step -- designated target, lock,
            # FOV, inventory, cooldown clear, under the per-target cap -- REGARDLESS of
            # whether the trigger was pulled. This is the denominator the shot-discipline
            # rates were missing: `valid_shot_rate` and `trigger_precision_rate` both divide
            # by trigger PRESSES, so an agent that never presses reads 0.0 and one that
            # presses once in perfect geometry reads 1.0 while ignoring every other
            # opening. Counting opportunities independently of presses is what makes
            # "did it shoot when it could?" answerable for a passive policy.
            "shot_opportunity_this_step": 0,
            # Numerator for P(fire | can_fire); see weapon_firing.py.
            "fire_attempt_on_opportunity_this_step": 0,
            "last_lock_ok": False,
            "last_fov_ok": False,
            "last_target_id": None,
        }
        # `wasted` split by the gate that rejected the launch, read off the step's
        # FireGates (see fire_veto). Overlapping conditions, so each is bounded by
        # `vetoed_missile_wasted_this_step` rather than summing to it. The collapsed
        # counter stays because every downstream rate is defined against it.
        for category in WASTED_VETO_CATEGORIES:
            signals[wasted_category_key(category)] = 0
        return signals

    def reset_step_counters(self, state: dict):
        """Reset per-step training signal counters."""
        for category in WASTED_VETO_CATEGORIES:
            state[wasted_category_key(category)] = 0
        state["valid_missile_fires_this_step"] = 0
        state["vetoed_missile_attempts_this_step"] = 0
        state["vetoed_missile_suppressed_this_step"] = 0
        state["vetoed_missile_no_target_this_step"] = 0
        state["vetoed_missile_wasted_this_step"] = 0
        state["in_envelope_missile_fires_this_step"] = 0
        state["out_of_envelope_missile_fires_this_step"] = 0
        state["valid_gun_fires_this_step"] = 0
        state["vetoed_gun_attempts_this_step"] = 0
        state["shot_opportunity_this_step"] = 0
        state["fire_attempt_on_opportunity_this_step"] = 0

    def get_debug_info(
        self,
        state: dict,
        use_energy_space: bool,
        trigger_threshold: float,
        trigger_thresholds: dict[int, float] | None = None,
    ) -> dict:
        """
        Get debug information about action processing state.

        Args:
            state: Agent state dict
            use_energy_space: Whether using energy space mode
            trigger_threshold: Trigger activation threshold
            trigger_thresholds: Optional per-action trigger thresholds

        Returns:
            Debug information dict
        """
        trigger_thresholds = trigger_thresholds or {}
        debug_info = {
            "filtered_commands": {
                "u_bar": state["u_bar"].tolist(),
                "v_bar": state["v_bar"],
            },
            "target_state": {
                "index": state["target_index"],
                "hold_time_left_s": state["target_hold_time_left_s"],
                "last_bin": state["last_target_bin"],
                "sorted_candidates_count": len(state.get("target_candidates_sorted", [])),
            },
            "trigger_mode": "simple_threshold",
            "trigger_threshold": trigger_threshold,
            "trigger_thresholds": dict(trigger_thresholds),
            "mode": "energy_space" if use_energy_space else "legacy",
        }

        if use_energy_space:
            debug_info["energy_space"] = {
                "n_cmd_filtered": state.get("n_cmd_filtered", 1.0),
                "phi_cmd_filtered": state.get("phi_cmd_filtered", 0.0),
            }

        # Add trigger states (computed from u_bar and per-index thresholds)
        u_bar = state["u_bar"]
        debug_info["trigger_states"] = [
            val >= trigger_thresholds.get(index, trigger_threshold)
            for index, val in enumerate(u_bar)
        ]

        return debug_info

    def get_training_signals(self, state: dict) -> dict:
        """
        Get training signals for reward hygiene.

        Args:
            state: Agent state dict

        Returns:
            Training signals dict
        """
        signals = {
            "valid_missile_fires": state.get("valid_missile_fires_this_step", 0),
            "vetoed_missile_attempts": state.get("vetoed_missile_attempts_this_step", 0),
            "vetoed_missile_suppressed": state.get("vetoed_missile_suppressed_this_step", 0),
            "vetoed_missile_no_target": state.get("vetoed_missile_no_target_this_step", 0),
            "vetoed_missile_wasted": state.get("vetoed_missile_wasted_this_step", 0),
            "in_envelope_missile_fires": state.get("in_envelope_missile_fires_this_step", 0),
            "out_of_envelope_missile_fires": state.get(
                "out_of_envelope_missile_fires_this_step", 0
            ),
            "valid_gun_fires": state.get("valid_gun_fires_this_step", 0),
            "vetoed_gun_attempts": state.get("vetoed_gun_attempts_this_step", 0),
            "shot_opportunity": state.get("shot_opportunity_this_step", 0),
            "fire_attempt_on_opportunity": state.get("fire_attempt_on_opportunity_this_step", 0),
        }
        for category in WASTED_VETO_CATEGORIES:
            signals[wasted_signal_key(category)] = state.get(wasted_category_key(category), 0)
        return signals
