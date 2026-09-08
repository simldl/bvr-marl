"""
Base action processor initialization and configuration.
Manages agent state and component initialization.
"""

import math

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.rl.environment.spaces.action_space.automation import (
    MissileAutomation,
    WeaponCooldowns,
)
from bvr_marl_core.rl.environment.spaces.action_space.physics import (
    DragCalculator,
    EnergyCalculator,
    EnvelopeCalculator,
)
from bvr_marl_core.rl.environment.spaces.action_space.processors import (
    EnergyProcessor,
    LiftVectorProcessor,
    TriggerProcessor,
)
from bvr_marl_core.rl.environment.spaces.action_space.utils import (
    DeadzoneFilter,
    DebugInfoCollector,
    TargetSorter,
)


class ActionProcessorBase:
    """Base class for action processor initialization and state management."""

    def __init__(self, simulator, information_mode=None):
        """Initialize action processor with all components."""
        self.simulator = simulator
        self.information_mode = resolve_information_mode(
            information_mode, default=InformationMode.SENSOR_LIMITED
        )
        self.use_energy_space = True
        self.use_speed_control = False

        # Time constants
        self.tau_v = 1.0
        self.tau_T = 0.3

        # Initialize components
        self.envelope_calc = EnvelopeCalculator()
        self.drag_calc = DragCalculator(self.envelope_calc)
        self.energy_calc = EnergyCalculator(self.drag_calc)
        self.missile_auto = MissileAutomation()
        self.weapon_cooldowns = WeaponCooldowns()
        self.energy_proc = EnergyProcessor(self.energy_calc)
        self.lift_vector_proc = LiftVectorProcessor()
        self.trigger_proc = TriggerProcessor()
        self.deadzone_filter = DeadzoneFilter()
        self.target_sorter = TargetSorter(
            max_missiles_per_target=self.weapon_cooldowns.max_missiles_per_target
        )
        self.debug_collector = DebugInfoCollector()

        # Weapon toggles (overridden by configure_automation)
        self.enable_gun = True

        self.agent_states = {}
        self.state = {}  # Visualization state

    def configure_automation(
        self,
        enable_missile_automation=False,
        missile_auto_sqi_threshold=0.3,
        missile_auto_max_per_target=2,
        missile_auto_long_cooldown_s=10.0,
        missile_fire_threshold=0.5,
        enable_gun=True,
    ):
        """Configure automated missile firing system and weapon toggles."""
        self.missile_auto.configure(
            enable_missile_automation,
            missile_auto_sqi_threshold,
            missile_auto_max_per_target,
            missile_auto_long_cooldown_s,
        )
        # Keep the saturation cap consistent across the firing veto, the
        # selection filter (target allocation), and the automation.
        self.weapon_cooldowns.max_missiles_per_target = int(missile_auto_max_per_target)
        self.target_sorter.max_missiles_per_target = int(missile_auto_max_per_target)
        self.trigger_proc.set_threshold(3, float(missile_fire_threshold))
        self.enable_gun = enable_gun

    def _init_agent_state(self, agent_id: int, unit):
        """Initialize state for an agent if not exists."""
        if agent_id not in self.agent_states:
            # The inner loop holds achieved bank and load factor across ticks.
            # That state is meaningless across an episode boundary and would
            # otherwise carry a previous episode's bank into a fresh spawn.
            autopilot = getattr(getattr(unit, "control", None), "autopilot", None)
            if autopilot is not None:
                autopilot.reset()
            self.agent_states[agent_id] = {
                "u_bar": np.zeros(9),
                "v_bar": unit.speed,
                "v_desired": unit.speed,
                "throttle_filtered": unit.control.throttle,
                "n_cmd_filtered": 1.0,
                "phi_cmd_filtered": 0.0,
                **self.target_sorter.init_target_state(),
                **self.weapon_cooldowns.init_agent_cooldowns(),
                **self.debug_collector.init_training_signals(),
            }

    def _get_alpha_dt_aware(self, dt: float, tau: float) -> float:
        """Δt-aware exponential smoothing coefficient."""
        return 1.0 - math.exp(-dt / tau)

    def get_debug_info(self, agent_id: int) -> dict:
        """Get debug information about action processing state."""
        if agent_id not in self.agent_states:
            return {}
        state = self.agent_states[agent_id]
        return self.debug_collector.get_debug_info(
            state,
            self.use_energy_space,
            self.trigger_proc.trigger_threshold,
            trigger_thresholds=self.trigger_proc.per_index_thresholds,
        )

    def get_training_signals(self, agent_id: int) -> dict:
        """Get training signals for reward hygiene."""
        if agent_id not in self.agent_states:
            return self.debug_collector.get_training_signals({})
        return self.debug_collector.get_training_signals(self.agent_states[agent_id])

    def reset_agent_state(self, agent_id: int):
        """Reset state for an agent."""
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]

    def reset(self) -> None:
        """Drop all per-agent action state. MUST be called on episode reset.

        `agent_states` holds the contact-slot registry, the filtered flight state
        (`v_bar`, `n_cmd_filtered`, `throttle_filtered`), target hold timers and
        weapon cooldowns. None of it survives an episode boundary meaningfully, and
        the registry actively breaks if it does: its coast expiry compares
        `sim.elapsed_time_s` against each contact's last-seen time, and that clock
        RESETS to 0 every episode. A contact last seen at t=700 in the previous
        episode is then evaluated as `0 - 700 = -700`, which is never greater than
        the coast timeout, so it never expires.

        Measured on a reused env instance with a trained checkpoint: occupied
        registry slots grew 1.00 -> 1.91 -> 2.06 -> 3.04 -> 4.57 across successive
        episodes while the radar held a steady ~1.3 live tracks. Because the
        target-selection axis bins over OCCUPIED slots, a policy emitting a fixed
        0.6 addresses slot 0 with one contact but slot 1, 2 or 3 as the ghosts pile
        up -- so it designates a stale identity the radar cannot possibly have
        locked. `lock_rate` fell 0.950 -> 0.279 -> 0.025 in lockstep, which is
        exactly the collapse seen in training (0.889 at iteration 2 -> 0.026 by
        iteration 10) and which no single-episode probe can reproduce.
        """
        self.agent_states.clear()

    def get_envelope_scalars(self, agent_id: int) -> dict:
        """Get envelope scalars for policy observation.

        Reuses factors cached in agent state by apply() to avoid a second computation per step.
        """
        if agent_id not in self.agent_states:
            return {"s_factor": 1.0, "c_factor": 1.0}

        unit = self.simulator.active_units[agent_id]
        state = self.agent_states[agent_id]
        # Use factors already computed during apply(); fall back to recompute if missing
        s_factor = state.get("_cached_s_factor")
        c_factor = state.get("_cached_c_factor")
        if s_factor is None or c_factor is None:
            s_factor, c_factor = self.envelope_calc.compute_envelope_scalars(
                unit, state["v_bar"], unit.position.alt
            )

        return self._build_envelope_scalars_dict(unit, state, s_factor, c_factor)

    def _build_envelope_scalars_dict(self, unit, state, s_factor, c_factor) -> dict:
        """Build envelope scalars dictionary with Ps limits."""
        scalars = {"s_factor": s_factor, "c_factor": c_factor}

        if self.use_energy_space:
            alt = unit.position.alt
            v_bar = max(state["v_bar"], 30.0)
            n_current = state.get("n_cmd_filtered", 1.0)

            if hasattr(unit.physics, "get_engine_force"):
                Ps_min, Ps_max = self.energy_calc.get_ps_limits(
                    unit, v_bar, alt, n_current, s_factor, c_factor
                )
                scalars.update(
                    {
                        "Ps_max": Ps_max,
                        "Ps_min": Ps_min,
                        "Ps_range": Ps_max - Ps_min,
                        "n_max": min(unit.physics.n_max, s_factor * c_factor * unit.physics.n_max),
                        "phi_max_deg": self.lift_vector_proc.phi_max_deg * s_factor * c_factor,
                    }
                )

        return scalars
