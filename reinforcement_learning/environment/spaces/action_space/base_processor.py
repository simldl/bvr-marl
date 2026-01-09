"""
Base action processor initialization and configuration.
Manages agent state and component initialization.
"""
import math
import numpy as np
from .physics import EnvelopeCalculator, DragCalculator, EnergyCalculator
from .automation import MissileAutomation, WeaponCooldowns
from .processors import EnergyProcessor, LiftVectorProcessor, TriggerProcessor
from .utils import DeadzoneFilter, TargetSorter, DebugInfoCollector


class ActionProcessorBase:
    """Base class for action processor initialization and state management."""

    def __init__(self, simulator):
        """Initialize action processor with all components."""
        self.simulator = simulator
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
        self.target_sorter = TargetSorter()
        self.debug_collector = DebugInfoCollector()

        # Agent states
        self.agent_states = {}
        self.state = {}  # Visualization state

    def configure_automation(self, enable_missile_automation=False, missile_auto_sqi_threshold=0.3,
                           missile_auto_max_per_target=2, missile_auto_long_cooldown_s=10.0):
        """Configure automated missile firing system."""
        self.missile_auto.configure(
            enable_missile_automation,
            missile_auto_sqi_threshold,
            missile_auto_max_per_target,
            missile_auto_long_cooldown_s
        )

    def _init_agent_state(self, agent_id: int, unit):
        """Initialize state for an agent if not exists."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {
                'u_bar': np.zeros(10),
                'v_bar': unit.speed,
                'v_desired': unit.speed,
                'throttle_filtered': unit.control.throttle,
                'n_cmd_filtered': 1.0,
                'phi_cmd_filtered': 0.0,
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
        return self.debug_collector.get_debug_info(state, self.use_energy_space, self.trigger_proc.trigger_threshold)

    def get_training_signals(self, agent_id: int) -> dict:
        """Get training signals for reward hygiene."""
        if agent_id not in self.agent_states:
            return self.debug_collector.get_training_signals({})
        return self.debug_collector.get_training_signals(self.agent_states[agent_id])

    def reset_agent_state(self, agent_id: int):
        """Reset state for an agent."""
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]

    def get_envelope_scalars(self, agent_id: int) -> dict:
        """Get envelope scalars for policy observation."""
        if agent_id not in self.agent_states:
            return {'s_factor': 1.0, 'c_factor': 1.0}

        unit = self.simulator.active_units[agent_id]
        state = self.agent_states[agent_id]
        s_factor, c_factor = self.envelope_calc.compute_envelope_scalars(unit, state['v_bar'], unit.position.alt)

        return self._build_envelope_scalars_dict(unit, state, s_factor, c_factor)

    def _build_envelope_scalars_dict(self, unit, state, s_factor, c_factor) -> dict:
        """Build envelope scalars dictionary with Ps limits."""
        scalars = {'s_factor': s_factor, 'c_factor': c_factor}

        if self.use_energy_space:
            alt = unit.position.alt
            v_bar = max(state['v_bar'], 30.0)
            n_current = state.get('n_cmd_filtered', 1.0)

            if hasattr(unit.physics, 'get_engine_force'):
                Ps_min, Ps_max = self.energy_calc.get_ps_limits(unit, v_bar, alt, n_current, s_factor, c_factor)
                scalars.update({
                    'Ps_max': Ps_max,
                    'Ps_min': Ps_min,
                    'Ps_range': Ps_max - Ps_min,
                    'n_max': min(unit.physics.n_max, s_factor * c_factor * unit.physics.n_max),
                    'phi_max_deg': self.lift_vector_proc.phi_max_deg * s_factor * c_factor,
                })

        return scalars
