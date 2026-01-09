"""
Debug information collector for action processing.
Provides diagnostic data and training signals.
"""


class DebugInfoCollector:
    """Collect debug and training signal information."""

    def __init__(self):
        """Initialize debug info collector."""
        pass

    def init_training_signals(self) -> dict:
        """Initialize training signal counters for an agent."""
        return {
            'valid_missile_fires_this_step': 0,
            'vetoed_missile_attempts_this_step': 0,
            'valid_gun_fires_this_step': 0,
            'vetoed_gun_attempts_this_step': 0,
            'last_lock_ok': False,
            'last_fov_ok': False,
            'last_target_id': None,
        }

    def reset_step_counters(self, state: dict):
        """Reset per-step training signal counters."""
        state['valid_missile_fires_this_step'] = 0
        state['vetoed_missile_attempts_this_step'] = 0
        state['valid_gun_fires_this_step'] = 0
        state['vetoed_gun_attempts_this_step'] = 0

    def get_debug_info(self, state: dict, use_energy_space: bool, trigger_threshold: float) -> dict:
        """
        Get debug information about action processing state.

        Args:
            state: Agent state dict
            use_energy_space: Whether using energy space mode
            trigger_threshold: Trigger activation threshold

        Returns:
            Debug information dict
        """
        debug_info = {
            'filtered_commands': {
                'u_bar': state['u_bar'].tolist(),
                'v_bar': state['v_bar'],
            },
            'target_state': {
                'index': state['target_index'],
                'hold_time_left_s': state['target_hold_time_left_s'],
                'last_bin': state['last_target_bin'],
                'sorted_candidates_count': len(state.get('target_candidates_sorted', [])),
            },
            'trigger_mode': 'simple_threshold',
            'trigger_threshold': trigger_threshold,
            'mode': 'energy_space' if use_energy_space else 'legacy',
        }

        if use_energy_space:
            debug_info['energy_space'] = {
                'n_cmd_filtered': state.get('n_cmd_filtered', 1.0),
                'phi_cmd_filtered': state.get('phi_cmd_filtered', 0.0),
            }

        # Add trigger states (computed from u_bar and trigger_threshold)
        u_bar = state['u_bar']
        debug_info['trigger_states'] = [val >= trigger_threshold for val in u_bar]

        return debug_info

    def get_training_signals(self, state: dict) -> dict:
        """
        Get training signals for reward hygiene.

        Args:
            state: Agent state dict

        Returns:
            Training signals dict
        """
        return {
            'valid_missile_fires': state.get('valid_missile_fires_this_step', 0),
            'vetoed_missile_attempts': state.get('vetoed_missile_attempts_this_step', 0),
            'valid_gun_fires': state.get('valid_gun_fires_this_step', 0),
            'vetoed_gun_attempts': state.get('vetoed_gun_attempts_this_step', 0)
        }
