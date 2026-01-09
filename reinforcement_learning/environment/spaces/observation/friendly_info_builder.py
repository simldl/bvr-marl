"""
Friendly Info Builder - Extracts friendly missiles and fighters information.

Builds observations for:
- Friendly missiles: relative states + phase + seeker lock
- Friendly fighters: relative states
- Missile target indices: which enemy each missile is targeting
- Fighter lock indices: which enemy each fighter has locked
"""
from typing import Tuple, List
import numpy as np
from .helpers import rel_state, pad_generic
from .constants import d_FF, d_FM


class FriendlyInfoBuilder:
    """Builds observation components for friendly units (missiles and fighters)."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self._ff_state_dim = d_FF
        self._fm_state_dim = d_FM

    def build(self, agent_id: str) -> Tuple[
        np.ndarray, np.ndarray,  # fm_states, fm_mask
        np.ndarray, np.ndarray,  # ff_states, ff_mask
        np.ndarray, np.ndarray,  # fm_target_indices, fm_t_mask
        np.ndarray, np.ndarray,  # ff_lock_indices, ff_l_mask
    ]:
        """
        Build friendly info observations.

        Returns:
            Tuple of 8 arrays:
            - fm_states: Friendly missile relative states + phase + seeker [fm_slots, 8]
            - fm_mask: Friendly missile mask [fm_slots]
            - ff_states: Friendly fighter relative states [ff_slots, 6]
            - ff_mask: Friendly fighter mask [ff_slots]
            - fm_target_indices: Target indices for each missile [fm_slots]
            - fm_t_mask: Missile target mask [fm_slots]
            - ff_lock_indices: Lock indices for each fighter [ff_slots]
            - ff_l_mask: Fighter lock mask [ff_slots]
        """
        unit = self.simulator.active_units[agent_id]
        all_agent_ids = self.config.all_agent_ids if hasattr(self.config, 'all_agent_ids') else []

        # Extract friendly missiles and fighters
        friend_missiles = [
            u for u in self.simulator.active_units.values()
            if u.group == unit.group and getattr(u, "is_missile", False)
        ]
        friend_fighters = [
            u for u in self.simulator.active_units.values()
            if u.group == unit.group and not getattr(u, "is_missile", False)
               and u.id != agent_id
        ]

        # Build relative states for fighters (keep simple)
        ff_states = [rel_state(unit, f) for f in friend_fighters]

        # Build enhanced states for missiles (state + phase + seeker)
        fm_states = [self._build_missile_state(unit, m) for m in friend_missiles]

        fm_arr, fm_mask = pad_generic(fm_states, self.config.fm_slots, self._fm_state_dim)
        ff_arr, ff_mask = pad_generic(ff_states, self.config.ff_slots, self._ff_state_dim)

        # Missile-Target-Indices
        fm_targets = []
        for m in friend_missiles:
            tgt = getattr(m, "target", None)
            tgt_id = tgt.id if tgt is not None else None
            idx = np.float32(all_agent_ids.index(tgt_id)) if tgt_id in all_agent_ids else -1.0
            fm_targets.append([idx])
        fm_t_arr, fm_t_mask = pad_generic(fm_targets, self.config.fm_slots, 1)

        # Fighter-Lock-Indices
        ff_locks = []
        for f in friend_fighters:
            locked = f.sensor.get_locked_targets()
            lock_id = next(iter(locked), None) if locked else None
            idx = np.float32(all_agent_ids.index(lock_id)) if lock_id in all_agent_ids else -1.0
            ff_locks.append([idx])
        ff_l_arr, ff_l_mask = pad_generic(ff_locks, self.config.ff_slots, 1)

        return (
            fm_arr, fm_mask,
            ff_arr, ff_mask,
            fm_t_arr.flatten(), fm_t_mask,
            ff_l_arr.flatten(), ff_l_mask,
        )

    def _build_missile_state(self, unit, missile) -> List[float]:
        """
        Build enhanced missile state with phase and seeker info.

        Returns 8-element list:
        [0-5]: Relative state (dx, dy, dz, dvx, dvy, dvz)
        [6]: Phase encoded [0-1] (boost=0.33, mid=0.67, terminal=1.0)
        [7]: Seeker lock state [0/1]
        """
        # Get basic relative state (returns list)
        state = rel_state(unit, missile)

        # Get missile phase
        phase_encoded = self._get_missile_phase(missile)

        # Get seeker lock state
        seeker_lock = self._get_seeker_lock(missile)

        # Concatenate: convert to list if needed and append
        if isinstance(state, np.ndarray):
            return state.tolist() + [phase_encoded, seeker_lock]
        else:
            return state + [phase_encoded, seeker_lock]

    def _get_missile_phase(self, missile) -> float:
        """
        Determine missile phase from flight state.

        Phases:
        - boost: 0.33 (initial powered flight)
        - mid: 0.67 (ballistic/cruise phase)
        - terminal: 1.0 (final approach, seeker active)

        Returns:
            Normalized phase value [0-1]
        """
        try:
            # Check if missile has explicit phase attribute
            if hasattr(missile, 'phase'):
                phase = missile.phase.lower()
                if 'boost' in phase:
                    return 0.33
                elif 'terminal' in phase:
                    return 1.0
                else:
                    return 0.67

            # Fallback: infer from time-of-flight and target distance
            if hasattr(missile, 'time_alive_s'):
                tof = missile.time_alive_s
                # Simple heuristic: boost < 5s, terminal when close to target
                if tof < 5.0:
                    return 0.33

                # Check if close to target for terminal phase
                target = getattr(missile, 'target', None)
                if target and hasattr(missile, 'position') and hasattr(target, 'position'):
                    dx = target.position.lon - missile.position.lon
                    dy = target.position.lat - missile.position.lat
                    dz = getattr(target.position, 'alt', 0) - getattr(missile.position, 'alt', 0)
                    dist_m = np.sqrt((dx * 111_000)**2 + (dy * 111_000)**2 + dz**2)

                    if dist_m < 5000.0:  # Within 5km = terminal
                        return 1.0

                return 0.67  # mid-course

            # Default to mid-course if no info available
            return 0.67

        except Exception:
            return 0.67  # Safe default

    def _get_seeker_lock(self, missile) -> float:
        """
        Check if missile seeker has lock on target.

        Returns:
            1.0 if seeker has lock, 0.0 otherwise
        """
        try:
            # Check for explicit seeker lock attribute
            if hasattr(missile, 'seeker_locked'):
                return 1.0 if missile.seeker_locked else 0.0

            # Check if seeker object exists and has lock
            if hasattr(missile, 'seeker'):
                seeker = missile.seeker
                if hasattr(seeker, 'has_lock'):
                    return 1.0 if seeker.has_lock else 0.0
                if hasattr(seeker, 'locked'):
                    return 1.0 if seeker.locked else 0.0

            # Fallback: assume lock if missile has target and is in terminal phase
            if hasattr(missile, 'target') and missile.target is not None:
                phase = self._get_missile_phase(missile)
                if phase >= 0.9:  # Terminal phase
                    return 1.0

            return 0.0

        except Exception:
            return 0.0  # Safe default
