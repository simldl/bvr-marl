"""
Target sorting and selection logic.
Provides deterministic ordering for target selection.
"""
import math


class TargetSorter:
    """Sort and manage target selection."""

    def __init__(self, tau_hold_s: float = 2.0):
        """
        Initialize target sorter.

        Args:
            tau_hold_s: Stickiness duration for target hold (seconds)
        """
        self.tau_hold_s = tau_hold_s

    def sort_target_candidates(self, unit, candidates: list) -> list:
        """
        Deterministic target ordering by slant-range (meters) then ID.

        Args:
            unit: Aircraft unit
            candidates: List of target units

        Returns:
            Sorted list of targets
        """
        def sort_key(target):
            try:
                # Calculate slant-range in meters (3D distance)
                lat_m_per_deg = 111320.0  # meters per degree latitude
                lon_m_per_deg = 111320.0 * math.cos(math.radians(unit.position.lat))

                dx = (target.position.lon - unit.position.lon) * lon_m_per_deg
                dy = (target.position.lat - unit.position.lat) * lat_m_per_deg
                dz = target.position.alt - unit.position.alt

                slant_range_m = math.sqrt(dx*dx + dy*dy + dz*dz)
                return (slant_range_m, str(target.id))
            except Exception:
                return (float('inf'), str(target.id))

        return sorted(candidates, key=sort_key)

    def init_target_state(self) -> dict:
        """Initialize target selection state for an agent."""
        return {
            'target_index': 0,
            'target_hold_time_left_s': 0.0,
            'last_target_bin': -1,
            'target_candidates_sorted': [],
        }

    def select_target(self, unit, simulator, action_target: float, state: dict, dt: float):
        """
        Select target with deterministic ordering and stickiness.

        Args:
            unit: Aircraft unit
            simulator: Simulator instance
            action_target: Target selection action [0,1]
            state: Agent state dict
            dt: Time delta (seconds)

        Returns:
            Selected target unit or None
        """
        # Get target candidates
        raw_candidates = [
            u for u in simulator.active_units.values()
            if u.id != unit.id and u.group != unit.group and not getattr(u, "is_missile", False)
        ]

        if not raw_candidates:
            state['target_candidates_sorted'] = []
            return None

        # Sort deterministically by range then ID
        target_candidates = self.sort_target_candidates(unit, raw_candidates)
        state['target_candidates_sorted'] = target_candidates

        n_targets = len(target_candidates)
        target_bin = int(n_targets * action_target)
        target_bin = min(target_bin, n_targets - 1)

        # Apply Δt-aware stickiness
        if target_bin != state['last_target_bin']:
            state['target_index'] = target_bin
            state['last_target_bin'] = target_bin
            state['target_hold_time_left_s'] = self.tau_hold_s
        elif state['target_hold_time_left_s'] <= 0:
            # Timer expired, allow target change
            state['target_index'] = target_bin
            state['last_target_bin'] = target_bin
            state['target_hold_time_left_s'] = self.tau_hold_s
        else:
            # Decrement timer by actual time step
            state['target_hold_time_left_s'] -= dt

        return target_candidates[state['target_index']]
