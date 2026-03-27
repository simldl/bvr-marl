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

    def sort_target_candidates(
        self, unit, candidates: list, inbound_counts: dict | None = None
    ) -> list:
        """
        Deterministic target ordering.

        Primary key: number of friendly missiles already inbound to that target
        (targets with fewer inbound missiles are preferred, so unengaged targets
        are attacked first).
        Secondary key: slant-range (meters).
        Tertiary key: unit ID (tie-break for determinism).

        Args:
            unit: Aircraft unit
            candidates: List of target units
            inbound_counts: Optional mapping of target_id -> inbound missile count.
                            Targets absent from the dict are assumed to have 0 inbound.

        Returns:
            Sorted list of targets
        """
        counts = inbound_counts or {}

        def sort_key(target):
            try:
                lat_m_per_deg = 111320.0
                lon_m_per_deg = 111320.0 * math.cos(math.radians(unit.position.lat))

                dx = (target.position.lon - unit.position.lon) * lon_m_per_deg
                dy = (target.position.lat - unit.position.lat) * lat_m_per_deg
                dz = target.position.alt - unit.position.alt

                if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
                    return (float("inf"), float("inf"), str(target.id))

                slant_range_m = math.sqrt(dx * dx + dy * dy + dz * dz)
                return (counts.get(getattr(target, "id", None), 0), slant_range_m, str(target.id))
            except Exception:
                return (float("inf"), float("inf"), str(target.id))

        return sorted(candidates, key=sort_key)

    def init_target_state(self) -> dict:
        """Initialize target selection state for an agent."""
        return {
            "target_index": 0,
            "target_hold_time_left_s": 0.0,
            "last_target_bin": -1,
            "target_candidates_sorted": [],
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
        # Get target candidates (exclude own unit, same-group, missiles, and non-engageable assets)
        raw_candidates = [
            u
            for u in simulator.active_units.values()
            if u.id != unit.id
            and u.group != unit.group
            and not getattr(u, "is_missile", False)
            and not getattr(u, "is_non_engageable", False)
        ]

        if not raw_candidates:
            state["target_candidates_sorted"] = []
            return None

        # Count friendly missiles already inbound to each candidate so that
        # un-engaged targets are preferred over already-targeted ones.
        inbound_counts: dict = {}
        for u in simulator.active_units.values():
            if getattr(u, "is_missile", False) and u.group == unit.group:
                tgt = getattr(u, "target", None)
                tgt_id = getattr(tgt, "id", None)
                if tgt_id is not None:
                    inbound_counts[tgt_id] = inbound_counts.get(tgt_id, 0) + 1

        # Sort: fewest inbound missiles first, then by range, then by ID.
        target_candidates = self.sort_target_candidates(unit, raw_candidates, inbound_counts)

        # Among the preferred tier (minimum inbound count), rotate by the
        # fighter's index among alive friendlies so each fighter naturally
        # covers a different target when counts are tied.
        if len(target_candidates) > 1:
            min_count = inbound_counts.get(getattr(target_candidates[0], "id", None), 0)
            preferred = [
                u
                for u in target_candidates
                if inbound_counts.get(getattr(u, "id", None), 0) == min_count
            ]
            if len(preferred) > 1:
                friendly_fighters = sorted(
                    [
                        u
                        for u in simulator.active_units.values()
                        if u.group == unit.group
                        and not getattr(u, "is_missile", False)
                        and not getattr(u, "is_non_engageable", False)
                    ],
                    key=lambda u: getattr(u, "id", 0),
                )
                fighter_idx = next(
                    (
                        i
                        for i, f in enumerate(friendly_fighters)
                        if getattr(f, "id", None) == getattr(unit, "id", None)
                    ),
                    0,
                )
                offset = fighter_idx % len(preferred)
                preferred = preferred[offset:] + preferred[:offset]
                rest = [
                    u
                    for u in target_candidates
                    if inbound_counts.get(getattr(u, "id", None), 0) > min_count
                ]
                target_candidates = preferred + rest

        state["target_candidates_sorted"] = target_candidates

        n_targets = len(target_candidates)
        target_bin = int(n_targets * action_target)
        target_bin = min(target_bin, n_targets - 1)

        # Apply Δt-aware stickiness
        if target_bin != state["last_target_bin"]:
            state["target_index"] = target_bin
            state["last_target_bin"] = target_bin
            state["target_hold_time_left_s"] = self.tau_hold_s
        elif state["target_hold_time_left_s"] <= 0:
            # Timer expired, allow target change
            state["target_index"] = target_bin
            state["last_target_bin"] = target_bin
            state["target_hold_time_left_s"] = self.tau_hold_s
        else:
            # Decrement timer by actual time step
            state["target_hold_time_left_s"] -= dt

        return target_candidates[state["target_index"]]
