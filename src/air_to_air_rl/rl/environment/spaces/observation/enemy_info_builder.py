"""
Enemy Info Builder - Extracts enemy units from sensor tracks with per-fighter NEZ and TTI for missiles.

Builds observations for:
- Enemy missiles: relative states + time-to-impact (TTI)
- Enemy fighters: relative states + confidence + per-fighter NEZ (active + passive)
"""

import numpy as np

from .constants import d_EF, d_EM
from .helpers import pad_generic


class EnemyInfoBuilder:
    """Builds observation components for enemy units detected by sensors."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self._ef_state_dim = d_EF
        self._em_state_dim = d_EM

    def build(
        self, agent_id: str
    ) -> tuple[
        np.ndarray,
        np.ndarray,  # em_states, em_mask
        np.ndarray,
        np.ndarray,  # ef_states, ef_mask
    ]:
        """
        Build enemy info observations from sensor tracks.

        Uses Sensor.get_nez_features for consistency with behavior tree.
        Filters out suspected deception tracks (ghosts from jamming).

        Returns:
            tuple of 4 arrays:
            - em_states: Enemy missile relative states + TTI [em_slots, 7]
            - em_mask: Enemy missile mask [em_slots]
            - ef_states: Enemy fighter relative states + confidence + NEZ [ef_slots, 9]
            - ef_mask: Enemy fighter mask [ef_slots]
        """
        unit = self.simulator.active_units[agent_id]
        tracks = unit.sensor.sensor_tracks

        missile_states = []
        fighter_states = []
        fighter_target_ids = []

        # Sort tracks: fighters first (is_support_asset=False), AWACS/support last.
        # Within each group, sort by track ID for stable ordering across episodes.
        def _track_sort_key(t):
            tgt = t[3]
            is_support = 1 if (tgt is not None and getattr(tgt, "is_support_asset", False)) else 0
            tid = t[0] if t[0] is not None else -1
            return (is_support, tid)

        sorted_tracks = sorted(tracks, key=_track_sort_key)

        # Extract states from sensor tracks
        for (
            tid,
            state,
            cov,
            tgt,
            utype,
            ref,
            confidence,
            n_obs,
            lifetime,
            update_count,
            is_deception,
            suspect_deception,
            engagement_id,
            jammer_id,
            engageable,
        ) in sorted_tracks:
            if tgt is None:
                continue
            # Skip tracks that are suspected deception (ghosts) - triangulation removes these
            if suspect_deception:
                continue
            # Skip non-engageable targets (support assets like AWACS when configured)
            if getattr(tgt, "is_non_engageable", False):
                continue
            if getattr(tgt, "is_missile", False):
                # Add TTI for enemy missiles
                tti_normalized = self._estimate_time_to_impact(unit, tgt, state)
                features = list(state[:6]) + [tti_normalized]
                missile_states.append(features)
            else:
                # Enemy fighters (and AWACS): add confidence + support flag (NEZ added below)
                is_support = 1.0 if getattr(tgt, "is_support_asset", False) else 0.0
                features = list(state[:6]) + [confidence, is_support]
                fighter_states.append(features)
                fighter_target_ids.append(tgt.id)

        # Get NEZ features from sensor (same source as BT)
        selected_target_id = (
            getattr(unit.target, "id", None) if hasattr(unit, "target") and unit.target else None
        )
        nez_features = unit.sensor.get_nez_features(self.simulator, selected_target_id)

        # Build NEZ arrays for all enemy fighters and append to fighter states.
        # fighter_states[i] currently has 8 elements: [dx,dy,dz,dvx,dvy,dvz, confidence, is_support].
        # We insert NEZ values between confidence and is_support to get the canonical layout:
        # [0-5] state, [6] confidence, [7] nez_active, [8] nez_passive, [9] is_support_asset
        enhanced_fighter_states = []
        for i, tid in enumerate(fighter_target_ids):
            # Active NEZ: use active_nez from features if this is the selected target
            active_nez_val = (
                nez_features.get("active_nez", 0.0) if tid == selected_target_id else 0.0
            )
            # Passive NEZ: use passive_nez dict
            passive_nez_dict = nez_features.get("passive_nez", {})
            passive_nez_val = passive_nez_dict.get(tid, 0.0)

            # fighter_states[i]: [..., confidence, is_support]  (8 elements)
            base = fighter_states[i]
            # Insert NEZ before the is_support flag (which is the last element)
            enhanced_state = base[:7] + [active_nez_val, passive_nez_val, base[7]]
            enhanced_fighter_states.append(enhanced_state)

        # Pad to fixed size
        ef_arr, ef_mask = pad_generic(
            enhanced_fighter_states, self.config.ef_slots, self._ef_state_dim
        )
        em_arr, em_mask = pad_generic(missile_states, self.config.em_slots, self._em_state_dim)

        return em_arr, em_mask, ef_arr, ef_mask

    def _estimate_time_to_impact(self, ownship, missile, rel_state) -> float:
        """
        Estimate time-to-impact for incoming enemy missile.

        Uses relative position and velocity to estimate TTI,
        normalized to [0, 1] range (0 = imminent, 1 = far/no threat).

        Args:
            ownship: Own aircraft
            missile: Enemy missile object
            rel_state: Relative state vector [dx, dy, dz, dvx, dvy, dvz, ...]

        Returns:
            Normalized TTI [0-1] (0 = impact imminent, 1 = far future)
        """
        try:
            # Extract relative position and velocity
            dx, dy, dz = rel_state[0], rel_state[1], rel_state[2]
            dvx, dvy, dvz = rel_state[3], rel_state[4], rel_state[5]

            # Calculate current range
            range_m = np.sqrt(dx**2 + dy**2 + dz**2)

            # Calculate closing velocity (negative of relative velocity magnitude projected onto range vector)
            closing_velocity = -(dvx * dx + dvy * dy + dvz * dz) / (range_m + 1e-6)

            # Estimate TTI (simple linear extrapolation)
            if closing_velocity > 0:
                tti_s = range_m / closing_velocity
            else:
                # Missile not closing (maybe defeated or off-track)
                tti_s = 1000.0  # Very large value

            # Normalize TTI: 0-60s maps to [0, 1]
            # 0s (imminent) -> 0.0
            # 60s+ (far) -> 1.0
            tti_normalized = np.clip(tti_s / 60.0, 0.0, 1.0)

            return tti_normalized

        except Exception:
            return 1.0  # Safe default (far threat)
