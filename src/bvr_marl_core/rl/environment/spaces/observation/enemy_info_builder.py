"""
Enemy Info Builder - Extracts enemy units from sensor tracks with per-fighter
NEZ/DLZ/R3 shot-window cues and TTI for missiles.

Builds observations for:
- Enemy missiles: relative states + time-to-impact (TTI)
- Enemy fighters: relative states + confidence + per-fighter NEZ/DLZ/SQI/R3 window
"""

import math

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.domain.track_confidence import TRACK_CONFIDENCE_FIRM
from bvr_marl_core.domain.truth_access_guard import resolve_truth_unit
from bvr_marl_core.rl.environment.spaces.observation.constants import d_EF, d_EM, ef_token_dim
from bvr_marl_core.rl.environment.spaces.observation.helpers import (
    normalize_pos_vel,
    normalize_range_m,
    pad_tokens,
    to_body_frame,
)
from bvr_marl_core.rl.environment.spaces.observation.rid import (
    airframe_id_norm,
    missiles_remaining_norm,
    weapon_id_norm,
)

_DLZ_ZONE_TO_OBS = {"R1": 0.25, "R2": 0.50, "R3": 0.75, "R4": 1.00}

# A contact whose identity is resolved: either the ownship holds a weapons-quality
# lock on it, or the fused track confidence is high enough to read the signature.
# Was a hand-picked 0.85, reached on 3% of measured samples -- so the RID feature
# was effectively always 'unknown' unless the target was locked. See
# domain.track_confidence for why anything >= 0.80 acts as 'never' at BVR range.
_RID_CONFIDENCE_THRESHOLD = TRACK_CONFIDENCE_FIRM


class EnemyInfoBuilder:
    """Builds observation components for enemy units detected by sensors."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self._ef_state_dim = ef_token_dim(self.ef_extra_feature_dim)
        self._em_state_dim = d_EM
        self.information_mode = resolve_information_mode(
            getattr(config, "information_mode", None), default=InformationMode.SENSOR_LIMITED
        )

    @property
    def ef_extra_feature_dim(self) -> int:
        """Number of extension columns this builder appends to each enemy token.

        Core adds none. A subclass that overrides ``ef_extra_features`` must report
        the matching width here, and the env config must declare the same value as
        ``ef_extra_dim``, or the observation space and the built token disagree.
        """
        return int(getattr(self.config, "ef_extra_dim", 0) or 0)

    def ef_extra_features(self, *, track, covariance, ownship) -> list[float]:
        """Extension hook: extra enemy-fighter columns, appended before the mask.

        Called once per occupied enemy-fighter slot. ``track`` is the operational
        track the token was built from (``None`` for a slot with no track), and
        ``covariance`` is its state covariance when available. Core returns nothing.
        """
        return []

    @property
    def sensor_limited(self) -> bool:
        return self.information_mode is InformationMode.SENSOR_LIMITED

    @staticmethod
    def _type_name(unit_type) -> str:
        if unit_type is None:
            return ""
        return str(getattr(unit_type, "value", unit_type)).lower()

    @classmethod
    def _is_missile_type(cls, unit_type) -> bool:
        return "missile" in cls._type_name(unit_type)

    @classmethod
    def _is_support_type(cls, unit_type) -> bool:
        return any(name in cls._type_name(unit_type) for name in ("awacs", "support", "aew"))

    def build(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Build enemy info observations from sensor tracks.

        Uses Sensor.get_nez_features for consistency with behavior tree.
        Filters out suspected deception tracks (ghosts from jamming).

        Returns:
            tuple of 2 token arrays (validity mask folded into the last column):
            - em_tokens: Enemy missile tokens [em_slots, d_EM]
            - ef_tokens: Enemy fighter tokens [ef_slots, d_EF]
        """
        unit = self.simulator.active_units[agent_id]
        tracks = unit.sensor.sensor_tracks

        missile_states = []
        fighter_states = []
        fighter_target_ids = []
        fighter_targets = []
        fighter_raw_states = []
        fighter_covariances = []
        # Retained so an extension builder can derive its own columns from the track.
        fighter_tracks = []

        # Sort tracks: fighters first (is_support_asset=False), AWACS/support last.
        # Within each group, sort by track ID for stable ordering across episodes.
        def _track_sort_key(t):
            return (int(self._is_support_type(t.classification)), str(t.track_id))

        sorted_tracks = sorted(tracks, key=_track_sort_key)

        # Extract states from sensor tracks
        for track in sorted_tracks:
            tid = track.track_id
            state = track.state
            cov = track.covariance
            confidence = track.confidence
            # Oracle mode is an explicitly privileged diagnostic surface. Its
            # evaluator lookup is kept here, at that labelled boundary, and is
            # never available to the sensor-limited branch.
            target = (
                None
                if self.sensor_limited
                else resolve_truth_unit(self.simulator, tid, reason="oracle_enemy_identity")
            )
            # Skip tracks that are suspected deception (ghosts) - triangulation removes these
            if track.suspect_deception:
                continue
            # Skip non-engageable targets (support assets like AWACS when configured)
            if not track.engageable or (
                target is not None and getattr(target, "is_non_engageable", False)
            ):
                continue
            body_state = to_body_frame(state[:6], unit.yaw_deg)
            is_missile = (
                self._is_missile_type(track.classification)
                if self.sensor_limited
                else bool(target is not None and getattr(target, "is_missile", False))
            )
            if is_missile:
                # Add TTI for enemy missiles (computed from raw ENU state, then
                # body-frame + normalize the pos/vel block for the observation).
                tti_normalized = self._estimate_time_to_impact(state)
                features = normalize_pos_vel(body_state) + [tti_normalized]
                missile_states.append(features)
            else:
                # Enemy fighters (and AWACS): add confidence + support flag
                # (NEZ/DLZ/R3 shot-window cues are inserted below)
                is_support = (
                    float(self._is_support_type(track.classification))
                    if self.sensor_limited
                    else float(target is not None and getattr(target, "is_support_asset", False))
                )
                features = normalize_pos_vel(body_state) + [confidence, is_support]
                fighter_states.append(features)
                fighter_target_ids.append(tid)
                fighter_targets.append(target)
                fighter_raw_states.append(state)
                fighter_covariances.append(cov)
                fighter_tracks.append(track)

        # Get NEZ features from sensor (same source as BT)
        selected_target_id = None
        nez_features = {}
        if not self.sensor_limited:
            selected_target_id = (
                getattr(unit.target, "id", None)
                if hasattr(unit, "target") and unit.target
                else None
            )
            nez_features = unit.sensor.get_nez_features(self.simulator, selected_target_id)

        # Build per-fighter range features and append to fighter states.
        # fighter_states[i] currently has 8 elements:
        # [dx,dy,dz,dvx,dvy,dvz, confidence, is_support].
        # We insert the trimmed range/tactical cues between confidence and
        # is_support: [0-5] state, [6] confidence, [7] NEZ (merged),
        # [8-11] DLZ r_min/r_tr/SQI/zone, [12] is_support, [13-16] RID, [17] BDA.
        active_nez_by_target = self._dict_feature(
            nez_features.get("active_nez_by_target", {}),
            fallback_key=selected_target_id,
            fallback_value=nez_features.get("active_nez", 0.0),
        )
        passive_nez_by_target = self._dict_feature(nez_features.get("passive_nez", {}))
        active_dlz_by_target = self._dict_feature(nez_features.get("active_dlz_by_target", {}))
        if selected_target_id is not None and nez_features.get("active_dlz") is not None:
            active_dlz_by_target.setdefault(selected_target_id, nez_features.get("active_dlz"))

        # Own weapons-quality locks: identities are readable for these contacts.
        try:
            locked_ids = set(unit.sensor.get_locked_targets())
        except Exception:
            locked_ids = set()

        # Battle-damage-assessment: contacts this platform has confirmed as killed.
        bda_ids = set(getattr(getattr(unit, "sensor", None), "bda_confirmed", None) or ())

        enhanced_fighter_states = []
        for i, tid in enumerate(fighter_target_ids):
            target = fighter_targets[i] if i < len(fighter_targets) else None
            raw_state = fighter_raw_states[i] if i < len(fighter_raw_states) else None
            covariance = fighter_covariances[i] if i < len(fighter_covariances) else None
            dlz = active_dlz_by_target.get(tid)
            active_nez_val = active_nez_by_target.get(tid, 0.0)
            estimated_dlz = None
            if self.sensor_limited:
                try:
                    estimated_dlz = unit.wez.compute_dlz_from_track(raw_state, covariance)
                    dlz = estimated_dlz.nominal
                    active_nez_val = estimated_dlz.nominal.r_nez_out_m
                except (AttributeError, TypeError, ValueError):
                    dlz = None
            if dlz is None and not self.sensor_limited:
                dlz = self._compute_dlz(unit, target)

            passive_nez_val = passive_nez_by_target.get(tid, 0.0)
            dlz_features = self._dlz_obs_features(
                unit, target, raw_state, dlz, estimated_dlz=estimated_dlz
            )

            # fighter_states[i]: [..., confidence, is_support]  (8 elements)
            base = fighter_states[i]
            confidence = base[6]
            rid_features = self._rid_features(target, tid, confidence, locked_ids)
            bda_flag = 1.0 if tid in bda_ids else 0.0
            # A single NEZ range (max of active/passive) replaces the two
            # correlated NEZ numbers used before.
            nez_merged = normalize_range_m(max(active_nez_val, passive_nez_val))
            # Insert NEZ/DLZ before the is_support flag, then append RID + BDA.
            enhanced_state = (
                [*base[:6], base[6], nez_merged, *dlz_features, base[7]] + rid_features + [bda_flag]
            )
            # Extension columns, if a subclass supplies any, before the mask column.
            if self._ef_state_dim > d_EF:
                enhanced_state = enhanced_state + self.ef_extra_features(
                    track=fighter_tracks[i] if i < len(fighter_tracks) else None,
                    covariance=covariance,
                    ownship=unit,
                )
            enhanced_fighter_states.append(enhanced_state)

        # Pad to fixed size, folding the validity mask into each token's last column.
        ef_arr = pad_tokens(enhanced_fighter_states, self.config.ef_slots, self._ef_state_dim - 1)
        em_arr = pad_tokens(missile_states, self.config.em_slots, self._em_state_dim - 1)

        return em_arr, ef_arr

    @staticmethod
    def _rid_features(target, tid, confidence, locked_ids) -> list[float]:
        """Radar-identification block: [known, airframe_id, weapon_id, missiles_left].

        Identity is revealed only for a weapons-quality contact — one the ownship
        currently locks, or one whose fused track confidence is high enough to read
        the return signature. Otherwise all four features are 0.0 (unidentified).
        """
        known = (tid in locked_ids) or (float(confidence) >= _RID_CONFIDENCE_THRESHOLD)
        if not known or target is None:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            1.0,
            airframe_id_norm(target),
            weapon_id_norm(target),
            missiles_remaining_norm(target),
        ]

    @staticmethod
    def _dict_feature(value, *, fallback_key=None, fallback_value=None) -> dict:
        """Return a dict for feature maps; tolerate old scalar mocks."""
        if isinstance(value, dict):
            return dict(value)
        if fallback_key is not None and fallback_value is not None:
            return {fallback_key: fallback_value}
        return {}

    @staticmethod
    def _compute_dlz(unit, target):
        try:
            wez = getattr(unit, "wez", None)
            if wez is not None and target is not None:
                return wez.compute_dlz(target)
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
        return None

    @staticmethod
    def _raw_slant_range_m(raw_state) -> float:
        try:
            return float(math.sqrt(sum(float(x) ** 2 for x in raw_state[:3])))
        except Exception:
            return 0.0

    @staticmethod
    def _zone_from_bands(slant_range_m: float, dlz) -> str:
        if dlz is None:
            return "R4"
        try:
            if slant_range_m < float(dlz.r_min_m):
                return "R1"
            if slant_range_m < float(dlz.r_tr_m):
                return "R2"
            if slant_range_m < float(dlz.r_pi_m):
                return "R3"
            return "R4"
        except Exception:
            return "R4"

    def _dlz_obs_features(self, unit, target, raw_state, dlz, *, estimated_dlz=None) -> list[float]:
        """Return the trimmed, decorrelated DLZ cues: [r_min, r_tr, sqi, zone].

        The r_min/r_tr bracket plus the ordinal zone already localize the shot
        envelope, so the previous r_pi, r_aero edges and the F-pole/supported-shot
        window flag (redundant with zone == R3) were dropped.
        """
        if dlz is None:
            return [0.0, 0.0, 0.0, 0.0]

        slant_range_m = self._raw_slant_range_m(raw_state)
        try:
            wez = getattr(unit, "wez", None)
            zone = wez.zone_for_range(slant_range_m, dlz) if wez is not None else None
        except Exception:
            zone = None
        zone = zone or self._zone_from_bands(slant_range_m, dlz)

        try:
            wez = getattr(unit, "wez", None)
            if estimated_dlz is not None and wez is not None:
                sqi = float(
                    wez.sqi_from_estimate(slant_range_m, estimated_dlz.closing_speed_mps, dlz)
                )
            else:
                sqi = (
                    float(wez.sqi(unit, target, missile=None, dlz=dlz)) if wez is not None else 0.0
                )
        except Exception:
            sqi = 0.0

        return [
            normalize_range_m(getattr(dlz, "r_min_m", 0.0)),
            normalize_range_m(getattr(dlz, "r_tr_m", 0.0)),
            float(np.clip(sqi, 0.0, 1.0)),
            _DLZ_ZONE_TO_OBS.get(str(zone), 0.0),
        ]

    def _estimate_time_to_impact(self, rel_state) -> float:
        """
        Estimate time-to-impact for incoming enemy missile.

        Uses relative position and velocity to estimate TTI,
        normalized to [0, 1] range (0 = imminent, 1 = far/no threat).

        Args:
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
