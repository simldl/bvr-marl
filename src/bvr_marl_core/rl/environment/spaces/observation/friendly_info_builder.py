"""
Friendly Info Builder - Extracts friendly missiles and fighters information.

Builds observations for:
- Friendly missiles: relative states + phase + seeker lock
- Friendly fighters: relative states
- Missile target indices: which enemy each missile is targeting
- Fighter lock indices: which enemy each fighter has locked
"""

import numpy as np

from bvr_marl_core.domain.information_mode import InformationMode, resolve_information_mode
from bvr_marl_core.radar.core.friendly_picture import FriendlyPictureAdapter
from bvr_marl_core.rl.environment.spaces.observation.constants import d_FF, d_FM
from bvr_marl_core.rl.environment.spaces.observation.helpers import (
    normalize_pos_vel,
    pad_tokens,
    rel_state,
    to_body_frame,
)


class FriendlyInfoBuilder:
    """Builds observation components for friendly units (missiles and fighters)."""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self._ff_state_dim = d_FF
        self._fm_state_dim = d_FM
        self.information_mode = resolve_information_mode(
            getattr(config, "information_mode", None), default=InformationMode.SENSOR_LIMITED
        )
        # Precomputed O(1) lookup: agent_id → float index in all_agent_ids
        all_ids = config.all_agent_ids if hasattr(config, "all_agent_ids") else []
        self._agent_id_to_idx: dict = {aid: float(i) for i, aid in enumerate(all_ids)}

    def build(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Build friendly info observations as self-contained tokens.

        The relation "which enemy-fighter slot is this missile/fighter guiding on"
        is folded into the owning token as two compact scalars — a has-target/lock
        flag and the normalized enemy slot index — replacing the old sparse
        one-hot maps (which scaled with ef_slots and broke agent-count robustness).

        Returns:
            tuple of 2 token arrays (validity mask folded into the last column):
            - fm_tokens: Friendly missile tokens [fm_slots, d_FM]
            - ff_tokens: Friendly fighter tokens [ff_slots, d_FF]
        """
        if self.information_mode is InformationMode.SENSOR_LIMITED:
            unit = self.simulator.active_units[agent_id]
            return self._build_from_datalink_reports(unit)

        unit = self.simulator.active_units[agent_id]

        # Extract friendly missiles and fighters
        # Sort by ID for stable observation ordering across episodes
        friend_missiles = sorted(
            [
                u
                for u in self.simulator.active_units.values()
                if u.group == unit.group and getattr(u, "is_missile", False)
            ],
            key=lambda x: x.id,
        )
        friend_fighters = sorted(
            [
                u
                for u in self.simulator.active_units.values()
                if u.group == unit.group
                and not getattr(u, "is_missile", False)
                and u.id != agent_id
                and getattr(u, "type", None) == "Aircraft"
            ],
            key=lambda x: x.id,
        )

        # Enemy-fighter slot index each missile/fighter is guiding on, as
        # (has_relation, slot_norm). slot_norm spreads slot i over [0,1].
        ef_slots = self.config.ef_slots
        id2idx = self._enemy_target_id_to_obs_idx(unit)
        denom = max(ef_slots - 1, 1)

        def _relation(target_id) -> list[float]:
            idx = int(id2idx.get(target_id, -1))
            if 0 <= idx < ef_slots:
                return [1.0, idx / denom]
            return [0.0, 0.0]

        # Friendly fighters: body-frame normalized pos/vel + (has_lock, lock_slot).
        ff_states = []
        for f in friend_fighters:
            state = normalize_pos_vel(to_body_frame(rel_state(unit, f), unit.yaw_deg))
            locked = f.sensor.get_locked_targets()
            lock_id = next(iter(locked), None) if locked else None
            ff_states.append(state + _relation(lock_id))

        # Friendly missiles: state + phase + seeker + (has_target, target_slot).
        fm_states = []
        for m in friend_missiles:
            state = self._build_missile_state(unit, m)
            fm_states.append(state + _relation(self._missile_target_id(m)))

        fm_arr = pad_tokens(fm_states, self.config.fm_slots, self._fm_state_dim - 1)
        ff_arr = pad_tokens(ff_states, self.config.ff_slots, self._ff_state_dim - 1)

        return fm_arr, ff_arr

    def _build_from_datalink_reports(self, unit) -> tuple[np.ndarray, np.ndarray]:
        reports = FriendlyPictureAdapter(self.simulator).reports_for(unit)
        track_slots = self._sensor_limited_track_slots(unit)
        denominator = max(self.config.ef_slots - 1, 1)

        def relation(track_id) -> list[float]:
            slot = track_slots.get(track_id)
            if slot is None:
                return [0.0, 0.0]
            return [1.0, float(slot) / denominator]

        missile_states = []
        fighter_states = []
        for report in reports:
            state = normalize_pos_vel(to_body_frame(report.relative_state_enu, unit.yaw_deg))
            if report.is_missile:
                missile_states.append(
                    state
                    + [report.phase, float(report.seeker_locked)]
                    + relation(report.target_track_id)
                )
            else:
                fighter_states.append(state + relation(report.lock_track_id))
        return (
            pad_tokens(missile_states, self.config.fm_slots, self._fm_state_dim - 1),
            pad_tokens(fighter_states, self.config.ff_slots, self._ff_state_dim - 1),
        )

    def _sensor_limited_track_slots(self, unit) -> dict:
        tracks = getattr(getattr(unit, "sensor", None), "sensor_tracks", ()) or ()
        track_ids = []
        for track in sorted(tracks, key=lambda item: str(item.track_id)):
            if "missile" in track.classification or track.suspect_deception or not track.engageable:
                continue
            track_ids.append(track.track_id)
        return {track_id: slot for slot, track_id in enumerate(track_ids[: self.config.ef_slots])}

    def _missile_target_id(self, missile):
        provider = getattr(missile, "target_provider", None)
        # What the weapon is prosecuting now, then what it was committed to at
        # launch. These differ once the seeker re-issues its track.
        for candidate in (
            getattr(provider, "current_target_id", None),
            getattr(missile, "launch_contact_id", None),
            getattr(missile, "designated_target_id", None),
        ):
            if candidate is not None:
                return candidate
        tgt = getattr(missile, "target", None)
        return getattr(tgt, "id", None) if tgt is not None else None

    def _enemy_target_id_to_obs_idx(self, unit) -> dict:
        target_ids = []
        tracks = getattr(getattr(unit, "sensor", None), "sensor_tracks", []) or []

        def _track_sort_key(track):
            return (int("support" in track.classification), str(track.track_id))

        for track in sorted(tracks, key=_track_sort_key):
            # Oracle observations enumerate evaluator units below. TrackSnapshot
            # intentionally contains no evaluator identity to resolve here.
            continue

        if not target_ids:
            target_ids = [
                u.id
                for u in sorted(
                    [
                        u
                        for u in self.simulator.active_units.values()
                        if getattr(u, "group", None) != unit.group
                        and getattr(u, "type", None) == "Aircraft"
                        and not getattr(u, "is_non_engageable", False)
                    ],
                    key=lambda x: x.id,
                )
            ]

        id2idx = {tid: float(i) for i, tid in enumerate(target_ids[: self.config.ef_slots])}
        for aid, idx in self._agent_id_to_idx.items():
            id2idx.setdefault(aid, idx)
        return id2idx

    def _build_missile_state(self, unit, missile) -> list[float]:
        """
        Build enhanced missile state with phase and seeker info.

        Returns 8-element list:
        [0-5]: Relative state (dx, dy, dz, dvx, dvy, dvz)
        [6]: Phase encoded [0-1] (boost=0.33, mid=0.67, terminal=1.0)
        [7]: Seeker lock state [0/1]
        """
        # Get basic relative state (body-frame, normalized pos/vel)
        state = normalize_pos_vel(to_body_frame(rel_state(unit, missile), unit.yaw_deg))

        phase_encoded = self._get_missile_phase(missile)

        # Get seeker lock state
        seeker_lock = self._get_seeker_lock(missile)

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
            if hasattr(missile, "phase"):
                phase = missile.phase.lower()
                if "boost" in phase:
                    return 0.33
                elif "terminal" in phase:
                    return 1.0
                else:
                    return 0.67

            # Fallback: infer from time-of-flight and target distance
            if hasattr(missile, "time_alive_s"):
                tof = missile.time_alive_s
                # Simple heuristic: boost < 5s, terminal when close to target
                if tof < 5.0:
                    return 0.33

                # Check if close to target for terminal phase
                target = getattr(missile, "target", None)
                if target and hasattr(missile, "position") and hasattr(target, "position"):
                    dx = target.position.lon - missile.position.lon
                    dy = target.position.lat - missile.position.lat
                    dz = getattr(target.position, "alt", 0) - getattr(missile.position, "alt", 0)
                    dist_m = np.sqrt((dx * 111_000) ** 2 + (dy * 111_000) ** 2 + dz**2)

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
            if hasattr(missile, "seeker_locked"):
                return 1.0 if missile.seeker_locked else 0.0

            # Check if seeker object exists and has lock
            if hasattr(missile, "seeker"):
                seeker = missile.seeker
                if hasattr(seeker, "has_lock"):
                    return 1.0 if seeker.has_lock else 0.0
                if hasattr(seeker, "locked"):
                    return 1.0 if seeker.locked else 0.0

            # Fallback: assume lock if missile has target and is in terminal phase
            if hasattr(missile, "target") and missile.target is not None:
                phase = self._get_missile_phase(missile)
                if phase >= 0.9:  # Terminal phase
                    return 1.0

            return 0.0

        except Exception:
            return 0.0  # Safe default
