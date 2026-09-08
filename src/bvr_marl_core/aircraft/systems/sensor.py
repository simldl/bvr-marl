import math

import numpy as np

from bvr_marl_core.aircraft.core.nez import NoEscapeZoneCalculator
from bvr_marl_core.aircraft.core.target_prio import TrackPrioritySystem
from bvr_marl_core.aircraft.systems.missile_warner import MissileWarner
from bvr_marl_core.aircraft.systems.passive_radar import PassiveRadar
from bvr_marl_core.domain.information import TrackSnapshot
from bvr_marl_core.domain.sensing_visibility import is_sensor_invisible_to
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.simulator.core.helpers import units_distance_km

# Observable-evidence hazard rates used by the probabilistic battle-damage
# assessment.  They deliberately permit both false positives and false negatives;
# true aircraft damage is evaluator-only state and is never consulted here.
_BDA_BASE_RATE_PER_S = 0.002
_BDA_DESCENT_RATE_PER_S = 0.9
_BDA_INCONSISTENCY_RATE_PER_S = 0.35
_BDA_CONFIDENCE_COLLAPSE_RATE_PER_S = 0.4
_BDA_TRACK_LOSS_RATE_PER_S = 0.12


class AircraftSensorSystem:
    """
    Orchestrates all onboard sensor systems:
    - Radar (detection & tracking)
    - Passive radar observations
    - Track prioritization
    - No-Escape Zone (active & passive)
    - Missile warning
    """

    def __init__(self, parent):
        self.parent = parent

        self.radar = parent.radar if hasattr(parent, "radar") and parent.radar is not None else None

        self.sensor_tracks = []
        self.prioritized_tracks = []
        self.active_nez = {}
        self.passive_nez = {}
        self._active_dlz = {}  # full DLZ object per target_id; reused by get_nez_features
        self.warnings = []
        # Battle-damage assessment: target ids this platform has confirmed as
        # killed by watching them go down (see update_sensor_data). Exposed to the
        # agent so it does not keep shooting an already-dead contact.
        self.bda_confirmed: set = set()
        self.bda_probability: dict[object, float] = {}
        self._bda_history: dict[object, dict] = {}
        self._bda_thresholds: dict[object, float] = {}

        conf = getattr(parent, "config", {}) or {}
        ang_err = conf.get(
            "passive_radar_angular_error_deg",
            getattr(parent, "passive_radar_angular_error_deg", 5.0),
        )
        rng_err = conf.get(
            "passive_radar_range_error_m", getattr(parent, "passive_radar_range_error_m", 2000.0)
        )
        max_age = conf.get(
            "passive_radar_max_age_s", getattr(parent, "passive_radar_max_age_s", 3.0)
        )

        self.passive_radar = PassiveRadar(
            angular_error_deg=float(ang_err),
            range_error_m=float(rng_err),
            max_age_s=float(max_age),
        )

        self.nez_calc = NoEscapeZoneCalculator(parent)

        mw_delay = conf.get("missile_warning_delay_s", 0.0)
        mw_std = conf.get("missile_warning_delay_std", 0.0)
        self.missile_warner = MissileWarner(
            parent, detection_delay_s=mw_delay, detection_delay_std=mw_std
        )

        self.track_prioritizer = TrackPrioritySystem(parent)

    def tactical_contact(self, track: TrackSnapshot) -> TacticalContact:
        """Export a truth-free contact with the track's source-qualified lineage."""
        if not isinstance(track, TrackSnapshot):
            raise TypeError("Operational sensor tracks must be TrackSnapshot values.")
        return TacticalContact.from_track_snapshot(track)

    def stage_sensor_reports(self, sim, tick_secs) -> None:
        """Freeze raw reports against the common start-of-tick world state."""
        if self.radar:
            stage = getattr(self.radar, "stage_reports_for_sensors", None)
            if callable(stage):
                stage(
                    tick_secs,
                    sim,
                    owner_position=self.parent.position,
                    steer_h=getattr(self.parent, "beam_rate_hz", 5.0),
                    steer_p=getattr(self.parent, "beam_rate_p_hz", 3.0),
                )
                self._legacy_staged_tracks = None
            else:
                self._legacy_staged_tracks = self.radar.update_for_sensors(
                    tick_secs,
                    sim,
                    owner_position=self.parent.position,
                    steer_h=getattr(self.parent, "beam_rate_hz", 5.0),
                    steer_p=getattr(self.parent, "beam_rate_p_hz", 3.0),
                )

    def update_sensor_data(self, sim, tick_secs):
        """Compatibility wrapper for callers outside the globally staged simulator."""
        self.stage_sensor_reports(sim, tick_secs)
        self.update_from_staged_reports(sim, tick_secs)

    def update_from_staged_reports(self, sim, tick_secs) -> None:
        """Fuse frozen reports, publish contacts, warnings, aids, and BDA."""
        sim_time = getattr(sim, "elapsed_time_s", None)
        if self.radar:
            finish = getattr(self.radar, "update_from_staged_sensor_reports", None)
            self.sensor_tracks = (
                finish(tick_secs, sim, owner_position=self.parent.position)
                if callable(finish)
                else self._legacy_staged_tracks
            )
        else:
            self.sensor_tracks = []

        # Single pass: collect non-self units and process passive radar emissions together
        if not hasattr(self, "_last_nez_positions"):
            self.active_nez = {}
            self.passive_nez = {}
            self._active_dlz = {}
            self.dlz_uncertainty = {}
            self._last_nez_positions = {}

        elapsed = sim.elapsed_time_s
        parent_group = self.parent.group
        all_non_self = []
        for unit in sim.active_units.values():
            if unit is self.parent:
                continue
            # Sensor-invisible HOSTILES are dropped here for the same reason the radar
            # drops them at enumeration: everything below is per-candidate. Two paths
            # would otherwise put an "invisible" AWACS back into the picture:
            #   * the RWR/passive branch just below -- an AWACS is a 250 km emitter, and
            #     one-way RWR range is a multiple of that, so it would be the single most
            #     detectable thing on the map while being unseeable by radar;
            #   * the oracle branch, which builds `current_enemy_ids` and the DLZ table
            #     straight from `all_non_self`.
            # The helper is hostile-only, so a friendly AWACS stays fully visible to the
            # team it is supporting.
            if is_sensor_invisible_to(unit, parent_group):
                continue
            all_non_self.append(unit)
            # RWR/passive detection: an enemy is picked up only while its radar is
            # actively emitting (EMCON off = invisible here) and only within the
            # one-way RWR detection range, which exceeds the emitter's own two-way
            # detection range (a receiver hears an emitter before the emitter can
            # burn through on it).
            if unit.group != parent_group and self._is_rwr_detectable(unit):
                streams = getattr(sim, "random_streams", None)
                rng = (
                    streams.generator(
                        "passive_rf", f"{getattr(self.parent, 'id', 0)}:{getattr(unit, 'id', 0)}"
                    )
                    if streams is not None
                    else None
                )
                self.passive_radar.receive_emission(unit, self.parent, elapsed, rng=rng)

        sensor_limited = str(getattr(sim, "information_mode", "sensor_limited")).lower() != "oracle"
        current_enemy_ids = (
            {track.track_id for track in self.sensor_tracks}
            if sensor_limited
            else {u.id for u in all_non_self if u.group != parent_group}
        )
        self.active_nez = {k: v for k, v in self.active_nez.items() if k in current_enemy_ids}
        self.passive_nez = {k: v for k, v in self.passive_nez.items() if k in current_enemy_ids}
        self._active_dlz = {k: v for k, v in self._active_dlz.items() if k in current_enemy_ids}
        self.dlz_uncertainty = {
            k: v for k, v in self.dlz_uncertainty.items() if k in current_enemy_ids
        }
        self._last_nez_positions = {
            k: v for k, v in self._last_nez_positions.items() if k in current_enemy_ids
        }

        if sensor_limited:
            for track in self.sensor_tracks:
                tid, state, covariance = track.track_id, track.state, track.covariance
                try:
                    estimate = self.nez_calc.compute_dlz_from_track(state, covariance)
                except (TypeError, ValueError):
                    continue
                self._active_dlz[tid] = estimate.nominal
                self.active_nez[tid] = max(estimate.nominal.r_nez_out_m, estimate.nominal.r_min_m)
                self.passive_nez[tid] = 0.0
                self.dlz_uncertainty[tid] = estimate
        else:
            # Oracle diagnostics retain exact geometry for comparison studies.
            for u in all_non_self:
                if u.group == parent_group:
                    continue
                dlz = self.nez_calc.compute_dlz(u)
                self._active_dlz[u.id] = dlz
                self.active_nez[u.id] = max(dlz.r_nez_out_m, dlz.r_min_m)
                self.passive_nez[u.id] = self.nez_calc.passive_nez(u)

        if sim_time is not None:
            self.missile_warner.check_for_new_missiles(sim_time, sim)
            self.warnings = self.missile_warner.update(sim_time)
        else:
            self.warnings = []

        self._update_bda(sim, tick_secs)

        raw_tracks = [
            (track.state, track.covariance) for track in getattr(self, "sensor_tracks", [])
        ]
        self.prioritized_tracks = self.track_prioritizer.prioritize(raw_tracks)

    def _update_bda(self, sim, tick_secs: float) -> None:
        """Update kill-assessment beliefs using sensor products only.

        A contact becomes increasingly suspect after persistent rapid descent,
        kinematic motion inconsistent with its previous estimate, confidence
        collapse, or short-term disappearance.  A stable per-contact random
        threshold turns that probability into a fallible confirmation without
        consulting a target object or evaluator damage flag.
        """
        now_s = float(getattr(sim, "elapsed_time_s", 0.0) or 0.0)
        dt_s = max(float(tick_secs), 1e-6)
        seen: set[object] = set()

        for track in self.sensor_tracks:
            tid = track.track_id
            raw_state, raw_covariance = track.state, track.covariance
            if tid is None:
                continue
            try:
                state = np.asarray(raw_state, dtype=float).reshape(-1)
                covariance = np.asarray(raw_covariance, dtype=float)
            except (TypeError, ValueError):
                continue
            if state.size < 6 or covariance.shape[0] < 3 or covariance.shape[1] < 3:
                continue

            seen.add(tid)
            confidence = float(track.confidence)
            previous = self._bda_history.get(tid)
            descent_score = float(np.clip((-state[5] - 15.0) / 45.0, 0.0, 1.0))
            inconsistency_score = 0.0
            confidence_collapse = 0.0
            if previous is not None:
                history_dt = max(now_s - float(previous["time_s"]), dt_s)
                predicted_position = previous["state"][:3] + previous["state"][3:6] * history_dt
                residual_m = float(np.linalg.norm(state[:3] - predicted_position))
                sigma_m = math.sqrt(max(float(np.trace(covariance[:3, :3])), 1.0))
                inconsistency_score = float(np.clip((residual_m / sigma_m - 2.0) / 4.0, 0.0, 1.0))
                confidence_collapse = float(
                    np.clip((float(previous["confidence"]) - confidence) / 0.4, 0.0, 1.0)
                )

            hazard = (
                _BDA_BASE_RATE_PER_S
                + _BDA_DESCENT_RATE_PER_S * descent_score
                + _BDA_INCONSISTENCY_RATE_PER_S * inconsistency_score
                + _BDA_CONFIDENCE_COLLAPSE_RATE_PER_S * confidence_collapse
            )
            self._advance_bda_probability(sim, tid, hazard, dt_s)
            self._bda_history[tid] = {
                "state": state[:6].copy(),
                "confidence": confidence,
                "time_s": now_s,
                "descent_score": descent_score,
            }

        # A recently observed contact disappearing after anomalous flight is
        # weak evidence, not an automatic kill declaration.
        for tid, previous in tuple(self._bda_history.items()):
            if tid in seen:
                continue
            age_s = now_s - float(previous["time_s"])
            if age_s <= 8.0:
                loss_evidence = max(0.1, float(previous.get("descent_score", 0.0)))
                self._advance_bda_probability(
                    sim, tid, _BDA_TRACK_LOSS_RATE_PER_S * loss_evidence, dt_s
                )
            elif age_s > 30.0:
                self._bda_history.pop(tid, None)

    def _advance_bda_probability(self, sim, tid, hazard_per_s: float, dt_s: float) -> None:
        prior = float(self.bda_probability.get(tid, 0.0))
        posterior = 1.0 - (1.0 - prior) * math.exp(-max(hazard_per_s, 0.0) * dt_s)
        self.bda_probability[tid] = float(np.clip(posterior, 0.0, 1.0))
        if tid not in self._bda_thresholds:
            streams = getattr(sim, "random_streams", None)
            if streams is None:
                # Deterministic fallback keeps the operational path independent
                # of module-global random state in lightweight test doubles.
                threshold = 0.5
            else:
                observer_id = getattr(self.parent, "id", "unregistered")
                threshold = float(streams.generator("bda", f"{observer_id}:{tid}").random())
            self._bda_thresholds[tid] = threshold
        if posterior >= self._bda_thresholds[tid]:
            self.bda_confirmed.add(tid)

    # Multiplier from an emitter's two-way radar range to the one-way range at
    # which an RWR can passively detect its emissions.
    RWR_RANGE_FACTOR = 1.3

    def _is_rwr_detectable(self, emitter) -> bool:
        """True if ``emitter`` is currently detectable by this platform's RWR.

        Requires the emitter to have a radar that is actively emitting, and to lie
        within the one-way RWR detection range (RWR_RANGE_FACTOR x the emitter's
        radar max range). Emitters without a radar (e.g. missiles) are ignored here.
        """
        radar = getattr(emitter, "radar", None)
        if radar is None:
            return False
        if not getattr(emitter, "radar_emitting", True):
            return False
        rwr_range_m = float(getattr(radar, "max_range_m", 0.0)) * self.RWR_RANGE_FACTOR
        if rwr_range_m <= 0.0:
            return False
        dist_m = units_distance_km(self.parent, emitter) * 1000.0
        return dist_m <= rwr_range_m

    def get_locked_targets(self):
        """Returns set of locked target_id(s) (Multi-Lock!)."""
        return self.radar.get_locked_targets() if self.radar else set()

    def has_radar_lock(self, target):
        """True if radar lock exists on target."""
        return self.radar.has_radar_lock(target) if self.radar else False

    def select_target(self, candidates, selection_value=None):
        """Returns selected target or None."""
        if not self.radar:
            return None
        locked_candidates = [t for t in candidates if self.has_radar_lock(t)]
        sorted_targets = (
            sorted(locked_candidates, key=lambda t: units_distance_km(self.parent, t))
            if locked_candidates
            else sorted(candidates, key=lambda t: units_distance_km(self.parent, t))
        )
        n = len(sorted_targets)
        if n == 0:
            return None
        selection_value = 0.0 if selection_value is None else max(0.0, min(selection_value, 1.0))
        index = int(selection_value * n)
        if index >= n:
            index = n - 1
        return sorted_targets[index]

    def get_prio_tracks(self):
        return list(self.prioritized_tracks)

    def get_active_nez(self):
        return dict(self.active_nez)

    def get_passive_nez(self):
        return dict(self.passive_nez)

    def get_missile_warnings(self):
        return list(self.warnings)

    def get_nez_features(self, sim, selected_target_id):
        """
        Get NEZ features for selected target.

        Returns dict with:
        - active_nez: scalar (backward compat)
        - active_nez_by_target: dict {target_id: scalar}
        - active_dlz: full DLZ object for the selected target (or None)
        - active_dlz_by_target: dict {target_id: full DLZ object}
        - passive_nez: dict {target_id: scalar}

        Uses the DLZ stored during update_sensor_data(); performs no new computation.
        """
        # O(1) lookup — no linear scan over active_units
        active_dlz = self._active_dlz.get(selected_target_id)
        active_val = self.active_nez.get(selected_target_id, 0.0)

        recognized_ids = set(self.active_nez.keys()) | set(self.passive_nez.keys())
        passive_dict = {tid: self.passive_nez.get(tid, 0.0) for tid in recognized_ids}

        return {
            "active_nez": active_val,
            "active_nez_by_target": dict(self.active_nez),
            "active_dlz": active_dlz,
            "active_dlz_by_target": dict(self._active_dlz),
            "dlz_uncertainty_by_target": dict(getattr(self, "dlz_uncertainty", {})),
            "passive_nez": passive_dict,
        }
