import math

import numpy as np

from bvr_marl_core.domain.classification import most_likely_class, signature_class_probabilities
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.core.utils import _angles_dist, _doppler
from bvr_marl_core.radar.lock.missile import MissileLockController
from bvr_marl_core.radar.obs.observation import resolution_cell_sigma
from bvr_marl_core.radar.radar import Radar
from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km

# Designated-track association gate (see MissileRadar._designated_association_gate_m).
# A measurement may only inherit the weapon's track identity inside this many sigma of
# the current designated estimate.
_DESIGNATED_GATE_SIGMA_FACTOR = 3.0
# How fast the gate opens while the designated track coasts unmeasured.
_DESIGNATED_COAST_GROWTH_MPS = 400.0
# Hard ceiling, so a long coast never opens the gate onto a separate formation.
_DESIGNATED_GATE_CEILING_M = 15_000.0


class MissileRadar(Radar):
    def __init__(
        self,
        *args,
        lock_ctrl=None,
        owner=None,
        target_provider=None,
        initial_datalink_mode=None,
        original_data_link_mode=None,
        **kwargs,
    ):
        lock_ctrl = lock_ctrl or MissileLockController()
        # Missile seekers ARE susceptible to noise jamming: against a jammer the
        # seeker measures only a bearing (range denied) and homes on the jam. It
        # burns through to a real range only at short (terminal) range. This is
        # gated by the jammer capability, which is off by default.
        kwargs.setdefault("jam_susceptible", True)
        # A seeker update refines an already confirmed launch WeaponTrack, so one
        # valid gated hit is sufficient; aircraft surveillance still requires
        # three independent revisits.
        kwargs.setdefault("tracker_confirmation_hits", 1)
        super().__init__(*args, lock_ctrl=lock_ctrl, owner=owner, **kwargs)

        # Store references
        self.owner = owner
        self.target_provider = target_provider
        # If no explicit initial mode was provided, mirror 'original'
        if initial_datalink_mode is None:
            initial_datalink_mode = original_data_link_mode

        # If a data_link component exists, apply initial mode
        try:
            if hasattr(self, "data_link") and self.data_link is not None and initial_datalink_mode:
                self.data_link.set_mode(initial_datalink_mode)
            # Remember the 'original' (used when switching back)
            self._original_data_link_mode = original_data_link_mode or initial_datalink_mode
        except Exception:
            self._original_data_link_mode = original_data_link_mode or initial_datalink_mode

        self._has_switched = False
        self.gimbal_h_lim_deg = 70.0
        self.gimbal_v_lim_deg = 60.0
        self.gimbal_rate_deg_s = 240.0
        self.slave_seeker_to_los = True
        self.cached_tracks = []
        self.locked_target = None  # Initialize to None
        self.mode = "search"  # Initialize mode

        # Track designated target visibility for retargeting logic
        self._time_since_designated_target_seen = 0.0
        self._designated_target_loss_threshold_s = 5.0  # Allow retargeting after 5s of track loss

        # Consecutive seeker substeps that fed a real (FOV+range-gated) measurement
        # of the guidance target. A streak means a stable own-radar track, so the
        # missile can guide on the seeker alone; a miss resets it, which reopens
        # the datalink so other sources can help re-acquire (see Missile.update).
        self._seeker_meas_streak = 0
        self._seeker_stable_streak = 2

        # Option A: a single fused datalink (fighter/AWACS) measurement staged once
        # per tick, consumed by the first coasting substep so a missile that has
        # lost its own seeker track can re-acquire from the other sources.
        self._pending_datalink_meas = None
        self._datalink_meas_used = False
        # True once the designated track has been measured (seeker or datalink)
        # at any substep this tick. Reset per tick by begin_designated_tick().
        # Predict-only substeps after a measurement don't count as missed updates,
        # so a track measured this tick stays "fresh" (guidance keeps using it /
        # PN stays engaged) instead of being skipped as stale mid-tick.
        self._measured_designated_this_tick = False

    def update(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None
    ):
        self._manage_datalink_mode()
        self.param_policy.update_dynamic_params()
        self._update_beam_steering(tick_secs, steer_h, steer_p)

        own_detections = self._generate_and_transform_detections(owner_position, targets)

        # Noise jamming denies the seeker its range (bearing-only -> home-on-jam)
        # until it burns through at short range. Reuses the shared range-denial path.
        if self.jam_susceptible and sim is not None and hasattr(sim, "ew_world"):
            t = getattr(sim, "elapsed_time_s", 0.0)
            denial = sim.ew_world.collect_range_denial(self, t)
            if denial:
                self._apply_range_denial(own_detections, denial, owner_position)

        if group_radars is None and sim is not None:
            group_radars = self.get_default_group_radars(sim)
        # A full-datalink (next-gen) missile keeps fusing the friendly network — so
        # it inherits cooperative triangulation on a jammer throughout the flight.
        # A classic Fox-3 fuses the network only during mid-course (before its own
        # seeker goes active); once autonomous it flies bearing-only (HOJ) on a jammer.
        cooperative = bool(getattr(self.owner, "full_datalink", False)) or not self._has_switched
        fused_measurements = self._get_fused_measurements(
            owner_position, group_radars, own_detections, cooperative
        )
        operational = getattr(self.owner, "weapon_track", None) is not None
        tracker_measurements = (
            [self._anonymous_cluster(cluster) for cluster in fused_measurements]
            if operational
            else fused_measurements
        )
        if operational and tracker_measurements:
            self._stamp_designated_measurement(tracker_measurements)
        filtered_measurements = self._filter_measurements_for_lock(tracker_measurements, tick_secs)
        self.cached_detections = (
            [self._anonymous_cluster(cluster) for cluster in own_detections]
            if operational
            else own_detections
        )
        self.cached_tracks = self.tracker_manager.update_tracks(
            filtered_measurements, tick_secs, owner_position
        )
        self._update_lock_ctrl(self.cached_tracks)
        return self.cached_tracks

    def _stamp_designated_measurement(self, tracker_measurements):
        """Give the weapon's track identity to a measurement, if one is consistent.

        ``retarget_policy == "track_only"`` means the weapon prosecutes the hypothesis
        it was launched with and nothing else. Stamping whichever measurement is merely
        nearest breaks that in two ways, because the tracker treats the stamp as
        identity rather than as a hint: it gates a stamped pair far more loosely
        (100.0 against the usual chi-square 16.266), and an unassociated stamped
        measurement is used to *create* a track under that very id. So once the engaged
        aircraft leaves the seeker gate and its track retires, the closest remaining
        contact -- a wingman, another formation member -- is handed the weapon's
        identity outright, and the shot transfers to an aircraft nobody engaged.

        Only a measurement consistent with the current designated estimate may carry
        the identity. With none, the track coasts on its prediction and the aircraft
        is re-acquired through it rather than by relabelling a neighbour.
        """
        designated = min(tracker_measurements, key=self._guidance_distance_m)
        if self._designated_measurement_gates(designated):
            designated["preferred_track_id"] = self.owner.weapon_track.snapshot.track_id

    def _designated_measurement_gates(self, measurement) -> bool:
        """True when a measurement is close enough to be this weapon's own target."""
        guidance_position = (
            self.target_provider.get_guidance_target() if self.target_provider is not None else None
        )
        if guidance_position is None:
            # No designated estimate to contradict yet (pre-acquisition): accept, which
            # is what lets the seeker take up its launch hypothesis in the first place.
            return True
        separation_m = self._guidance_distance_m(measurement)
        if not math.isfinite(separation_m):
            return False
        return separation_m <= self._designated_association_gate_m(measurement)

    def _designated_association_gate_m(self, measurement) -> float:
        """Association gate for the designated track, from seeker resolution.

        Sized from the seeker's own measurement uncertainty rather than its raw
        resolution-cell width (see ``resolution_cell_sigma``), then opened by how long
        the estimate has coasted, because an unmeasured track's true position becomes
        less certain over time. Bounded, so a long coast never opens onto a separate
        formation.
        """
        range_m = max(1.0, float(measurement.get("d", 0.0)))
        sigma_m = max(
            resolution_cell_sigma(self.r_res_m),
            math.radians(resolution_cell_sigma(self.a_res_deg)) * range_m,
        )
        coast_growth_m = _DESIGNATED_COAST_GROWTH_MPS * self._time_since_designated_target_seen
        return min(
            _DESIGNATED_GATE_CEILING_M,
            _DESIGNATED_GATE_SIGMA_FACTOR * sigma_m + coast_growth_m,
        )

    def _manage_datalink_mode(self):
        """Manage datalink mode transitions (other -> full for active seeker)."""
        if not self._has_switched:
            if self.should_activate_own_radar():
                import logging

                logger = logging.getLogger(__name__)
                old_mode = self.data_link.get_mode()
                self.data_link.set_mode(self._original_data_link_mode)
                logger.debug(
                    f"Missile {getattr(self.owner, 'id', 'unknown')} activating own radar: "
                    f"datalink {old_mode} -> {self._original_data_link_mode}"
                )
                self._has_switched = True
            else:
                self.data_link.set_mode("other")
        elif self.data_link.get_mode() == "other":
            self.data_link.set_mode(self._original_data_link_mode)

    def has_stable_seeker_track(self) -> bool:
        """True when the own seeker has fed the tracker a real measurement (target
        inside FOV + range) on enough recent substeps — i.e. the missile has a
        stable own-radar track and does not need the datalink to guide. Reset to
        coasting by ``_predict_designated_track_substep`` on a seeker miss."""
        return self._seeker_meas_streak >= self._seeker_stable_streak

    def prepare_datalink_reacquire(self, sim, owner_position, target_id):
        """Stage one fused datalink (fighter/AWACS) measurement of ``target_id``
        for this tick (Option A).

        Consumed by the first coasting substep so a missile that has lost its own
        seeker track re-acquires from the other sources instead of dead-reckoning.
        The measurement carries whatever quality those sources have (e.g. AWACS
        surveillance noise), so re-acquisition is imperfect by design.
        """
        self._datalink_meas_used = False
        self._pending_datalink_meas = None
        if target_id is None or sim is None or owner_position is None:
            return
        try:
            group_radars = self.get_default_group_radars(sim)
            if not group_radars:
                return
            fused = self._get_fused_measurements(
                owner_position, group_radars, self.cached_detections or []
            )
            candidates = [
                self._anonymous_cluster(cluster)
                for cluster in fused or []
                if not cluster.get("is_deception", False)
            ]
            if candidates:
                self._pending_datalink_meas = min(
                    candidates,
                    key=lambda cluster: self._guidance_distance_m(cluster),
                )
        except Exception:
            self._pending_datalink_meas = None

    def _get_fused_measurements(
        self, owner_position, group_radars, own_detections, cooperative=True
    ):
        if self.data_link is not None:
            return self.data_link.get_fused_clusters(
                own_radar=self,
                group_radars=group_radars,
                own_position=owner_position,
                clusterer=self.clusterer,
                cooperative=cooperative,
            )
        else:
            return self.clusterer.cluster(own_detections)

    def get_default_group_radars(self, sim):
        """Friendly radars that support this missile's mid-course this tick.

        A full-datalink (next-gen) missile is supported by the entire friendly
        network. A classic Fox-3 is supported ONLY by its launching aircraft, and
        only while that aircraft still holds the target lock and the missile is
        inside its radar FOV/range (the datalink cone). If the shooter dies, drops
        the lock, or turns cold so the missile leaves its cone, mid-course support
        is cut and the missile coasts on its own seeker (once active) or its last
        estimate. Datalink dropout (from ``update_group_radars``) still applies.
        """
        full = DataLink.update_group_radars(sim, owner=self.owner)
        if getattr(self.owner, "full_datalink", False):
            return full

        source = getattr(self.owner, "source", None)
        if source is None:
            return []
        src_id = getattr(source, "id", None)
        supported = [
            (r, p) for (r, p) in full if getattr(getattr(r, "owner", None), "id", None) == src_id
        ]
        if not supported or not self._source_supports_shot(source, sim):
            return []
        return supported

    def _source_supports_shot(self, source, sim) -> bool:
        target_id = getattr(self.owner, "designated_target_id", None)
        if target_id is None:
            tp = getattr(self, "target_provider", None)
            target_id = getattr(tp, "current_target_id", None) if tp is not None else None
        if target_id is None:
            return False

        # The launching aircraft must still hold the target lock.
        sensor = getattr(source, "sensor", None)
        try:
            locked = set(sensor.get_locked_targets() or []) if sensor is not None else set()
        except Exception:
            locked = set()
        supported_contact_ids = {target_id}
        resolver = getattr(sim, "evaluator_contact_ids_for_truth", None)
        if callable(resolver):
            supported_contact_ids.update(resolver(getattr(source, "id", None), target_id))
        if not (supported_contact_ids & locked):
            return False

        # The missile must lie inside the shooter's radar FOV and range (cone).
        src_radar = getattr(source, "radar", None)
        if src_radar is None:
            return False
        try:
            az_off, el_off, dist = _angles_dist(
                source.position,
                float(src_radar.yaw_deg),
                float(src_radar.pitch_deg),
                self.owner.position,
            )
        except Exception:
            return False
        h_half = float(getattr(src_radar, "h_fov_deg", 120.0)) * 0.5
        v_half = float(getattr(src_radar, "v_fov_deg", 60.0)) * 0.5
        max_range = float(getattr(src_radar, "max_range_m", 1e9))
        return abs(az_off) <= h_half and abs(el_off) <= v_half and dist <= max_range

    def _filter_measurements_for_lock(self, fused_measurements, tick_secs):
        """Discard invalid returns and update designated-track loss time."""
        # Filter out suspected deception (ghosts) - missile seekers prioritize real targets
        operational = getattr(self.owner, "weapon_track", None) is not None
        valid_clusters = []
        for cluster in fused_measurements:
            if cluster.get("is_deception", False) or cluster.get("suspect_deception", False):
                continue
            contact_class = most_likely_class(cluster.get("classification_probabilities"))
            if contact_class in {"missile", "countermeasure"}:
                continue
            valid_clusters.append(cluster)

        if not operational:
            valid_clusters = [
                cluster
                for cluster in fused_measurements
                if cluster.get("T") is not None
                and not cluster.get("is_deception", False)
                and not cluster.get("suspect_deception", False)
                and not getattr(cluster.get("T"), "is_missile", False)
                and not getattr(cluster.get("T"), "is_countermeasure", False)
                and not getattr(cluster.get("T"), "is_non_engageable", False)
            ]

        designated_id = getattr(
            getattr(getattr(self.owner, "weapon_track", None), "snapshot", None),
            "track_id",
            None,
        )
        designated_seen = not operational or any(
            cluster.get("preferred_track_id") == designated_id for cluster in valid_clusters
        )
        if designated_seen:
            self._time_since_designated_target_seen = 0.0
        else:
            self._time_since_designated_target_seen += tick_secs
        return valid_clusters

    def _anonymous_cluster(self, cluster):
        anonymous = dict(cluster)
        if anonymous.get("classification_probabilities") is None:
            anonymous["classification_probabilities"] = signature_class_probabilities(
                anonymous.get("T")
            )
        anonymous["T"] = None
        anonymous["engagement_id"] = None
        anonymous["jammer_id"] = None
        if anonymous.get("covariance_cartesian") is None:
            range_m = max(1.0, float(anonymous.get("d", 0.0)))
            angular_sigma = np.radians(resolution_cell_sigma(self.a_res_deg)) * range_m
            range_sigma = resolution_cell_sigma(self.r_res_m)
            horizontal_sigma = max(50.0, angular_sigma, range_sigma)
            anonymous["covariance_cartesian"] = np.diag(
                [horizontal_sigma**2, horizontal_sigma**2, max(50.0, range_sigma) ** 2]
            )
            anonymous["measurement_ref"] = anonymous.get("source_pos")
        return anonymous

    def _guidance_distance_m(self, cluster) -> float:
        guidance_position = (
            self.target_provider.get_guidance_target() if self.target_provider is not None else None
        )
        measured_position = cluster.get("measurement_position")
        if guidance_position is None or measured_position is None:
            return float(cluster.get("d", float("inf")))
        return (
            geodetic_distance_km(
                guidance_position.lat,
                guidance_position.lon,
                guidance_position.alt,
                measured_position.lat,
                measured_position.lon,
                measured_position.alt,
            )
            * 1000.0
        )

    def _update_lock_ctrl(self, tracks):
        # A1: collect engageable target IDs into a de-duplicated, deterministically-sorted list
        seen_ids = set()
        detected_list = []
        track_scores = {}  # target_id -> (confidence, n_obs, range_m)

        operational = getattr(self.owner, "weapon_track", None) is not None
        retarget_policy = getattr(self.owner, "retarget_policy", "locked_override")
        designated_id = getattr(
            getattr(getattr(self.owner, "weapon_track", None), "snapshot", None),
            "track_id",
            None,
        )
        hold_designation = operational and (
            retarget_policy == "track_only"
            or self._time_since_designated_target_seen < self._designated_target_loss_threshold_s
        )

        for track in tracks:
            if not track.engageable or "missile" in track.classification:
                continue
            if operational:
                target_id = track.track_id
                if hold_designation and target_id != designated_id:
                    continue
            else:
                # Non-operational test/oracle seekers use their designated unit ID.
                target_id = self.locked_target or track.track_id
            range_m = (
                float(np.linalg.norm(track.state[:3])) if len(track.state) >= 3 else float("inf")
            )
            if target_id not in seen_ids:
                seen_ids.add(target_id)
                detected_list.append(target_id)
            # Keep best score if same target appears in multiple tracks
            prev = track_scores.get(target_id)
            if prev is None or track.confidence > prev[0]:
                track_scores[target_id] = (
                    track.confidence,
                    len(track.source_ids),
                    range_m,
                )

        detected_list.sort()  # stable deterministic order by ID (A1)

        # Update confirmation counters (A2 fix in base.py ensures insertion order = sorted order)
        self.lock_ctrl.update_locks(detected_list)

        # B1+B2+C: score-based selection with hysteresis; radar owns the decision
        confirmed = self.lock_ctrl.locked_target_ids()
        if confirmed:
            if self.locked_target in confirmed and self._time_since_designated_target_seen == 0.0:
                self.lock_ctrl.force_active(self.locked_target)
                self.locked_target = self.lock_ctrl.get_locked()
                self.mode = self.lock_ctrl.get_mode()
                return

            def _score(target_id):
                conf, n_ob, rng = track_scores.get(target_id, (0.0, 0, float("inf")))
                return conf + 0.1 * n_ob + 500.0 / max(rng, 1.0)

            best_tid = max(confirmed, key=_score)
            current = self.lock_ctrl.get_locked()

            if current is None or current not in confirmed:
                # Active target lost — switch unconditionally
                self.lock_ctrl.force_active(best_tid)
            elif best_tid != current and _score(best_tid) > _score(current) * 1.2:
                # New target is ≥20% better — switch (hysteresis gate)
                self.lock_ctrl.force_active(best_tid)
            # else: keep current target (stickiness)

        self.locked_target = self.lock_ctrl.get_locked()
        self.mode = self.lock_ctrl.get_mode()

    def update_designated_track_substep(
        self,
        tick_secs,
        target,
        owner_position,
        target_position=None,
    ):
        """Refresh the current missile-target track inside a flight substep.

        This is deliberately not a full radar update: it does not run stochastic
        detection, group fusion, datalink, or lock-controller confirmation. It
        only advances seeker pointing and feeds the existing tracker one
        geometry-gated measurement for the already selected target.
        """
        if owner_position is None:
            return False
        if target is None:
            return self._predict_designated_track_substep(tick_secs, owner_position)
        if (
            getattr(target, "is_destroyed", False)
            or getattr(target, "is_missile", False)
            or getattr(target, "is_countermeasure", False)
            or getattr(target, "is_non_engageable", False)
        ):
            return self._predict_designated_track_substep(tick_secs, owner_position)

        if target_position is None:
            target_position = getattr(target, "position", None)
        if target_position is None:
            return self._predict_designated_track_substep(tick_secs, owner_position)

        self.param_policy.update_dynamic_params()
        self._update_beam_steering(tick_secs, 0.0, 0.0)

        try:
            az_rel, el_rel, dist = _angles_dist(
                owner_position,
                float(self.yaw_deg),
                float(self.pitch_deg),
                target_position,
            )
        except Exception:
            return self._predict_designated_track_substep(tick_secs, owner_position)

        if dist > float(self.max_range_m):
            return self._predict_designated_track_substep(tick_secs, owner_position)
        if abs(az_rel) > float(self.h_fov_deg) * 0.5:
            return self._predict_designated_track_substep(tick_secs, owner_position)
        if abs(el_rel) > float(self.v_fov_deg) * 0.5:
            return self._predict_designated_track_substep(tick_secs, owner_position)

        az_abs = float(self.yaw_deg) + float(az_rel)
        el_abs = float(self.pitch_deg) + float(el_rel)
        operational = getattr(self.owner, "weapon_track", None) is not None
        target_id = getattr(target, "id", None)
        cluster = {
            "T": None if operational else target,
            "az": az_abs,
            "el": el_abs,
            "d": dist,
            "dop": _doppler(target, az_abs, el_abs, self.freq_hz),
            "lat": float(target_position.lat),
            "lon": float(target_position.lon),
            "alt": float(target_position.alt),
            "n_obs": 1,
            "is_deception": False,
            "suspect_deception": False,
            "engagement_id": None if operational else target_id,
            "jammer_id": None,
            "classification_probabilities": signature_class_probabilities(target),
            "measurement_position": target_position,
            "measurement_ref": owner_position,
            "covariance_cartesian": np.diag([5.0**2, 5.0**2, 5.0**2]),
            "preferred_track_id": (
                self.owner.weapon_track.snapshot.track_id if operational else None
            ),
        }

        self.cached_detections = [cluster]
        self.cached_clusters = [cluster]
        self.tracker_manager.set_export_reference(owner_position)
        self.cached_tracks = self.tracker_manager.update_tracks(
            [cluster],
            tick_secs,
            owner_position,
        )

        # locked_target is single-namespace per seeker: anonymous track IDs for a
        # WeaponTrack-guided missile, physical unit IDs only for an oracle launch.
        # A weapon-track seeker with no track yet stays unlocked rather than
        # borrowing the truth ID, which downstream consumers would read as a track.
        if operational and self.cached_tracks:
            # The tracker returns every track this seeker holds, not only the one this
            # substep measured, so position 0 is an arbitrary contact once the seeker
            # resolves more than one aircraft. Prefer the weapon's own track; the
            # tracker may still retire and re-create it under a fresh local id, so
            # fall back rather than leave the seeker unlocked.
            preferred_track_id = cluster["preferred_track_id"]
            self.locked_target = next(
                (
                    track.track_id
                    for track in self.cached_tracks
                    if track.track_id == preferred_track_id
                ),
                self.cached_tracks[0].track_id,
            )
        elif not operational and self.locked_target is None and target_id is not None:
            self.locked_target = target_id
        self._time_since_designated_target_seen = 0.0
        self._seeker_meas_streak += 1  # stable own-radar track building
        self._measured_designated_this_tick = True
        return True

    def _predict_designated_track_substep(self, tick_secs, owner_position):
        """Advance seeker tracks through a substep when the seeker has no fresh
        measurement of its own.

        Option A: if a fused datalink (fighter/AWACS) measurement was staged this
        tick and not yet used, feed it here so the tracker is corrected by the
        other sources instead of pure dead-reckoning — this is "lost track ->
        check all sources". It is consumed once per tick (one tracker advance with
        a measurement); subsequent coasting substeps predict-only.
        """
        self._seeker_meas_streak = 0  # the seeker itself did not measure this substep

        datalink_meas = None
        if self._pending_datalink_meas is not None and not self._datalink_meas_used:
            datalink_meas = self._pending_datalink_meas
            self._datalink_meas_used = True

        clusters = [datalink_meas] if datalink_meas is not None else []
        self.cached_detections = list(clusters)
        self.cached_clusters = list(clusters)
        try:
            self.tracker_manager.set_export_reference(owner_position)
            if clusters:
                self.cached_tracks = self.tracker_manager.update_tracks(
                    clusters, tick_secs, owner_position
                )
            else:
                # Don't age the track for substeps that merely coast AFTER a
                # measurement was already taken this tick (seeker or datalink) —
                # otherwise the designated track looks stale by the last substep
                # and guidance falls back to dead-reckoning within the same tick.
                missed_inc = 0.0 if self._measured_designated_this_tick else float(tick_secs)
                self.cached_tracks = self.tracker_manager.update_tracks(
                    [], tick_secs, owner_position, missed_increment=missed_inc
                )
        except Exception:
            self.cached_tracks = []

        if datalink_meas is not None:
            # Re-acquired (coarsely) via the datalink — reset the loss timer.
            self._time_since_designated_target_seen = 0.0
            self._measured_designated_this_tick = True
        else:
            self._time_since_designated_target_seen += float(tick_secs)
        return False

    def begin_designated_tick(self):
        """Reset per-tick designated-track measurement state. Called once at the
        start of the missile's tick, before its flight substeps."""
        self._measured_designated_this_tick = False

    def has_radar_lock(self, target):
        return hasattr(target, "id") and self.locked_target == target.id

    def get_locked_target(self):
        return self.locked_target

    def get_locked_targets(self):
        return {self.locked_target} if self.locked_target is not None else set()

    def get_mode(self):
        return getattr(self, "mode", "search")

    def should_activate_own_radar(self):
        target_pos = None
        if hasattr(self, "target_provider") and self.target_provider:
            target_pos = self.target_provider.get_guidance_target()
        if target_pos is None:
            return False
        missile_pos = self.owner.position if self.owner else None
        if missile_pos is None:
            return False
        distance = (
            geodetic_distance_km(
                missile_pos.lat,
                missile_pos.lon,
                missile_pos.alt,
                target_pos.lat,
                target_pos.lon,
                target_pos.alt,
            )
            * 1000.0
        )
        in_terminal = (
            hasattr(self.owner, "phase_manager")
            and getattr(self.owner.phase_manager, "current_phase", "") == "terminal"
        )
        return distance < self.max_range_m or in_terminal

    def get_tracker_info(self):
        locked_id = self.get_locked_target()
        for track in self.get_tracks():
            if str(track.track_id) == str(locked_id) and track.engageable:
                tracker = self.tracker_manager.tracks.get(track.track_id, None)
                metadata = self.tracker_manager.track_meta.get(track.track_id, {})
                velocity = self._extract_velocity(track.state, tracker)
                position = self._state_to_position(
                    track.state,
                    track.reference_frame
                    if track.reference_frame is not None
                    else self.owner.position,
                )
                return {
                    "tid": track.track_id,
                    "state": track.state,
                    "cov": track.covariance,
                    "tgt": None,
                    "ref": track.reference_frame,
                    "confidence": track.confidence,
                    "n_obs": int((metadata.get("n_obs_hist") or [0])[-1]),
                    "lifetime": int(metadata.get("lifetime", 0)),
                    "update_count": int(metadata.get("update_count", 0)),
                    "tracker": tracker,
                    "velocity": velocity,
                    "position": position,
                    # ECM fields
                    "is_deception": track.suspect_deception,
                    "suspect_deception": track.suspect_deception,
                    "engagement_id": None,
                    "jammer_id": track.emitter_hypothesis_id,
                    "engageable": track.engageable,
                }
        return None

    def _extract_velocity(self, state, tracker=None):
        if tracker is not None and hasattr(tracker, "get_velocity"):
            return tracker.get_velocity()
        # Default assumption: state[3:6] = vx, vy, vz
        if len(state) >= 6:
            return state[3:6]
        return np.zeros(3)

    def _state_to_position(self, state, ref):
        from bvr_marl_core.radar.core.utils import enu_to_geodetic
        from bvr_marl_core.simulator.core.helpers import Position

        enu = state[:3]
        lat = getattr(ref, "lat", getattr(ref, "latitude_deg", None))
        lon = getattr(ref, "lon", getattr(ref, "longitude_deg", None))
        alt = getattr(ref, "alt", getattr(ref, "altitude_m", None))
        lat, lon, alt = enu_to_geodetic(enu, lat, lon, alt)
        return Position(lat, lon, alt)

    def _update_beam_steering(self, tick_secs: float, steer_h: float = 0.0, steer_p: float = 0.0):

        if not getattr(self, "slave_seeker_to_los", True):
            try:
                return super()._update_beam_steering(tick_secs, steer_h, steer_p)
            except Exception:
                return

        tgt_pos = None
        if getattr(self, "target_provider", None):
            tgt_pos = self.target_provider.get_guidance_target()
        if tgt_pos is None:
            info = self.get_tracker_info()
            if info is not None:
                tgt_pos = info.get("position", None)

        if tgt_pos is None:
            try:
                return super()._update_beam_steering(tick_secs, steer_h, steer_p)
            except Exception:
                max_step = self.gimbal_rate_deg_s * tick_secs
                self.yaw_offset_deg += max(
                    -max_step, min(max_step, -getattr(self, "yaw_offset_deg", 0.0))
                )
                self.pitch_offset_deg += max(
                    -max_step, min(max_step, -getattr(self, "pitch_offset_deg", 0.0))
                )
                return

        own_pos = self.owner.position
        ref_yaw = getattr(self.owner, "yaw_deg", 0.0)
        ref_pitch = getattr(self.owner, "pitch_deg", 0.0)

        az_rel, el_rel, _ = _angles_dist(own_pos, ref_yaw, ref_pitch, tgt_pos)
        az_cmd = float(max(-self.gimbal_h_lim_deg, min(self.gimbal_h_lim_deg, az_rel)))
        el_cmd = float(max(-self.gimbal_v_lim_deg, min(self.gimbal_v_lim_deg, el_rel)))

        # Gimbal-Limits
        az_cmd = float(max(-self.gimbal_h_lim_deg, min(self.gimbal_h_lim_deg, az_cmd)))
        el_cmd = float(max(-self.gimbal_v_lim_deg, min(self.gimbal_v_lim_deg, el_cmd)))

        max_step = self.gimbal_rate_deg_s * tick_secs
        cur_az = float(getattr(self, "yaw_offset_deg", 0.0))
        cur_el = float(getattr(self, "pitch_offset_deg", 0.0))
        d_az = max(-max_step, min(max_step, az_cmd - cur_az))
        d_el = max(-max_step, min(max_step, el_cmd - cur_el))

        self.yaw_offset_deg = cur_az + d_az
        self.pitch_offset_deg = cur_el + d_el
