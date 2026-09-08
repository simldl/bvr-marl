import math
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from bvr_marl_core.domain.information import TrackLifecycle
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.radar.tracking.helpers.measurement_builder import (
    associate_measurements,
    build_measurements_with_ref,
)
from bvr_marl_core.radar.tracking.helpers.noise_and_bias import (
    apply_xy_bias_correction,
    apply_z_bias_correction,
    get_default_noise_parameters,
)
from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
    get_target_state_for_pn,
    maybe_recenter_reference,
    transform_state_between_refs,
    transform_states_between_refs_batch,
)
from bvr_marl_core.radar.tracking.helpers.track_manager import (
    TRACK_DELETION_MISSED_UPDATES,
    build_track_snapshots,
    get_or_create_meta,
    prune_tracks,
    spawn_tracker,
)
from bvr_marl_core.simulator.core.helpers import Position


def _classification_entropy(probabilities) -> float | None:
    if probabilities is None:
        return None
    values = np.asarray(probabilities, dtype=float)
    return float(-np.sum(values * np.log(np.maximum(values, 1e-12))))


def _fuse_classification_belief(previous, current, weight: float) -> tuple[float, ...]:
    """Temper correlated evidence while retaining full weight for new sensors."""
    current_values = np.maximum(np.asarray(current, dtype=float), 1e-6)
    if previous is None:
        fused = current_values
    else:
        fused = np.maximum(np.asarray(previous, dtype=float), 1e-6) * current_values ** max(
            0.0, float(weight)
        )
    return tuple((fused / fused.sum()).tolist())


class TrackerManager:
    """
    Orchestrates multi-target tracking with:
      • Anisotropic measurement covariance (xy >> z)
      • Simple z-bias observer (EWMA)
      • Auto-recenter of each track's ENU reference WITH PROPER ROTATION
      • Export of velocity rotated into the missile's ENU for PN

    State convention:
      Each track's KF state is kept in the ENU of self.track_refs[tid].
      When the reference changes (recenter), we transform:
        x_pos' = R * x_pos + Δ
        v'     = R * v
        P'     = S P S^T, with S = blkdiag(R, R)
    """

    def __init__(
        self,
        assoc_dist: float,
        *,
        max_tentative_tracks: int = 64,
        max_confirmed_tracks: int = 128,
        confirmation_hits: int = 3,
        motion_model: str = "cv",
        coast_extra_updates: int = 0,
    ):
        self.tracks = {}  # tid -> KF
        self.track_refs = {}  # tid -> Position (ENU origin/basis)
        self.track_meta = {}  # tid -> misc
        self.assoc_dist = float(assoc_dist)
        self._next_track_id = 1
        self.max_tentative_tracks = int(max_tentative_tracks)
        self.max_confirmed_tracks = int(max_confirmed_tracks)
        self.confirmation_hits = max(1, int(confirmation_hits))
        # Extra missed-update tolerance before a coasting track is pruned. The
        # imperfect-information benchmark raises this so tracks coast longer and their
        # estimates diverge further from a manoeuvring target (paper coast axis).
        self.coast_extra_updates = max(0, int(coast_extra_updates))
        self.motion_model = str(motion_model).lower()
        if self.motion_model not in {"cv", "imm_cv_ct"}:
            raise ValueError("motion_model must be 'cv' or 'imm_cv_ct'.")
        self._time_s = 0.0

        # Optional noise parameters (can be set by caller)
        noise_params = get_default_noise_parameters()
        self.range_resolution_m = noise_params["range_resolution_m"]
        self.angular_resolution_deg = noise_params["angular_resolution_deg"]
        self.export_ref: Position | None = None

    def set_export_reference(self, ref: Position | None):
        """If set, outputs will be transformed into this ENU before returning."""
        self.export_ref = ref

    def update_tracks(
        self,
        clusters: list,
        dt: float,
        default_ref: Position,
        own_yaw_deg: float | None = None,
        missed_increment: float | None = None,
    ):
        """
        Main tracking update cycle.

        Args:
            clusters: detection clusters
            dt: step [s]
            default_ref: default ENU reference
            own_yaw_deg: OPTIONAL ownship yaw (deg, heading) at this update. If provided,
                         ECCM bearing-invariance uses this instead of any export_ref.
        """
        if any(self._cluster_has_truth_handle(cluster) for cluster in clusters):
            # Transitional test/debug input. Production SensorReport clusters never
            # enter this branch and therefore never receive evaluator identities.
            meas_list = build_measurements_with_ref(clusters, self.track_refs, default_ref)
            associations = associate_measurements(meas_list, default_ref)
        else:
            associations = self._associate_anonymous_clusters(clusters, dt, default_ref)
        current_time_s = self._time_s
        updated_tids = self._update_or_create_tracks(
            associations,
            dt,
            own_yaw_deg=own_yaw_deg,
            current_time_s=current_time_s,
        )
        # Predict-only + increment missed_updates for tracks with no measurement this step.
        # Full radar updates use count semantics (default +1). Pair-local missile
        # seeker substeps pass dt so track aging remains in seconds instead of
        # pruning after a handful of substeps.
        miss_inc = 1.0 if missed_increment is None else max(0.0, float(missed_increment))
        for tid, kf in self.tracks.items():
            if tid not in updated_tids:
                kf.predict(dt)
                kf.missed_updates = kf.missed_updates + miss_inc
                metadata = self.track_meta[tid]
                if metadata.get("updates_with_meas", 0) >= self.confirmation_hits:
                    metadata["lifecycle"] = TrackLifecycle.COASTING
        for metadata in self.track_meta.values():
            metadata["lifetime_s"] = float(metadata.get("lifetime_s", 0.0)) + float(dt)
        prune_tracks(
            self.tracks,
            self.track_refs,
            self.track_meta,
            timeout=TRACK_DELETION_MISSED_UPDATES + self.coast_extra_updates,
        )
        self._merge_duplicate_tracks(default_ref)
        self._enforce_track_caps()
        snapshots = build_track_snapshots(
            self.tracks,
            self.track_refs,
            self.track_meta,
            clusters,
            export_ref=self.export_ref,
            current_time_s=current_time_s,
        )
        self._time_s += float(dt)
        return snapshots

    @staticmethod
    def _cluster_has_truth_handle(cluster) -> bool:
        target = cluster.get("T")
        if isinstance(target, list):
            return any(item is not None for item in target)
        return target is not None

    def _new_track_id(self) -> int:
        while self._next_track_id in self.tracks:
            self._next_track_id += 1
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _cluster_covariance_in_ref(self, cluster, ref: Position) -> np.ndarray:
        covariance = cluster.get("covariance_cartesian")
        if covariance is None:
            range_sigma = max(25.0, float(cluster.get("d", 0.0)) * 0.01)
            return np.eye(3) * range_sigma**2
        value = np.asarray(covariance, dtype=float)
        measurement_ref = cluster.get("measurement_ref")
        if measurement_ref is None:
            return value
        rotation = enu_rotation(
            measurement_ref.lat,
            measurement_ref.lon,
            ref.lat,
            ref.lon,
        )
        return rotation @ value @ rotation.T

    def _associate_anonymous_clusters(self, clusters, dt: float, default_ref: Position):
        """Globally associate anonymous reports by normalized innovation distance."""
        if not clusters:
            return []
        known_lineage = {
            lineage
            for metadata in self.track_meta.values()
            for lineage in metadata.get("report_lineage", ())
        }
        clusters = [
            cluster
            for cluster in clusters
            if not cluster.get("report_lineage")
            or not set(cluster["report_lineage"]).issubset(known_lineage)
        ]
        if not clusters:
            return []
        tids = list(self.tracks)
        if not tids:
            associations = []
            reserved_ids: set[object] = set()
            for cluster in clusters[: self.max_tentative_tracks]:
                preferred_id = cluster.get("preferred_track_id")
                if preferred_id is not None and preferred_id not in reserved_ids:
                    track_id = preferred_id
                else:
                    track_id = self._new_track_id()
                    while track_id in reserved_ids:
                        track_id = self._new_track_id()
                reserved_ids.add(track_id)
                associations.append(
                    (
                        track_id,
                        cluster,
                        self._cluster_to_track_measurement(cluster, default_ref),
                        int(cluster.get("n_obs", 1)),
                        default_ref.copy(),
                    )
                )
            return associations

        # Predict every track and transform its predicted position/covariance
        # into one shared ENU frame (default_ref) in a single O(T) batch, and
        # convert each cluster into that same frame once (O(C)). Gating is then
        # a broadcast over the (T, C) grid instead of a Python double loop that
        # redid a geodetic transform for (almost) every pair. NIS,
        # ||innovation||^2 and trace(S) are invariant under the orthogonal
        # ENU->ENU rotation, so the common-frame gate is numerically equivalent
        # to the per-track-frame computation this replaces.
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * float(dt)
        predicted_states: list[np.ndarray] = []
        predicted_covariances: list[np.ndarray] = []
        track_from_refs: list[Position] = []
        for tid in tids:
            kf = self.tracks[tid]
            state = np.asarray(kf.get_state(), dtype=float)
            covariance = np.asarray(kf.get_covariance(), dtype=float)
            predicted_states.append(transition @ state)
            predicted_covariances.append(transition @ covariance @ transition.T)
            track_from_refs.append(self.track_refs[tid])

        predicted_states_common, predicted_covariances_common = transform_states_between_refs_batch(
            predicted_states, predicted_covariances, track_from_refs, default_ref
        )
        predicted_positions = predicted_states_common[:, :3]  # (T, 3)
        predicted_position_covariances = predicted_covariances_common[:, :3, :3]  # (T, 3, 3)

        # Per-track ENU(ref_t) -> ENU(default_ref) rotation, matching the batch above.
        track_rotations = np.stack(
            [
                enu_rotation(ref.lat, ref.lon, default_ref.lat, default_ref.lon)
                for ref in track_from_refs
            ]
        )  # (T, 3, 3)

        cluster_measurements = np.stack(
            [self._cluster_to_track_measurement(cluster, default_ref) for cluster in clusters]
        )  # (C, 3), shared frame
        cluster_covariances = np.stack(
            [self._cluster_covariance_in_ref(cluster, default_ref) for cluster in clusters]
        )  # (C, 3, 3), shared frame

        innovations = (
            cluster_measurements[None, :, :] - predicted_positions[:, None, :]
        )  # (T, C, 3)
        innovation_covariances = (
            predicted_position_covariances[:, None, :, :] + cluster_covariances[None, :, :, :]
        )  # (T, C, 3, 3)

        # A cluster covariance with covariance_cartesian but no measurement_ref is
        # used unrotated by _cluster_covariance_in_ref, i.e. as if it already lived
        # in each track's own ENU frame. Its faithful image in the shared frame is
        # therefore R_t @ value @ R_t^T per track, not the single shared matrix used
        # above. These columns are a minority, so correct just them (preserves the
        # legacy behaviour exactly rather than silently reframing it).
        for column, cluster in enumerate(clusters):
            if (
                cluster.get("covariance_cartesian") is not None
                and cluster.get("measurement_ref") is None
            ):
                value = np.asarray(cluster["covariance_cartesian"], dtype=float)
                per_track = track_rotations @ value @ track_rotations.transpose(0, 2, 1)  # (T,3,3)
                innovation_covariances[:, column, :, :] = predicted_position_covariances + per_track
        innovation_sq = np.einsum("tck,tck->tc", innovations, innovations)  # (T, C)
        covariance_traces = (
            innovation_covariances[..., 0, 0]
            + innovation_covariances[..., 1, 1]
            + innovation_covariances[..., 2, 2]
        )  # (T, C)

        # chi-square(3) 99.9% gate; preferred (track-id) matches are gated loosely.
        gates = np.full((len(tids), len(clusters)), 16.266, dtype=float)
        row_of_tid = {tid: row for row, tid in enumerate(tids)}
        for column, cluster in enumerate(clusters):
            preferred = cluster.get("preferred_track_id")
            if preferred is not None and preferred in row_of_tid:
                gates[row_of_tid[preferred], column] = 100.0

        # For PSD S, lambda_max(S) <= trace(S), hence NIS >= ||innovation||^2 /
        # trace(S). Reject pairs whose cheapest possible NIS already misses the
        # gate, avoiding the 3x3 solve for obviously distant tracks (identical
        # decision to the scalar bound, just evaluated over the whole grid).
        candidates = innovation_sq <= gates * covariance_traces  # (T, C)

        costs = np.full((len(tids), len(clusters)), np.inf, dtype=float)
        for row, column in zip(*np.nonzero(candidates), strict=True):
            row = int(row)
            column = int(column)
            innovation = innovations[row, column]
            innovation_covariance = innovation_covariances[row, column]
            try:
                nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
            except np.linalg.LinAlgError:
                nis = float("inf")
            if nis <= gates[row, column]:
                costs[row, column] = nis

        finite_costs = np.where(np.isfinite(costs), costs, 1e12)
        rows, columns = self._linear_assignment(finite_costs)
        assigned_columns = set()
        associations = []
        lineages_by_track: dict[int, set[tuple[object, int]]] = {}
        for row, column in zip(rows, columns, strict=True):
            if not np.isfinite(costs[row, column]):
                continue
            tid = tids[row]
            cluster = clusters[column]
            associations.append(
                (
                    tid,
                    cluster,
                    self._cluster_to_track_measurement(cluster, self.track_refs[tid]),
                    int(cluster.get("n_obs", 1)),
                    self.track_refs[tid].copy(),
                )
            )
            assigned_columns.add(column)
            lineages_by_track.setdefault(tid, set()).update(cluster.get("report_lineage", ()))

        # Hungarian selects at most one cluster per track. Independent sensors may
        # legitimately supply several statistically compatible measurements in
        # one tick, so attach remaining gated clusters to their best existing
        # track before considering track creation. Report lineage prevents a
        # relayed copy of the same observation from being counted twice.
        for column, cluster in enumerate(clusters):
            if column in assigned_columns:
                continue
            finite_rows = np.flatnonzero(np.isfinite(costs[:, column]))
            if finite_rows.size == 0:
                continue
            report_lineage = set(cluster.get("report_lineage", ()))
            ranked_rows = sorted(finite_rows, key=lambda row: (costs[row, column], tids[row]))
            for row in ranked_rows:
                tid = tids[int(row)]
                if report_lineage & lineages_by_track.get(tid, set()):
                    continue
                associations.append(
                    (
                        tid,
                        cluster,
                        self._cluster_to_track_measurement(cluster, self.track_refs[tid]),
                        int(cluster.get("n_obs", 1)),
                        self.track_refs[tid].copy(),
                    )
                )
                assigned_columns.add(column)
                lineages_by_track.setdefault(tid, set()).update(report_lineage)
                break
        tentative_count = sum(
            metadata.get("updates_with_meas", 0) < self.confirmation_hits
            for metadata in self.track_meta.values()
        )
        new_track_budget = max(0, self.max_tentative_tracks - tentative_count)
        created = 0
        for column, cluster in enumerate(clusters):
            if column in assigned_columns:
                continue
            if created >= new_track_budget or len(self.tracks) + created >= (
                self.max_tentative_tracks + self.max_confirmed_tracks
            ):
                continue
            preferred_id = cluster.get("preferred_track_id")
            track_id = (
                preferred_id
                if preferred_id is not None
                and preferred_id not in self.tracks
                and all(item[0] != preferred_id for item in associations)
                else self._new_track_id()
            )
            associations.append(
                (
                    track_id,
                    cluster,
                    cluster_measurements[column].copy(),
                    int(cluster.get("n_obs", 1)),
                    default_ref.copy(),
                )
            )
            created += 1
        return associations

    @staticmethod
    def _linear_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictable global minimum-cost association via Hungarian assignment."""
        if min(costs.shape, default=0) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        return tuple(np.asarray(axis, dtype=int) for axis in linear_sum_assignment(costs))

    def _update_or_create_tracks(
        self,
        associations,
        dt: float,
        own_yaw_deg: float | None = None,
        current_time_s: float = 0.0,
    ):
        """Process track associations and update/create tracks. Returns set of updated track IDs."""
        updated_tids = set()
        predicted_tids = set()  # guard against double-predict when multiple radars share a tid
        for tid, c, meas, n_obs, current_ref in associations:
            updated_tids.add(tid)
            # A genuinely ranged radar return (measured range, not a range-denied
            # bearing nor a bearing-only triangulation). Only such a return can
            # corroborate that a track is a real body rather than a spurious
            # intersection of two passive (IRST) bearings.
            is_ranged_meas = not c.get("range_denied", False) and not c.get("triangulated", False)
            if tid in self.tracks:
                kf = self.tracks[tid]
                meta = self._meta(tid)
                known_classification_lineage = set(meta.get("classification_lineage", ()))
                was_coasting = kf.missed_updates > 0
                is_first = tid not in predicted_tids
                if is_first:
                    meta["update_count"] += 1

                recentered = self._maybe_recenter_reference(tid, current_ref)
                meta["last_recentered"] = bool(recentered)
                if recentered:
                    meas = self._cluster_to_track_measurement(c, self.track_refs[tid])

                # Predict once per tick regardless of how many sources report this tid
                if is_first:
                    kf.predict(dt)
                    predicted_tids.add(tid)

                c["track_ref"] = self.track_refs[tid]
                self._apply_anisotropic_R(meta, kf, meas, c)
                meas_adj = self._apply_z_bias(meta, kf, meas)
                meas_adj = apply_xy_bias_correction(meta, kf, meas_adj, alpha=0.01)
                kf.update(meas_adj)
                kf.missed_updates = 0

                nis = kf.get_last_update_stats().get("nis", 3.0)
                alpha_nis = 0.2
                meta["nis_ema"] = alpha_nis * nis + (1.0 - alpha_nis) * meta.get("nis_ema", 3.0)

                alpha_obs = 0.2
                meta["obs_ema"] = alpha_obs * n_obs + (1.0 - alpha_obs) * meta.get(
                    "obs_ema", float(n_obs)
                )

                if is_first:
                    meta["updates_with_meas"] = meta.get("updates_with_meas", 0) + 1
                    meta["last_measurement_time_s"] = float(current_time_s)
                    meta["lifetime"] += 1
                    meta["last_dt"] = dt
                    if meta["updates_with_meas"] >= self.confirmation_hits:
                        meta["lifecycle"] = (
                            TrackLifecycle.REACQUIRED if was_coasting else TrackLifecycle.CONFIRMED
                        )

                meta["n_obs_hist"].append(n_obs)
                if len(meta["n_obs_hist"]) > 10:
                    meta["n_obs_hist"] = meta["n_obs_hist"][-10:]
                is_dec = c.get("is_deception", False)
                meta["is_deception"] = is_dec
                meta["engagement_id"] = c.get("engagement_id", None)
                meta["jammer_id"] = c.get("jammer_id", None)
                if is_ranged_meas:
                    # Range corroboration latches on: once a real ranged return has
                    # touched this track it is a genuine body for good, even if it
                    # later coasts on bearing-only updates.
                    meta["range_corroborated"] = True
                meta["last_meas"] = meas
                if is_first:
                    meta["report_ids"] = tuple(c.get("report_ids", ()))
                    meta["source_ids"] = tuple(c.get("source_ids", ()))
                    meta["report_lineage"] = tuple(
                        sorted(
                            set(meta.get("report_lineage", ())) | set(c.get("report_lineage", ())),
                            key=lambda item: (str(item[0]), item[1]),
                        )
                    )
                else:
                    meta["report_ids"] = tuple(
                        sorted(set(meta.get("report_ids", ())) | set(c.get("report_ids", ())))
                    )
                    meta["source_ids"] = tuple(
                        sorted(
                            set(meta.get("source_ids", ())) | set(c.get("source_ids", ())),
                            key=str,
                        )
                    )
                    meta["report_lineage"] = tuple(
                        sorted(
                            set(meta.get("report_lineage", ())) | set(c.get("report_lineage", ())),
                            key=lambda item: (str(item[0]), item[1]),
                        )
                    )
                if c.get("classification_probabilities") is not None:
                    incoming_lineage = set(c.get("report_lineage", ()))
                    new_lineage = incoming_lineage - known_classification_lineage
                    if new_lineage:
                        known_sources = {source_id for source_id, _ in known_classification_lineage}
                        new_sources = {source_id for source_id, _ in new_lineage}
                        independent = len(new_sources - known_sources)
                        correlated = len(new_lineage) - independent
                        evidence_weight = float(independent) + 0.25 * float(correlated)
                        belief = _fuse_classification_belief(
                            meta.get("classification_probabilities"),
                            c["classification_probabilities"],
                            evidence_weight,
                        )
                        meta["classification_probabilities"] = belief
                        meta["classification_entropy_nats"] = _classification_entropy(belief)
                        meta["effective_classification_evidence"] = (
                            float(meta.get("effective_classification_evidence", 0.0))
                            + evidence_weight
                        )
                        meta["classification_lineage"] = tuple(
                            sorted(
                                known_classification_lineage | incoming_lineage,
                                key=lambda item: (str(item[0]), item[1]),
                            )
                        )
                # ECCM bearing-invariance: ghost detections from jammers don't rotate with ownship
                if is_dec:
                    if own_yaw_deg is not None:
                        bearing_deg = self._compute_bearing_from_meas(meas_adj)

                        yaw_hist = meta.setdefault("yaw_hist", deque(maxlen=5))
                        brg_hist = meta.setdefault("brg_hist", deque(maxlen=5))
                        yaw_hist.append(float(own_yaw_deg))
                        brg_hist.append(float(bearing_deg))

                        if len(yaw_hist) >= 3:
                            dpsi = sum(
                                abs(self._angle_diff(yaw_hist[i + 1], yaw_hist[i]))
                                for i in range(len(yaw_hist) - 1)
                            )
                            dtheta = sum(
                                abs(self._angle_diff(brg_hist[i + 1], brg_hist[i]))
                                for i in range(len(brg_hist) - 1)
                            )
                            rho = dtheta / (dpsi + 1e-3)

                            meta["suspect_deception"] = bool(dpsi > 20.0 and rho < 0.15)
                        else:
                            meta["suspect_deception"] = False
                    else:
                        meta["suspect_deception"] = False
                else:
                    meta["suspect_deception"] = False

            else:
                kf = self._spawn_tracker(c, meas, dt, current_ref)
                kf.missed_updates = 0
                self.tracks[tid] = kf
                self.track_refs[tid] = current_ref.copy()
                self.track_meta[tid] = {
                    "n_obs_hist": [n_obs],
                    "lifetime": 1,
                    "update_count": 1,
                    "last_meas": meas,
                    "last_dt": dt,
                    "z_bias": 0.0,
                    "last_recentered": False,
                    "is_deception": c.get("is_deception", False),
                    "suspect_deception": False,
                    "engagement_id": c.get("engagement_id", None),
                    "report_ids": tuple(c.get("report_ids", ())),
                    "source_ids": tuple(c.get("source_ids", ())),
                    "report_lineage": tuple(c.get("report_lineage", ())),
                    "classification_probabilities": c.get("classification_probabilities"),
                    "classification_entropy_nats": _classification_entropy(
                        c.get("classification_probabilities")
                    ),
                    "effective_classification_evidence": float(
                        c.get(
                            "effective_classification_evidence",
                            len(set(c.get("source_ids", ()))),
                        )
                    ),
                    "classification_lineage": tuple(c.get("report_lineage", ())),
                    "jammer_id": c.get("jammer_id", None),
                    "obs_ema": float(n_obs),
                    "updates_with_meas": 1,
                    "range_corroborated": is_ranged_meas,
                    "last_measurement_time_s": float(current_time_s),
                    "lifetime_s": 0.0,
                    "lifecycle": (
                        TrackLifecycle.CONFIRMED
                        if self.confirmation_hits == 1
                        else TrackLifecycle.TENTATIVE
                    ),
                    "nis_ema": 3.0,  # chi2 expected value (DOF=3)
                }

        return updated_tids

    def _merge_duplicate_tracks(self, reference: Position) -> None:
        """Remove near-identical tracks whose covariance ellipsoids strongly overlap."""
        tids = list(self.tracks)
        if len(tids) < 2:
            return
        states: dict[int, np.ndarray] = {}
        covariances: dict[int, np.ndarray] = {}
        for tid in tids:
            state, covariance = transform_state_between_refs(
                self.tracks[tid].get_state(),
                self.tracks[tid].get_covariance(),
                self.track_refs[tid],
                reference,
            )
            states[tid], covariances[tid] = state, covariance

        removed: set[int] = set()
        for index, left in enumerate(tids):
            if left in removed:
                continue
            for right in tids[index + 1 :]:
                if right in removed:
                    continue
                velocity_delta = float(np.linalg.norm(states[left][3:] - states[right][3:]))
                if velocity_delta > 100.0:
                    continue
                delta = states[left][:3] - states[right][:3]
                overlap = covariances[left][:3, :3] + covariances[right][:3, :3]
                # For PSD ``overlap``, lambda_max <= trace. Therefore a pair
                # outside this Euclidean bound cannot pass the Mahalanobis gate.
                # This inexpensive rejection avoids a matrix inverse for the vast
                # majority of unrelated tracks in large scenarios.
                if float(delta @ delta) > 0.05 * max(float(np.trace(overlap)), 0.0):
                    continue
                try:
                    distance = float(delta @ np.linalg.solve(overlap, delta))
                except np.linalg.LinAlgError:
                    distance = float(delta @ np.linalg.pinv(overlap) @ delta)
                if distance > 0.05:
                    continue
                left_updates = self.track_meta[left].get("updates_with_meas", 0)
                right_updates = self.track_meta[right].get("updates_with_meas", 0)
                keep, discard = (left, right) if left_updates >= right_updates else (right, left)
                keep_meta = self.track_meta[keep]
                discard_meta = self.track_meta[discard]
                keep_class_lineage = set(keep_meta.get("classification_lineage", ()))
                discard_class_lineage = set(discard_meta.get("classification_lineage", ()))
                novel_lineage = discard_class_lineage - keep_class_lineage
                discard_belief = discard_meta.get("classification_probabilities")
                if novel_lineage and discard_belief is not None:
                    known_sources = {source_id for source_id, _ in keep_class_lineage}
                    novel_sources = {source_id for source_id, _ in novel_lineage}
                    independent = len(novel_sources - known_sources)
                    correlated = len(novel_lineage) - independent
                    evidence_weight = float(independent) + 0.25 * float(correlated)
                    belief = _fuse_classification_belief(
                        keep_meta.get("classification_probabilities"),
                        discard_belief,
                        evidence_weight,
                    )
                    keep_meta["classification_probabilities"] = belief
                    keep_meta["classification_entropy_nats"] = _classification_entropy(belief)
                    keep_meta["effective_classification_evidence"] = (
                        float(keep_meta.get("effective_classification_evidence", 0.0))
                        + evidence_weight
                    )
                keep_meta["classification_lineage"] = tuple(
                    sorted(
                        keep_class_lineage | discard_class_lineage,
                        key=lambda item: (str(item[0]), item[1]),
                    )
                )
                keep_meta["report_ids"] = tuple(
                    sorted(
                        set(keep_meta.get("report_ids", ()))
                        | set(discard_meta.get("report_ids", ()))
                    )
                )
                keep_meta["source_ids"] = tuple(
                    sorted(
                        set(keep_meta.get("source_ids", ()))
                        | set(discard_meta.get("source_ids", ())),
                        key=str,
                    )
                )
                keep_meta["report_lineage"] = tuple(
                    sorted(
                        set(keep_meta.get("report_lineage", ()))
                        | set(discard_meta.get("report_lineage", ())),
                        key=lambda item: (str(item[0]), item[1]),
                    )
                )
                # Merging two estimates of one body: range corroboration from either
                # side carries over, so a real track absorbing a bearing-only ghost
                # (or vice versa) stays engageable.
                keep_meta["range_corroborated"] = bool(
                    keep_meta.get("range_corroborated", False)
                    or discard_meta.get("range_corroborated", False)
                )
                removed.add(discard)
        for tid in removed:
            self.tracks.pop(tid, None)
            self.track_refs.pop(tid, None)
            self.track_meta.pop(tid, None)

    def _enforce_track_caps(self) -> None:
        """Bound tentative and confirmed/coasting track populations deterministically."""
        tentative = [
            tid
            for tid, metadata in self.track_meta.items()
            if metadata.get("updates_with_meas", 0) < self.confirmation_hits
        ]
        confirmed = [tid for tid in self.track_meta if tid not in tentative]

        def overflow(track_ids: list[int], cap: int) -> list[int]:
            ranked = sorted(
                track_ids,
                key=lambda tid: (
                    self.tracks[tid].missed_updates,
                    -self.track_meta[tid].get("updates_with_meas", 0),
                    str(tid),
                ),
            )
            return ranked[max(0, cap) :]

        for tid in overflow(tentative, self.max_tentative_tracks) + overflow(
            confirmed, self.max_confirmed_tracks
        ):
            self.track_meta[tid]["lifecycle"] = TrackLifecycle.DELETED
            self.tracks.pop(tid, None)
            self.track_refs.pop(tid, None)
            self.track_meta.pop(tid, None)

    def _maybe_recenter_reference(self, tid, new_ref: Position) -> bool:
        """Check if track reference needs recentering and perform transformation."""
        return maybe_recenter_reference(tid, self.track_refs, self.tracks, new_ref)

    def _cluster_to_track_measurement(self, c, ref: Position):
        """Convert cluster to ENU measurement in specified reference frame."""
        from bvr_marl_core.radar.tracking.helpers.measurement_builder import (
            cluster_to_track_measurement,
        )

        return cluster_to_track_measurement(c, ref)

    def _apply_anisotropic_R(self, meta, tracker, meas_enu, cluster):
        """Apply anisotropic measurement noise model."""
        from bvr_marl_core.radar.tracking.helpers.noise_and_bias import apply_anisotropic_R

        apply_anisotropic_R(self, meta, tracker, meas_enu, cluster)

    def _apply_z_bias(self, meta, tracker, meas_enu):
        """Apply z-bias correction using EWMA."""
        return apply_z_bias_correction(meta, tracker, meas_enu)

    def _spawn_tracker(self, c, meas, dt: float, ref_pos: Position):
        """Create a new tracker for this measurement."""
        return spawn_tracker(self, c, meas, dt, ref_pos)

    def _meta(self, tid):
        """Get or create metadata for track ID."""
        return get_or_create_meta(self.track_meta, tid)

    def get_target_state_for_pn(self, tid: int, missile_ref: Position):
        return get_target_state_for_pn(tid, self.track_refs, self.tracks, missile_ref)

    def _compute_bearing_from_meas(self, meas_enu):
        """Compute bearing (0-360°, 0=North) from ENU position vector."""

        e, n, u = meas_enu[:3]
        return math.degrees(math.atan2(e, n)) % 360.0  # atan2(East, North)

    @staticmethod
    def _angle_diff(a1_deg, a2_deg):
        """Shortest angular difference in degrees, range [-180, 180]."""
        diff = (a1_deg - a2_deg) % 360.0
        if diff > 180.0:
            diff -= 360.0
        return diff
