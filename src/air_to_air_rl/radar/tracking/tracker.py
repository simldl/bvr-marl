import math
from collections import deque

from air_to_air_rl.radar.tracking.helpers.measurement_builder import (
    associate_measurements,
    build_measurements_with_ref,
)
from air_to_air_rl.radar.tracking.helpers.noise_and_bias import (
    apply_xy_bias_correction,
    apply_z_bias_correction,
    get_default_noise_parameters,
)
from air_to_air_rl.radar.tracking.helpers.recenter_logic import (
    get_target_state_for_pn,
    maybe_recenter_reference,
)
from air_to_air_rl.simulator.core.helpers import Position

from .helpers.track_manager import (
    build_track_outputs,
    get_or_create_meta,
    prune_tracks,
    spawn_tracker,
)


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

    def __init__(self, assoc_dist: float):
        self.tracks = {}  # tid -> KF
        self.track_refs = {}  # tid -> Position (ENU origin/basis)
        self.track_meta = {}  # tid -> misc
        self.assoc_dist = float(assoc_dist)

        # Optional noise parameters (can be set by caller)
        noise_params = get_default_noise_parameters()
        self.range_resolution_m = noise_params["range_resolution_m"]
        self.angular_resolution_deg = noise_params["angular_resolution_deg"]
        self.export_ref: Position | None = None

    def set_export_reference(self, ref: Position | None):
        """If set, outputs will be transformed into this ENU before returning."""
        self.export_ref = ref

    def update_tracks(
        self, clusters: list, dt: float, default_ref: Position, own_yaw_deg: float | None = None
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
        meas_list = build_measurements_with_ref(clusters, self.track_refs, default_ref)
        associations = associate_measurements(meas_list, default_ref)
        updated_tids = self._update_or_create_tracks(associations, dt, own_yaw_deg=own_yaw_deg)
        # Predict-only + increment missed_updates for tracks with no measurement this step
        for tid, kf in self.tracks.items():
            if tid not in updated_tids:
                kf.predict(dt)
                kf.missed_updates = getattr(kf, "missed_updates", 0) + 1
        prune_tracks(self.tracks, self.track_refs, self.track_meta, timeout=5)
        return build_track_outputs(
            self.tracks, self.track_refs, self.track_meta, clusters, export_ref=self.export_ref
        )

    def _update_or_create_tracks(self, associations, dt: float, own_yaw_deg: float | None = None):
        """Process track associations and update/create tracks. Returns set of updated track IDs."""
        updated_tids = set()
        predicted_tids = set()  # guard against double-predict when multiple radars share a tid
        for tid, c, meas, n_obs, current_ref in associations:
            updated_tids.add(tid)
            if tid in self.tracks:
                kf = self.tracks[tid]
                meta = self._meta(tid)
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
                    meta["lifetime"] += 1
                    meta["last_dt"] = dt

                meta["n_obs_hist"].append(n_obs)
                if len(meta["n_obs_hist"]) > 10:
                    meta["n_obs_hist"] = meta["n_obs_hist"][-10:]
                is_dec = c.get("is_deception", False)
                meta["is_deception"] = is_dec
                meta["engagement_id"] = c.get("engagement_id", None)
                meta["jammer_id"] = c.get("jammer_id", None)
                meta["last_meas"] = meas

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
                    "jammer_id": c.get("jammer_id", None),
                    "obs_ema": float(n_obs),
                    "updates_with_meas": 1,
                    "nis_ema": 3.0,  # chi2 expected value (DOF=3)
                }

        return updated_tids

    def _maybe_recenter_reference(self, tid, new_ref: Position) -> bool:
        """Check if track reference needs recentering and perform transformation."""
        return maybe_recenter_reference(tid, self.track_refs, self.tracks, new_ref)

    def _cluster_to_track_measurement(self, c, ref: Position):
        """Convert cluster to ENU measurement in specified reference frame."""
        from air_to_air_rl.radar.tracking.helpers.measurement_builder import (
            cluster_to_track_measurement,
        )

        return cluster_to_track_measurement(c, ref)

    def _apply_anisotropic_R(self, meta, tracker, meas_enu, cluster):
        """Apply anisotropic measurement noise model."""
        from air_to_air_rl.radar.tracking.helpers.noise_and_bias import apply_anisotropic_R

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
        import numpy as np

        e, n, u = meas_enu[:3]
        return math.degrees(math.atan2(e, n)) % 360.0  # atan2(East, North)

    @staticmethod
    def _angle_diff(a1_deg, a2_deg):
        """Shortest angular difference in degrees, range [-180, 180]."""
        diff = (a1_deg - a2_deg) % 360.0
        if diff > 180.0:
            diff -= 360.0
        return diff
