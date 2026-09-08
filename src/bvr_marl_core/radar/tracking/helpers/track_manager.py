import math

import numpy as np

from bvr_marl_core.domain.information import FrameReference, TrackLifecycle, TrackSnapshot
from bvr_marl_core.radar.tracking.filter.base_filter import BaseKFFilter
from bvr_marl_core.radar.tracking.filter.filters import (
    ConstantVelocityKFFilter,
    create_imm_cv_ct_filter,
)
from bvr_marl_core.radar.tracking.helpers.recenter_logic import (
    transform_states_between_refs_batch,
)
from bvr_marl_core.simulator.core.helpers import Position

N_MAX_RADARS = 3
# Detection-term reference: a single sensor that steadily holds the target (EMA
# of per-update hit count ≈ 1) fully satisfies the obs_score factor, so track
# confidence can reach ~1.0 from one radar instead of saturating at ~1/N_MAX.
OBS_REF = 1.0
CONFIDENCE_MATURITY_UPDATES = 5
CONFIDENCE_POSITION_FLOOR_M = 500.0
CONFIDENCE_ANGULAR_REFERENCE_RAD = 0.03
CONFIDENCE_ACCURACY_FLOOR = 0.55
# Recency decay constant, in missed updates. `recency` used to be
# `0.5 ** missed_updates` -- a halving for every single missed update. That treats a
# SCANNING radar's normal revisit gap as track degradation: as measured, the
# tracker sits at 0-4 missed updates with a MEDIAN of 1, so the median track had its
# confidence halved while obs_score, maturity and consistency were all 1.0 and
# accuracy was 0.78. Confidence came out bimodal (median 0.41, p75 0.80) and spent
# most of its time under every threshold consuming it, for a track that was in every
# other respect healthy.
#
# Decaying with a time constant instead keeps a normal scan gap survivable while a
# genuinely stale track still fades: at tau = 4 one missed update costs 22% rather
# than 50%, and the track is deleted at TRACK_DELETION_MISSED_UPDATES (5) anyway, so
# the decay only has to span that window.
CONFIDENCE_RECENCY_TAU_UPDATES = 4.0
# A track is deleted once it accumulates this many missed updates. Both update
# paths advance the counter in seconds at the default configuration (the network
# picture adds ``tick_secs``; a radar adds ``tick_secs / dwell_duration_s`` with
# a 1 s default dwell), so this is also the coast lifetime in seconds. Deletion
# happens at the threshold, so the oldest coasted state the tracker ever exports
# is ``TRACK_DELETION_MISSED_UPDATES - 1`` missed updates old.
TRACK_DELETION_MISSED_UPDATES = 5


def spawn_tracker(self, c, meas: np.ndarray, dt: float, ref_pos: Position) -> BaseKFFilter:
    """
    Spawn a CV KF and seed its initial covariance from the *first* measurement type.
    """
    if c.get("range_denied", False):
        # Bearing-only jammer strobe: born very uncertain in position (range unknown),
        # so it is not seeded from the ground-truth T it carries for identity.
        sx, sy, sz = 60_000.0, 60_000.0, 60_000.0
    elif c.get("triangulated", False):
        # Cross-radar range recovery: usable but coarser than a clean skin/datalink track.
        sx, sy, sz = 1500.0, 1500.0, 1500.0
    elif c.get("T") is not None and hasattr(c["T"], "position"):
        sx, sy, sz = 20.0, 20.0, 20.0
    else:
        range_res_m = float(getattr(self, "range_resolution_m", 150.0))
        ang_res_deg = float(getattr(self, "angular_resolution_deg", 2.0))
        ang_res_rad = math.radians(ang_res_deg)
        # Prefer the sensor-measured slant range (cluster['d']) over the ENU-derived
        # distance: meas may be in a transformed reference frame (datalink, recenter).
        sensor_range = c.get("d", None)
        if sensor_range is not None and float(sensor_range) > 0.0:
            rng_xy = float(sensor_range)
        else:
            rng_xy = float(np.hypot(meas[0], meas[1])) + 1e-6
        sx = max(0.5 * range_res_m, 0.5 * ang_res_rad * rng_xy)
        sy = sx
        sz = max(10.0, 0.25 * range_res_m)

    supplied_covariance = c.get("covariance_cartesian")
    if supplied_covariance is not None:
        R0 = np.asarray(supplied_covariance, dtype=float)
        measurement_ref = c.get("measurement_ref")
        if measurement_ref is not None:
            from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation

            rotation = enu_rotation(
                measurement_ref.lat,
                measurement_ref.lon,
                ref_pos.lat,
                ref_pos.lon,
            )
            R0 = rotation @ R0 @ rotation.T
        sx, sy, sz = np.sqrt(np.maximum(np.diag(R0), 0.0))
    else:
        R0 = np.diag([sx * sx, sy * sy, sz * sz])
    P0 = np.zeros((6, 6), dtype=float)
    P0[:3, :3] = R0
    diagonal = np.diag_indices(3)
    P0[:3, :3][diagonal] = np.maximum(P0[:3, :3][diagonal], 25.0)
    P0[3:, 3:] = np.diag([90000.0, 90000.0, 90000.0])  # (300 m/s)^2 on velocity

    if self.motion_model == "imm_cv_ct":
        kf = create_imm_cv_ct_filter(meas, dt=dt)
    else:
        kf = ConstantVelocityKFFilter(meas, dt, process_noise_std=50.0, measurement_noise_var=1.0)
    kf.set_state(np.concatenate([meas.astype(float), np.zeros(3, dtype=float)]), P0)
    kf.set_measurement_covariance(R0)
    return kf


def prune_tracks(
    tracks: dict,
    track_refs: dict,
    track_meta: dict,
    timeout: int = TRACK_DELETION_MISSED_UPDATES,
):
    """
    Remove tracks that haven't been updated for too long.

    Args:
        tracks: Dictionary of track filters
        track_refs: Dictionary of track references
        track_meta: Dictionary of track metadata
        timeout: Maximum number of missed updates before pruning
    """
    remove = []
    for tid, kf in tracks.items():
        if kf.missed_updates >= timeout:
            remove.append(tid)

    for tid in remove:
        tracks.pop(tid, None)
        track_refs.pop(tid, None)
        track_meta.pop(tid, None)


def build_track_snapshots(
    tracks: dict,
    track_refs: dict,
    track_meta: dict,
    clusters: list,
    export_ref: Position | None = None,
    current_time_s: float = 0.0,
) -> list[TrackSnapshot]:
    """
    Build immutable authoritative snapshots for all tracks.
    If export_ref is provided, states/covariances are rotated into that ENU.
    """
    tids = list(tracks.keys())
    states = [tracks[tid].get_state() for tid in tids]
    covs = [tracks[tid].get_covariance() for tid in tids]
    refs = [track_refs.get(tid) for tid in tids]

    # Batch the export-frame transform across every track that has a reference, in
    # one vectorized call instead of N per-track matmul triples. Tracks without a
    # ref (or when no export_ref is set) keep their raw filter-frame state. Equals
    # the per-track scalar transform to machine epsilon.
    if export_ref is not None and tids:
        batch_idx = [i for i, r in enumerate(refs) if r is not None]
        if batch_idx:
            X_to, P_to = transform_states_between_refs_batch(
                [states[i] for i in batch_idx],
                [covs[i] for i in batch_idx],
                [refs[i] for i in batch_idx],
                export_ref,
            )
            for k, i in enumerate(batch_idx):
                states[i] = X_to[k]
                covs[i] = P_to[k]

    outs = []
    for i, tid in enumerate(tids):
        kf = tracks[tid]
        meta = track_meta[tid]
        x6 = states[i]
        P6 = covs[i]
        ref = refs[i]
        out_ref = export_ref if (export_ref is not None and ref is not None) else ref

        # Gather context
        lifetime = meta.get("lifetime", 1)
        confidence = calculate_confidence(meta, kf)

        suspect_deception = meta.get("suspect_deception", False)
        is_deception = meta.get("is_deception", False)
        class_probabilities = meta.get("classification_probabilities")

        lifecycle = meta.get("lifecycle", TrackLifecycle.TENTATIVE)
        # A track is a valid weapons target only once a genuine ranged return has
        # corroborated it. Bearing-only triangulations (IRST, and jammer strobes
        # before burn-through) that never receive a real range are treated as
        # non-engageable: two passive sensors watching several targets cross their
        # bearings at spurious "ghost" points, and those must not become shootable
        # duplicates. Defaults True so any track without the flag (legacy inputs,
        # ranged radar tracks) keeps the prior behaviour.
        range_corroborated = meta.get("range_corroborated", True)
        engageable = (
            lifecycle
            in {
                TrackLifecycle.CONFIRMED,
                TrackLifecycle.REACQUIRED,
            }
            and (not suspect_deception)
            and (not is_deception)
            and range_corroborated
        )

        frame = (
            FrameReference(out_ref.lat, out_ref.lon, out_ref.alt) if out_ref is not None else None
        )
        outs.append(
            TrackSnapshot(
                track_id=tid,
                state_time_s=float(current_time_s),
                # The immutable boundary performs the one required conversion
                # and validation; pre-converting here doubled Python iteration.
                state=x6,
                covariance=P6,
                confidence=confidence,
                lifecycle=lifecycle,
                classification_probabilities=class_probabilities,
                classification_entropy_nats=meta.get("classification_entropy_nats"),
                effective_classification_evidence=float(
                    meta.get("effective_classification_evidence", 0.0)
                ),
                source_ids=tuple(meta.get("source_ids", ())),
                report_lineage=tuple(meta.get("report_lineage", ())),
                last_measurement_time_s=float(meta.get("last_measurement_time_s", current_time_s)),
                lifetime_s=float(meta.get("lifetime_s", lifetime)),
                engageable=engageable,
                reference_frame=frame,
                suspect_deception=bool(suspect_deception or is_deception),
                emitter_hypothesis_id=(
                    str(meta["jammer_id"]) if meta.get("jammer_id") is not None else None
                ),
            )
        )
    return outs


def calculate_confidence(meta: dict, kf) -> float:
    """
    Calculate track confidence from five orthogonal factors.

    confidence = clamp01(
        (0.40 * obs_score + 0.60 * maturity) * recency * consistency * accuracy
    )

    The confidence spans the full ``[0, 1]`` range: a single radar that steadily
    holds a target and matures the track can reach ~1.0 (the accuracy factor is
    what keeps it just under 1.0 until the estimate is tight).  This makes it a
    usable dense learning signal — previously a lone-radar track saturated near
    ``0.6`` because ``obs_score`` was normalised by the *maximum* sensor count
    (``N_MAX_RADARS``), so a single sensor could never contribute more than
    ``1 / N_MAX_RADARS`` to the detection term.

    Factors
    -------
    obs_score   : EMA of per-update sensor-hit count, normalised by ``OBS_REF``
                  (= 1.0), so a single steadily-tracking sensor fully satisfies
                  the detection term.  Extra radars / datalink still help — they
                  raise ``obs_ema`` (keeping the term saturated through dropouts)
                  and reach a firm track sooner via maturity/accuracy.
    maturity    : Fraction of five successful updates reached. This lets a
                  confirmed track cross the 0.60 commit gate without making a
                  one-hit tentative track weapons quality.
                  New tentative tracks start near 0 and grow linearly.
    recency     : exp(-missed_updates / CONFIDENCE_RECENCY_TAU_UPDATES).
                  Decays over the coast window rather than halving per missed
                  update, so a scanning radar's normal revisit gap does not read
                  as track degradation.
    consistency : exp(-max(0, nis_ema - expected) / expected).
                  NIS_EMA initialised to 3.0 (chi² DOF=3); exponential penalty
                  is smoother than the old reciprocal and decays faster at high NIS,
                  which is better for deception / mis-association detection.
    accuracy    : Range-normalised localisation quality with a 0.55 floor. The
                  floor keeps track-existence confidence distinct from absolute
                  position precision, which naturally degrades at BVR range.
    """
    # --- obs_score ---
    # Normalise by OBS_REF (one steadily-tracking sensor), not N_MAX_RADARS, so a
    # lone radar can fully satisfy the detection term and confidence spans [0, 1].
    obs_ema = meta.get("obs_ema", 1.0)
    obs_score = min(1.0, obs_ema / OBS_REF)

    # --- maturity ---
    updates_with_meas = meta.get("updates_with_meas", 0)
    maturity = min(1.0, updates_with_meas / CONFIDENCE_MATURITY_UPDATES)

    # --- recency: exp(-missed / tau), tau in missed updates ---
    # NOT 0.5^missed: see CONFIDENCE_RECENCY_TAU_UPDATES. A scanning radar's normal
    # revisit gap must not read as track degradation.
    missed = kf.missed_updates
    recency = math.exp(-missed / CONFIDENCE_RECENCY_TAU_UPDATES)

    # --- consistency: exponential NIS penalty (smoother than reciprocal) ---
    NIS_EXPECTED = 3.0
    nis_ema = meta.get("nis_ema", NIS_EXPECTED)
    consistency = math.exp(-max(0.0, nis_ema - NIS_EXPECTED) / NIS_EXPECTED)

    # --- accuracy: rational 1/(1+(sigma/SIGMA_REF)^2) ---
    # The reference is RANGE-ADAPTIVE so the score measures *angular* tracking
    # quality, not absolute position error. A radar's cross-range uncertainty grows
    # ~ angle*range, so a fixed 500 m reference collapsed confidence to ~0 at BVR
    # ranges on large maps even for a firm, mature track. Judging pos_std against
    # ANGULAR_REF*range keeps a well-held track confident at any range while still
    # rewarding tight (close-in) tracks via the SIGMA_REF_FLOOR floor.
    try:
        P = kf.get_covariance()
        pos_var_rms = (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0
        pos_std = math.sqrt(max(0.0, pos_var_rms))
    except Exception:
        pos_std = 5000.0  # safe pessimistic default
    try:
        state = kf.get_state()
        range_m = math.sqrt(float(state[0]) ** 2 + float(state[1]) ** 2 + float(state[2]) ** 2)
    except Exception:
        range_m = 0.0
    sigma_ref = max(
        CONFIDENCE_POSITION_FLOOR_M,
        CONFIDENCE_ANGULAR_REFERENCE_RAD * range_m,
    )
    localization_quality = 1.0 / (1.0 + (pos_std / sigma_ref) ** 2)
    accuracy = CONFIDENCE_ACCURACY_FLOOR + (1.0 - CONFIDENCE_ACCURACY_FLOOR) * localization_quality

    raw = (0.40 * obs_score + 0.60 * maturity) * recency * consistency * accuracy
    return max(0.0, min(1.0, raw))


def get_or_create_meta(track_meta: dict, tid: int) -> dict:
    """
    Get metadata for track ID, creating default if not exists.

    Args:
        track_meta: Track metadata dictionary
        tid: Track ID

    Returns:
        Metadata dictionary for the track
    """
    if tid not in track_meta:
        track_meta[tid] = {
            "n_obs_hist": [],
            "lifetime": 0,
            "update_count": 0,
            "last_meas": None,
            "last_dt": None,
            "z_bias": 0.0,
            "xy_bias": (0.0, 0.0),
            # ECM fields
            "is_deception": False,
            "suspect_deception": False,
            "engagement_id": None,
            "jammer_id": None,
            "report_ids": (),
            "source_ids": (),
            "report_lineage": (),
            # Confidence v2 fields
            "obs_ema": 0.0,
            "updates_with_meas": 0,
            "nis_ema": 3.0,  # initialised to chi2 expected value (DOF=3)
        }
    return track_meta[tid]
