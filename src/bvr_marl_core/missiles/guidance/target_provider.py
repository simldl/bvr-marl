from typing import NamedTuple

import numpy as np

from bvr_marl_core.domain.information import TrackLifecycle, TrackSnapshot, WeaponTrack
from bvr_marl_core.domain.sensing_visibility import is_sensor_invisible_to
from bvr_marl_core.radar.core.utils import enu_to_geodetic, geodetic_to_enu
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.simulator.core.helpers import Position

_POSITION_TYPE = Position

# Plausible upper bound on a target's speed; a track above this is treated as bad
# data and clamped rather than allowed to fling the position estimate.
_MAX_REASONABLE_TARGET_SPEED_MPS = 800.0
# Below this the tracker filter is still converging, so blend with the last estimate.
_UNCONVERGED_SPEED_MPS = 50.0
_VELOCITY_BLEND_ALPHA = 0.3

# Reference position RMS std (m) for normalising the guidance-track covariance into
# a [0, 1] terminal track-uncertainty (C_trk) consumed by the P_trk kill submodel.
# A well-resolved track (RMS std well below this) gets ~0 uncertainty and no Pk
# penalty; a noisy/jammed track whose RMS std approaches this saturates to 1. This
# is the live belief weapon-support channel: sensor/estimator covariance -> Pk.
_TRACK_UNCERTAINTY_REF_STD_M = 1500.0


def _normalized_track_uncertainty(covariance) -> float | None:
    """Map a guidance track's position covariance to C_trk in [0, 1].

    Uses the RMS of the position-block standard deviations, normalised by
    ``_TRACK_UNCERTAINTY_REF_STD_M``. Returns ``None`` when the covariance is
    unusable so the missile's uncertainty is left unchanged (neutral).
    """
    try:
        cov = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        return None
    if cov.ndim != 2 or cov.shape[0] < 3 or cov.shape[1] < 3:
        return None
    pos_var = np.array([cov[0, 0], cov[1, 1], cov[2, 2]], dtype=float)
    if not np.all(np.isfinite(pos_var)) or np.any(pos_var < 0.0):
        return None
    rms_std = float(np.sqrt(np.mean(pos_var)))
    return max(0.0, min(1.0, rms_std / _TRACK_UNCERTAINTY_REF_STD_M))


class _TrackCandidate(NamedTuple):
    """A track considered for guidance, with the quality fields used to rank it."""

    track_id: object | None
    state: tuple
    reference: Position | None
    confidence: float
    source_count: int
    lifetime_s: float
    target_obj: object | None


def _launch_designation(missile):
    """The contact a weapon was committed to at launch, if it carries one.

    ``launch_contact_id`` is the immutable launch identity; the older
    ``designated_target_id`` still carries a unit id on the oracle path. Ids are
    ints or strings in both namespaces, so anything else is an unset field rather
    than a designation, and guidance is better left unseeded than seeded wrong.
    """
    for name in ("launch_contact_id", "designated_target_id"):
        value = getattr(missile, name, None)
        if isinstance(value, (int, str)):
            return value
    return None


class GuidanceTargetProvider:
    """
    Caches the last confirmed guidance target position and velocity from the
    missile's tracker.

    Velocity frame invariant
    ------------------------
    ``last_confirmed_target_vel`` is expressed in ENU whose tangent point is
    stored as ``last_confirmed_target_vel_ref`` (a Position).  Before using the
    velocity for guidance, callers *must* rotate it into ENU at the current
    missile position:

        v_msl = enu_rotation(vel_ref.lat, vel_ref.lon,
                              missile_pos.lat, missile_pos.lon) @ v_cached

    The helper ``get_guidance_velocity_in_missile_enu(missile_pos)`` performs
    this rotation automatically.  Raw ``get_guidance_velocity()`` is kept for
    backward compatibility but should only be used when the caller knows the
    missile is still at the same position as when the velocity was exported.
    """

    def __init__(self, missile):
        self.missile = missile
        self.last_confirmed_target_pos = None

        # Velocity in ENU at last_confirmed_target_vel_ref (NOT necessarily
        # missile ENU — see class docstring before using directly).
        self.last_confirmed_target_vel = None
        # Tangent-point origin for the cached velocity vector.
        self.last_confirmed_target_vel_ref: Position | None = None

        # Seeded from the launch designation; from here on this is the *guidance*
        # identity and follows the seeker, which is why it is not read back from
        # the launch fields again.
        self.current_target_id = _launch_designation(missile)
        self.last_confirmed_track_tid = None  # track ID (not unit ID) for PN velocity lookup
        self.last_known_track_tid = None
        self.last_known_target_id = self.current_target_id
        self.last_update_had_fresh_track = False
        self.last_confirmed_track_age_s = float("inf")
        self._seeded = False

    def update(self, sim, tick_secs):
        """Refresh the cached guidance target from the missile's sensors.

        Ordered by authority: an active seduction outranks the weapon track, which
        outranks the seeker's own tracks. When nothing usable is available the last
        confirmed target is dead-reckoned forward.
        """
        self.last_update_had_fresh_track = False

        if self._apply_seduced_position():
            return

        if isinstance(getattr(self.missile, "weapon_track", None), WeaponTrack):
            self._update_from_weapon_track(sim, tick_secs)
            return

        if self._apply_countermeasure_seduction(sim):
            return

        self._maybe_seed_from_shooter()
        self._adopt_late_launch_designation()

        radar = self.missile.radar
        self._sync_identity_with_radar_lock(radar)

        tracks = self._radar_tracks(radar)
        non_targetable = self._non_targetable_track_ids(tracks)
        locked_ids = self._locked_track_ids(radar, non_targetable)

        best = self._select_best(sim, radar, tracks, non_targetable, locked_ids)
        if best is None:
            self._dead_reckon(tick_secs)
            return

        self._commit_candidate(sim, best)

    # -- seduction ------------------------------------------------------------

    def _apply_seduced_position(self) -> bool:
        """Home on an explicitly seduced position. True when guidance is settled."""
        seduced_position = getattr(self.missile, "seduced_position", None)
        if not isinstance(seduced_position, _POSITION_TYPE):
            return False
        self.last_confirmed_target_pos = Position(
            seduced_position.lat, seduced_position.lon, seduced_position.alt
        )
        self.last_confirmed_target_vel = [0.0, 0.0, 0.0]
        self.last_confirmed_target_vel_ref = self.last_confirmed_target_pos
        self.last_confirmed_track_tid = None
        self.last_known_track_tid = None
        return True

    def _apply_countermeasure_seduction(self, sim) -> bool:
        """Home on a countermeasure once decoyed. True when guidance is settled.

        The commitment survives the countermeasure expiring — guidance freezes at
        its last position rather than re-acquiring the real target.
        """
        seduced = getattr(self.missile, "seduced_by", None)
        if getattr(seduced, "is_countermeasure", False) is not True:
            return False
        if getattr(seduced, "id", None) in getattr(sim, "active_units", {}):
            p = seduced.position
            self.last_confirmed_target_pos = Position(p.lat, p.lon, p.alt)
            self.last_confirmed_target_vel = [0.0, 0.0, 0.0]
            self.last_confirmed_target_vel_ref = self.last_confirmed_target_pos
        # Drop the tracker-track id so PN uses the (decoy) guidance position instead
        # of the missile's radar track of the real aircraft.
        self.last_confirmed_track_tid = None
        self.last_known_track_tid = None
        return True

    # -- identity -------------------------------------------------------------

    def _adopt_late_launch_designation(self) -> None:
        """Pick up a designation assigned after __init__ (e.g. fire_missile_direct)."""
        if self.current_target_id is not None:
            return
        self.current_target_id = getattr(self.missile, "launch_contact_id", None) or getattr(
            self.missile, "designated_target_id", None
        )

    def _sync_identity_with_radar_lock(self, radar) -> None:
        """Follow the seeker when it retargets.

        MissileRadar can move its lock independently of whether the old target is
        alive; without this, guidance keeps steering at the old target until that
        target is confirmed dead.
        """
        radar_locked_id = None
        if radar is not None and hasattr(radar, "get_locked_target"):
            radar_locked_id = radar.get_locked_target()

        if radar_locked_id is None or radar_locked_id == self.current_target_id:
            return

        self.current_target_id = radar_locked_id
        # Clear all state tied to the previous target so guidance does not blend
        # old position/velocity into the new engagement.
        self.last_confirmed_track_tid = None
        self.last_known_track_tid = None
        self.last_known_target_id = self.current_target_id
        self.last_confirmed_target_pos = None
        self.last_confirmed_target_vel = None
        self.last_confirmed_target_vel_ref = None
        self.last_confirmed_track_age_s = float("inf")

    # -- track selection ------------------------------------------------------

    @staticmethod
    def _radar_tracks(radar) -> list:
        tracks = radar.get_tracks() or []
        if not all(isinstance(track, TrackSnapshot) for track in tracks):
            raise TypeError("Missile radar tracks must use the TrackSnapshot contract.")
        return tracks

    @staticmethod
    def _non_targetable_track_ids(tracks) -> set:
        return {
            track.track_id
            for track in tracks
            if "missile" in track.classification or not track.engageable
        }

    @staticmethod
    def _locked_track_ids(radar, non_targetable: set) -> set:
        locked_ids = set()
        if hasattr(radar, "get_locked_target"):
            lid = radar.get_locked_target()
            if lid is not None:
                locked_ids.add(lid)
        if not locked_ids and hasattr(radar, "get_locked_targets"):
            locked_ids = set(radar.get_locked_targets() or [])
        return locked_ids - non_targetable

    @staticmethod
    def _tracker_filter_map(radar):
        """Kalman-filter map keyed by track id, when the radar exposes one."""
        tracker_manager = getattr(radar, "tracker_manager", None)
        if tracker_manager is None:
            return None
        maybe_tracks = getattr(tracker_manager, "tracks", None)
        return maybe_tracks if isinstance(maybe_tracks, dict) else None

    def _best_track(self, tracks, radar, non_targetable: set, accept) -> "_TrackCandidate | None":
        """Highest-quality track passing ``accept``, ranked by (lifetime, confidence, sources).

        Skips deception-suspect, non-targetable, and coasting (missed-update) tracks.
        """
        best = None
        best_key = None
        tracker_map = self._tracker_filter_map(radar)

        for track in tracks:
            tid = track.track_id
            if track.suspect_deception or tid in non_targetable:
                continue
            if tracker_map is not None:
                kf = tracker_map.get(tid)
                if kf is not None and float(getattr(kf, "missed_updates", 0.0)) > 0.0:
                    continue
            if not accept(tid):
                continue

            frame = track.reference_frame
            key = (track.lifetime_s, track.confidence, len(track.source_ids))
            if best_key is None or key > best_key:
                best_key = key
                best = _TrackCandidate(
                    track_id=tid,
                    state=track.state,
                    reference=(
                        Position(frame.latitude_deg, frame.longitude_deg, frame.altitude_m)
                        if frame is not None
                        else None
                    ),
                    confidence=track.confidence,
                    source_count=len(track.source_ids),
                    lifetime_s=track.lifetime_s,
                    target_obj=None,
                )
        return best

    def _select_best(
        self, sim, radar, tracks, non_targetable: set, locked_ids: set
    ) -> "_TrackCandidate | None":
        """Pick the track to guide on, in descending order of authority."""
        tid0 = self.current_target_id
        if tid0 in non_targetable:
            tid0 = None
            self.current_target_id = None

        if tid0 is not None:
            best = self._best_track(tracks, radar, non_targetable, lambda tid: tid == tid0)
            if best is not None:
                return best

        retarget_policy = getattr(self.missile, "retarget_policy", "locked_override")
        if retarget_policy not in ("locked_override", "truth_fallback"):
            return None
        return self._retarget(sim, radar, tracks, non_targetable, locked_ids, retarget_policy)

    def _retarget(
        self, sim, radar, tracks, non_targetable: set, locked_ids: set, retarget_policy: str
    ) -> "_TrackCandidate | None":
        """Find a replacement target: seeker locks first, then any track, then truth."""
        for locked_id in locked_ids:
            cand = self._best_track(tracks, radar, non_targetable, lambda tid: tid == locked_id)
            if cand is not None:
                return self._adopt(cand)

        if tracks:
            cand = self._best_track(tracks, radar, non_targetable, lambda _tid: True)
            if cand is not None:
                return self._adopt(cand)

        # Truth fallback only: steer on sim truth when the seeker holds no track at
        # all (e.g. a new target still outside the seeker field of view).
        if retarget_policy == "truth_fallback":
            return self._find_truth_retarget(sim)
        return None

    def _adopt(self, candidate: "_TrackCandidate") -> "_TrackCandidate":
        """Make a retargeted candidate the current guidance identity."""
        if candidate.track_id is not None:
            self.current_target_id = candidate.track_id
        return candidate

    # -- committing a result --------------------------------------------------

    def _dead_reckon(self, tick_secs) -> None:
        """Propagate the last confirmed target forward when no track is available."""
        # No tracker-backed track this tick: clear the stale tid so guidance-phase
        # selection does not believe PN is available.
        self.last_confirmed_track_tid = None
        if np.isfinite(self.last_confirmed_track_age_s):
            self.last_confirmed_track_age_s += float(tick_secs)

        if self.last_confirmed_target_pos is None or self.last_confirmed_target_vel is None:
            return

        dt = float(tick_secs)
        ref = self.last_confirmed_target_pos
        vel_ref = self.last_confirmed_target_vel_ref
        vel = np.asarray(self.last_confirmed_target_vel, dtype=float)

        # Rotate the cached velocity from ENU(vel_ref) into ENU(ref) before
        # propagating; the frames differ whenever the missile has moved since the
        # tracker exported the velocity.
        if vel_ref is not None:
            dlat = abs(vel_ref.lat - ref.lat)
            dlon = abs(vel_ref.lon - ref.lon)
            if dlat > 1e-7 or dlon > 1e-7:
                try:
                    R = enu_rotation(vel_ref.lat, vel_ref.lon, ref.lat, ref.lon)
                    vel = R @ vel
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ZeroDivisionError,
                ):
                    pass

        vx, vy, vz = self._clamp_velocity(float(vel[0]), float(vel[1]), float(vel[2]))
        d_enu = np.array([vx * dt, vy * dt, vz * dt], dtype=float)
        lat, lon, alt = enu_to_geodetic(d_enu, ref.lat, ref.lon, ref.alt)
        new_pos = Position(lat, lon, alt)

        self.last_confirmed_target_pos = new_pos
        # Keep vel_ref in sync with the propagated position so the next
        # dead-reckoning step starts in the correct frame.
        self.last_confirmed_target_vel = [vx, vy, vz]
        self.last_confirmed_target_vel_ref = new_pos

    @staticmethod
    def _clamp_velocity(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        """Clamp to a plausible aircraft speed so a bad track cannot fling the estimate."""
        magnitude = float(np.linalg.norm([vx, vy, vz]))
        if magnitude > _MAX_REASONABLE_TARGET_SPEED_MPS:
            scale = _MAX_REASONABLE_TARGET_SPEED_MPS / magnitude
            return vx * scale, vy * scale, vz * scale
        return vx, vy, vz

    def _commit_candidate(self, sim, best: "_TrackCandidate") -> None:
        """Adopt the selected candidate as the confirmed guidance target."""
        self.last_confirmed_track_tid = best.track_id
        self.last_known_track_tid = best.track_id
        self.last_known_target_id = getattr(best.target_obj, "id", self.current_target_id)
        self.last_update_had_fresh_track = True
        self.last_confirmed_track_age_s = 0.0

        self._resync_missile_target(sim, best.target_obj)

        ref_pos = best.reference if best.reference is not None else self.missile.position
        pos = self._state_to_position(best.state, ref_pos)
        vel = self._extract_velocity(best.state)

        self.last_confirmed_target_pos = pos
        self.last_confirmed_target_vel = vel.tolist() if vel is not None else None
        self.last_confirmed_target_vel_ref = ref_pos if vel is not None else None

    def _resync_missile_target(self, sim, target_obj) -> None:
        """Keep ``missile.target`` on the unit guidance actually steers toward.

        The radar lock can switch between two live targets via score/hysteresis. If
        ``missile.target`` stayed on the launch target, hit detection would check the
        wrong unit and the missile would fly through the one it is guiding on.
        """
        if target_obj is None:
            return
        old_target_id = getattr(getattr(self.missile, "target", None), "id", None)
        new_target_id = getattr(target_obj, "id", None)
        if old_target_id != new_target_id:
            resync = getattr(sim, "resync_missile_target", None)
            if callable(resync):
                resync(getattr(self.missile, "group", None), old_target_id, new_target_id)
        self.missile.target = target_obj

    def _extract_velocity(self, state):
        """Bounded target velocity in ENU at the track's reference frame.

        ``state[3:6]`` is ENU at the tracker's export reference. Near-zero speeds are
        blended with the previous estimate because an unconverged filter would
        otherwise make dead-reckoning stall; the primary PN path reads
        ``get_target_state_for_pn`` instead, so the smoothing only affects fallback.
        """
        if len(state) < 6:
            return None

        vel = np.array(state[3:6], dtype=float)
        magnitude = float(np.linalg.norm(vel))
        if magnitude > _MAX_REASONABLE_TARGET_SPEED_MPS:
            return vel * (_MAX_REASONABLE_TARGET_SPEED_MPS / magnitude)
        if magnitude < _UNCONVERGED_SPEED_MPS and self.last_confirmed_target_vel is not None:
            alpha = _VELOCITY_BLEND_ALPHA  # trust the new measurement this much
            return alpha * vel + (1.0 - alpha) * np.array(
                self.last_confirmed_target_vel, dtype=float
            )
        return vel

    def get_guidance_target(self):
        return self.last_confirmed_target_pos

    def get_guidance_velocity(self):
        """
        Return the cached target velocity as stored (ENU at
        last_confirmed_target_vel_ref).  Use only when the caller knows the
        missile is at the same position as when the velocity was exported, or
        when a small frame error is acceptable.  Prefer
        get_guidance_velocity_in_missile_enu() for guidance computations.
        """
        return self.last_confirmed_target_vel

    def get_guidance_velocity_in_missile_enu(self, missile_position):
        """
        Return target velocity rotated into ENU at missile_position.

        If the cached velocity reference (last_confirmed_target_vel_ref) matches
        missile_position within ~1 m (same export tick), no rotation is applied.
        Otherwise, the vector is rotated from ENU(vel_ref) -> ENU(missile_pos)
        using the standard ENU-to-ENU rotation matrix.

        Returns None if no velocity is cached.
        """
        vel = self.last_confirmed_target_vel
        if vel is None:
            return None
        vel_ref = self.last_confirmed_target_vel_ref
        if vel_ref is None:
            return vel  # legacy path: no reference stored

        dlat = abs(vel_ref.lat - missile_position.lat)
        dlon = abs(vel_ref.lon - missile_position.lon)
        if dlat < 1e-7 and dlon < 1e-7:
            return vel  # same tangent point; no rotation needed

        try:
            R = enu_rotation(vel_ref.lat, vel_ref.lon, missile_position.lat, missile_position.lon)
            v_rotated = R @ np.asarray(vel, dtype=float)
            return v_rotated.tolist()
        except Exception:
            return vel  # rotation failed — return as-is

    def get_guidance_tid(self):
        """Return the tracker track ID of the current guidance target, or None.
        This is the key into TrackerManager.tracks / track_refs for PN velocity lookup.
        """
        return self.last_confirmed_track_tid

    def has_fresh_track(self) -> bool:
        return bool(
            self.last_update_had_fresh_track
            and self.last_confirmed_track_tid is not None
            and self.last_confirmed_track_age_s <= 1e-9
        )

    def has_coastable_track(self, max_age_s: float = 1.0) -> bool:
        """Return True for a short, same-target tracker coast."""
        if self.last_known_track_tid is None:
            return False
        if self.last_confirmed_target_pos is None:
            return False
        if self.last_confirmed_target_vel is None:
            return False
        if self.last_known_target_id != self.current_target_id:
            return False
        try:
            age = float(self.last_confirmed_track_age_s)
        except Exception:
            return False
        return 0.0 < age <= float(max_age_s)

    def get_coast_guidance_tid(self, max_age_s: float = 1.0):
        """Return the last same-target track ID when it is safe to coast briefly."""
        if self.has_coastable_track(max_age_s=max_age_s):
            return self.last_known_track_tid
        return None

    def _state_to_position(self, state, ref):
        enu = state[:3]
        lat, lon, alt = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)
        return Position(lat, lon, alt)

    @staticmethod
    def _position_reference(frame, fallback):
        """Convert an immutable frame record to the geodetic helper type."""
        if frame is None:
            return fallback
        return Position(frame.latitude_deg, frame.longitude_deg, frame.altitude_m)

    def _find_truth_retarget(self, sim):
        """Last-resort retarget using truth positions (truth_fallback policy only).

        Returns a synthetic best-track 7-tuple compatible with step 5, or None.
        The tuple is (tid=None, state_6d, ref_pos, conf, n_obs, upd_cnt, target_obj)
        where state_6d = [enu_x, enu_y, enu_z, vel_x, vel_y, vel_z] relative to
        the missile's current position (ref_pos).
        """
        missile_pos = self.missile.position
        best_unit = None
        best_range_sq = float("inf")

        for unit in sim.active_units.values():
            if (
                getattr(unit, "is_missile", False)
                or getattr(unit, "is_countermeasure", False)
                or getattr(unit, "is_non_engageable", False)
                # See missile.py: ROE (`is_non_engageable`) refuses the shot, this flag
                # says the seeker never sees the unit at all. They are set independently.
                or is_sensor_invisible_to(unit, self.missile.group)
            ):
                continue
            if unit.group == self.missile.group:
                continue
            try:
                dlat = (unit.position.lat - missile_pos.lat) * 111_320.0
                dlon = (
                    (unit.position.lon - missile_pos.lon)
                    * 111_320.0
                    * np.cos(np.radians(missile_pos.lat))
                )
                dalt = unit.position.alt - missile_pos.alt
                r2 = dlat**2 + dlon**2 + dalt**2
            except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
                continue
            if r2 < best_range_sq:
                best_range_sq = r2
                best_unit = unit

        if best_unit is None:
            return None

        try:
            enu = np.asarray(
                geodetic_to_enu(
                    best_unit.position.lat,
                    best_unit.position.lon,
                    best_unit.position.alt,
                    missile_pos.lat,
                    missile_pos.lon,
                    missile_pos.alt,
                ),
                dtype=float,
            )
        except Exception:
            return None

        vel = np.zeros(3, dtype=float)
        if hasattr(best_unit, "velocity_enu"):
            try:
                vel = np.asarray(best_unit.velocity_enu, dtype=float)
            except (TypeError, ValueError):
                pass

        state = np.concatenate([enu, vel])
        self.current_target_id = getattr(best_unit, "id", None)
        return _TrackCandidate(
            track_id=None,
            state=state,
            reference=missile_pos,
            confidence=1.0,
            source_count=1,
            lifetime_s=1,
            target_obj=best_unit,
        )

    def _maybe_seed_from_shooter(self):
        if self._seeded:
            return
        enu = getattr(self.missile, "initial_tracked_position_enu", None)
        ref = getattr(self.missile, "tracker_reference_pos", None)
        vel = getattr(self.missile, "initial_tracked_velocity_enu", None)
        if enu is not None and ref is not None:
            lat, lon, alt = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)
            self.last_confirmed_target_pos = Position(lat, lon, alt)
            # Velocity from shooter's tracker export: ENU at the shooter's
            # position (tracker_reference_pos = shooter.position at launch).
            self.last_confirmed_target_vel = vel
            self.last_confirmed_target_vel_ref = ref
        self._seeded = True

    #: Endgame re-association gate (m). A seeker return further than this from the
    #: coasted estimate is a different object, not the cued target. Sized well above
    #: the estimate's own drift over a few seconds of coasting and well below the
    #: separation between distinct contacts at BVR.
    REASSOCIATION_GATE_M: float = 2_000.0

    def _reassociate_seeker_tracks(self, candidates: list) -> list:
        """Seeker tracks consistent with the coasted estimate of the cued target.

        Returns the single closest track inside :attr:`REASSOCIATION_GATE_M`, or an
        empty list when nothing is close enough -- so a weapon that has genuinely
        lost its target still coasts rather than grabbing whatever it can see.
        """
        anchor = self.last_confirmed_target_pos
        if anchor is None or not candidates:
            return []
        best = None
        best_sq = None
        for track in candidates:
            ref = getattr(track, "reference_frame", None)
            ref_pos = self._position_reference(ref, self.missile.position)
            try:
                pos = self._state_to_position(track.state, ref_pos)
            except (TypeError, ValueError, IndexError):
                continue
            dn = (float(pos.lat) - float(anchor.lat)) * 111_000.0
            de = (float(pos.lon) - float(anchor.lon)) * 111_000.0
            du = float(pos.alt) - float(anchor.alt)
            sq = dn * dn + de * de + du * du
            if best_sq is None or sq < best_sq:
                best, best_sq = track, sq
        if best is None or best_sq > self.REASSOCIATION_GATE_M**2:
            return []
        return [best]

    def _update_from_weapon_track(self, sim, tick_secs: float) -> None:
        """Update/coast guidance using immutable estimates and no evaluator roster."""
        self._maybe_seed_from_shooter()
        radar = self.missile.radar
        tracks = radar.get_tracks() or []
        locked_id = getattr(radar, "get_locked_target", lambda: None)()
        candidates = [track for track in tracks if track.engageable]
        retarget_policy = getattr(self.missile, "retarget_policy", "locked_override")
        if retarget_policy == "track_only":
            designated_id = self.missile.weapon_track.snapshot.track_id
            matched = [track for track in candidates if track.track_id == designated_id]
            if not matched:
                # ENDGAME RE-ASSOCIATION.
                #
                # `track_only` exists to stop the weapon wandering onto a different
                # target, and it does that by identity. But the missile's OWN seeker
                # numbers its tracks in its own namespace, so when it acquires at
                # close range its track carries a different id than the contact the
                # weapon was launched against -- and the filter then discards the
                # only sensor that can still see the target.
                #
                # Measured on a stationary-target intercept: the seeker held the designated
                # id 1000001 out to 6.7 km, flipped to its own id 8 at 5.3 km, and
                # from there the provider committed nothing. At a 206 m closest
                # approach the shot was scored as 5.75 s stale and unlocked, so
                # P_trk fell to 0.349 and a well-guided intercept scored Pk 0.277.
                #
                # A real active seeker re-associates: a return in the predicted
                # basket IS the target it was cued onto. Gate spatially on the
                # coasted estimate, keep the designated identity for reporting, and
                # the no-wander guarantee still holds because a track outside the
                # gate is still rejected.
                matched = self._reassociate_seeker_tracks(candidates)
            candidates = matched
        elif locked_id is not None:
            locked = [track for track in candidates if track.track_id == locked_id]
            if locked:
                candidates = locked
        if candidates:
            track = max(
                candidates,
                key=lambda item: (item.confidence, item.lifetime_s, str(item.track_id)),
            )
            tid = track.track_id
            state, covariance = track.state, track.covariance
            ref, confidence = track.reference_frame, track.confidence
            try:
                state_time_s = float(getattr(sim, "elapsed_time_s", 0.0))
            except (TypeError, ValueError):
                state_time_s = float(self.missile.weapon_track.snapshot.state_time_s) + float(
                    tick_secs
                )
            lifecycle = (
                TrackLifecycle.REACQUIRED
                if self.last_confirmed_track_age_s > 0.0
                else TrackLifecycle.CONFIRMED
            )
            snapshot = TrackSnapshot(
                track_id=tid,
                state_time_s=state_time_s,
                state=tuple(state),
                covariance=tuple(tuple(row) for row in covariance),
                confidence=float(confidence),
                lifecycle=lifecycle,
                classification_probabilities=track.classification_probabilities,
                source_ids=tuple(
                    getattr(radar.tracker_manager, "track_meta", {})
                    .get(tid, {})
                    .get("source_ids", ())
                ),
                report_lineage=tuple(
                    getattr(radar.tracker_manager, "track_meta", {})
                    .get(tid, {})
                    .get("report_lineage", ())
                ),
                last_measurement_time_s=track.last_measurement_time_s,
                lifetime_s=track.lifetime_s,
                engageable=track.engageable,
                reference_frame=track.reference_frame,
                suspect_deception=track.suspect_deception,
                emitter_hypothesis_id=track.emitter_hypothesis_id,
            )
            previous = self.missile.weapon_track
            self.missile.weapon_track = WeaponTrack(
                snapshot=snapshot,
                launch_time_s=previous.launch_time_s,
                update_sequence=previous.update_sequence + 1,
            )
            self.current_target_id = tid
            self.last_confirmed_track_tid = tid
            self.last_known_track_tid = tid
            self.last_known_target_id = tid
            self.last_update_had_fresh_track = True
            self.last_confirmed_track_age_s = 0.0
            c_trk = _normalized_track_uncertainty(covariance)
            if c_trk is not None:
                self.missile.terminal_track_uncertainty = c_trk
            ref_pos = self._position_reference(ref, self.missile.position)
            self.last_confirmed_target_pos = self._state_to_position(state, ref_pos)
            self.last_confirmed_target_vel = np.asarray(state[3:6], dtype=float).tolist()
            self.last_confirmed_target_vel_ref = ref_pos
            return

        self.last_confirmed_track_tid = None
        self.last_confirmed_track_age_s = (
            0.0
            if not np.isfinite(self.last_confirmed_track_age_s)
            else self.last_confirmed_track_age_s
        ) + float(tick_secs)
        if self.last_confirmed_target_pos is None or self.last_confirmed_target_vel is None:
            return
        velocity = np.asarray(self.last_confirmed_target_vel, dtype=float)
        reference = self.last_confirmed_target_pos
        velocity_ref = self.last_confirmed_target_vel_ref
        if velocity_ref is not None:
            rotation = enu_rotation(
                velocity_ref.lat,
                velocity_ref.lon,
                reference.lat,
                reference.lon,
            )
            velocity = rotation @ velocity
        displacement = velocity * float(tick_secs)
        lat, lon, alt = enu_to_geodetic(
            displacement,
            reference.lat,
            reference.lon,
            reference.alt,
        )
        self.last_confirmed_target_pos = Position(lat, lon, alt)
        self.last_confirmed_target_vel = velocity.tolist()
        self.last_confirmed_target_vel_ref = self.last_confirmed_target_pos
