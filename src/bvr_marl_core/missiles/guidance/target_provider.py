import numpy as np

from bvr_marl_core.radar.core.utils import enu_to_geodetic, geodetic_to_enu
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.simulator.core.helpers import Position


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

        self.current_target_id = getattr(missile, "designated_target_id", None)
        self.last_confirmed_track_tid = None  # track ID (not unit ID) for PN velocity lookup
        self.last_known_track_tid = None
        self.last_known_target_id = self.current_target_id
        self.last_update_had_fresh_track = False
        self.last_confirmed_track_age_s = float("inf")
        self._seeded = False

    def update(self, sim, tick_secs):
        self.last_update_had_fresh_track = False
        self._maybe_seed_from_shooter()

        # designated_target_id may be set post-__init__ (e.g. fire_missile_direct).
        if self.current_target_id is None:
            self.current_target_id = getattr(self.missile, "designated_target_id", None)

        radar = self.missile.radar

        # Fix: synchronize guidance target identity with missile radar lock.
        # MissileRadar can retarget self.locked_target independently of whether
        # the old target is alive.  If we don't sync here, guidance keeps steering
        # toward the old target until it is confirmed dead.
        radar_locked_id = None
        if radar is not None and hasattr(radar, "get_locked_target"):
            radar_locked_id = radar.get_locked_target()

        if radar_locked_id is not None and radar_locked_id != self.current_target_id:
            self.current_target_id = radar_locked_id
            # Clear all state tied to the previous target so guidance does not
            # blend old position/velocity into the new engagement.
            self.last_confirmed_track_tid = None
            self.last_known_track_tid = None
            self.last_known_target_id = self.current_target_id
            self.last_confirmed_target_pos = None
            self.last_confirmed_target_vel = None
            self.last_confirmed_target_vel_ref = None
            self.last_confirmed_track_age_s = float("inf")

        tracks = radar.get_tracks() or []

        # Authoritative alive-unit snapshot: distinguishes confirmed-dead from temporarily untracked.
        active_ids = set(sim.active_units.keys())

        # Track IDs for non-engageable types (missiles, CMs, support assets).
        # Dead-target detection uses active_ids to avoid track-ID/unit-ID namespace collisions.
        # get_tracks() format: 15-value new or 11-value legacy (handled below).
        non_targetable_track_ids = set()
        for track in tracks:
            if len(track) >= 15:
                # New format (15 values)
                (
                    tid,
                    state,
                    cov,
                    tgt,
                    utype,
                    ref,
                    conf,
                    n_obs,
                    lifetime,
                    upd_cnt,
                    is_deception,
                    suspect_deception,
                    engagement_id,
                    jammer_id,
                    engageable,
                ) = track
            else:
                # Legacy 11-value format
                tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt = track

            if (
                getattr(tgt, "is_missile", False)
                or getattr(tgt, "is_countermeasure", False)
                or getattr(tgt, "is_non_engageable", False)
            ):
                non_targetable_track_ids.add(tid)

        locked_ids = set()
        if hasattr(radar, "get_locked_target"):
            lid = radar.get_locked_target()
            if lid is not None:
                locked_ids.add(lid)
        if not locked_ids and hasattr(radar, "get_locked_targets"):
            locked_ids = set(radar.get_locked_targets() or [])
        locked_ids -= non_targetable_track_ids

        # Helper: best track by quality, skipping non-targetable and dead tracks.
        def _best_track(filter_fn):
            best = None
            best_key = None
            tracker_map = None
            tracker_manager = getattr(radar, "tracker_manager", None)
            if tracker_manager is not None:
                maybe_tracks = getattr(tracker_manager, "tracks", None)
                if isinstance(maybe_tracks, dict):
                    tracker_map = maybe_tracks
            for track in tracks:
                if len(track) >= 15:
                    # New format (15 values)
                    (
                        tid,
                        state,
                        cov,
                        tgt,
                        utype,
                        ref,
                        conf,
                        n_obs,
                        lifetime,
                        upd_cnt,
                        is_deception,
                        suspect_deception,
                        engagement_id,
                        jammer_id,
                        engageable,
                    ) = track
                else:
                    # Legacy 11-value format
                    tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt = (
                        track
                    )
                    is_deception = is_false

                if is_deception:
                    continue
                if tid in non_targetable_track_ids:
                    continue
                if tracker_map is not None:
                    kf = tracker_map.get(tid)
                    if kf is not None and float(getattr(kf, "missed_updates", 0.0)) > 0.0:
                        continue
                if not filter_fn(tid, tgt):
                    continue
                key = (upd_cnt, conf, n_obs)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (tid, state, ref, conf, n_obs, upd_cnt, tgt)
            return best

        tid0 = self.current_target_id
        if tid0 in non_targetable_track_ids:
            tid0 = None
            self.current_target_id = None

        best = None

        target_confirmed_alive = tid0 is not None and tid0 in active_ids

        if tid0 is not None:
            best = _best_track(lambda tid, tgt: getattr(tgt, "id", None) == tid0)
            if best is not None and not target_confirmed_alive:
                best = None

        # Retarget only when the designated target is confirmed dead.
        # Alive-but-temporarily-untracked targets fall through to dead-reckoning to avoid
        # spurious target switches from transient tracker drops.
        if best is None and not target_confirmed_alive:
            retarget_policy = getattr(self.missile, "retarget_policy", "locked_override")
            allow_retarget = retarget_policy in ("locked_override", "truth_fallback")

            if allow_retarget:
                if locked_ids:
                    for locked_id in locked_ids:
                        cand = _best_track(lambda tid, tgt: getattr(tgt, "id", None) == locked_id)
                        if cand is not None:
                            best = cand
                            new_tid = getattr(cand[6] if len(cand) > 6 else None, "id", None)
                            if new_tid is not None:
                                self.current_target_id = new_tid
                            break

                if best is None and tracks:
                    cand = _best_track(lambda tid, tgt: True)
                    if cand is not None:
                        best = cand
                        new_tid = getattr(cand[6] if len(cand) > 6 else None, "id", None)
                        if new_tid is not None:
                            self.current_target_id = new_tid

                # Third priority (truth_fallback only): use truth position from sim
                # when no radar tracks are available (e.g. new target outside seeker FOV).
                if best is None and retarget_policy == "truth_fallback":
                    best = self._find_truth_retarget(sim)
                    if best is not None:
                        pass

            # Coast toward the frozen last-known position and retry next tick.
            # Do not clear last_confirmed_target_pos: that would cause straight-line flight.
            if best is None:
                return

        if best is None:
            # Fix 3: no valid tracker-backed track this tick — clear stale tid so
            # guidance phase selection does not mistakenly believe PN is available.
            self.last_confirmed_track_tid = None
            if np.isfinite(self.last_confirmed_track_age_s):
                self.last_confirmed_track_age_s += float(tick_secs)
            if (
                self.last_confirmed_target_pos is not None
                and self.last_confirmed_target_vel is not None
            ):
                dt = float(tick_secs)
                ref = self.last_confirmed_target_pos
                vel_ref = self.last_confirmed_target_vel_ref
                vel = np.asarray(self.last_confirmed_target_vel, dtype=float)

                # Fix 1: rotate cached velocity from ENU(vel_ref) into ENU(ref)
                # before propagating.  The two frames differ whenever the missile
                # has moved since the velocity was exported by the tracker.
                if vel_ref is not None:
                    dlat = abs(vel_ref.lat - ref.lat)
                    dlon = abs(vel_ref.lon - ref.lon)
                    if dlat > 1e-7 or dlon > 1e-7:
                        try:
                            R = enu_rotation(vel_ref.lat, vel_ref.lon, ref.lat, ref.lon)
                            vel = R @ vel
                        except Exception:
                            pass

                vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
                max_reasonable_vel = 800.0
                vel_magnitude = float(np.linalg.norm([vx, vy, vz]))
                if vel_magnitude > max_reasonable_vel:
                    scale = max_reasonable_vel / vel_magnitude
                    vx, vy, vz = vx * scale, vy * scale, vz * scale

                d_enu = np.array([vx * dt, vy * dt, vz * dt], dtype=float)
                lat, lon, alt = enu_to_geodetic(d_enu, ref.lat, ref.lon, ref.alt)
                new_pos = Position(lat, lon, alt)
                self.last_confirmed_target_pos = new_pos
                # Keep vel_ref in sync with the propagated position so each
                # subsequent dead-reckoning step starts in the correct frame.
                self.last_confirmed_target_vel = [vx, vy, vz]
                self.last_confirmed_target_vel_ref = new_pos
            return

        if len(best) >= 7:
            tid, state, ref, conf, n_obs, upd_cnt, target_obj = best
        else:
            tid, state, ref, conf, n_obs, upd_cnt = best[:6]
            target_obj = None

        self.last_confirmed_track_tid = tid
        self.last_known_track_tid = tid
        self.last_known_target_id = getattr(target_obj, "id", self.current_target_id)
        self.last_update_had_fresh_track = True
        self.last_confirmed_track_age_s = 0.0

        # Keep missile.target synchronized with the target guidance is actually
        # steering toward, every tick — not only on dead-target retargets.  The
        # radar lock (and therefore current_target_id) can switch between two
        # live targets via the radar's score/hysteresis logic; if missile.target
        # is left at the launch target, hit detection (substepper/CCD) checks the
        # wrong unit and the missile flies straight through the one it guides on.
        if target_obj is not None:
            old_target_id = getattr(getattr(self.missile, "target", None), "id", None)
            new_target_id = getattr(target_obj, "id", None)
            if old_target_id != new_target_id:
                group = getattr(self.missile, "group", None)
                missile_counts = getattr(sim, "_missile_target_counts", None)
                if missile_counts is not None and group is not None:
                    if old_target_id is not None:
                        grp_map = missile_counts.get(group, {})
                        if grp_map.get(old_target_id, 0) > 1:
                            grp_map[old_target_id] -= 1
                        elif old_target_id in grp_map:
                            del grp_map[old_target_id]
                    if new_target_id is not None:
                        grp_map = missile_counts.setdefault(group, {})
                        grp_map[new_target_id] = grp_map.get(new_target_id, 0) + 1
            self.missile.target = target_obj

        ref_pos = ref if ref is not None else self.missile.position
        pos = self._state_to_position(state, ref_pos)

        # --- Velocity extraction and validation ---
        # state[3:6] is in ENU at ref_pos (set by the tracker's export_ref,
        # which equals the missile's position at the time of the last radar
        # update).  We store both the vector and its tangent-point origin so
        # callers can rotate into any other ENU on demand.
        vel = state[3:6] if len(state) >= 6 else None
        if vel is not None:
            vel = np.array(vel, dtype=float)
            max_reasonable_vel = 800.0  # m/s - reasonable aircraft limit
            vel_magnitude = float(np.linalg.norm(vel))

            # Apply velocity bounds and smoothing
            if vel_magnitude > max_reasonable_vel:
                scale = max_reasonable_vel / vel_magnitude
                vel = vel * scale
            elif vel_magnitude < 50.0 and self.last_confirmed_target_vel is not None:
                # Blend slowly when KF velocity is still near zero (early convergence).
                # This prevents dead-reckoning from overshooting during track-loss periods
                # before the filter has settled. The primary PN path uses get_target_state_for_pn
                # instead of this value, so smoothing only affects dead-reckoning fallback.
                alpha = 0.3  # trust new measurement 30%, previous 70%
                vel = alpha * vel + (1.0 - alpha) * np.array(
                    self.last_confirmed_target_vel, dtype=float
                )

        self.last_confirmed_target_pos = pos
        self.last_confirmed_target_vel = vel.tolist() if vel is not None else None
        self.last_confirmed_target_vel_ref = ref_pos if vel is not None else None

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
            except Exception:
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
            except Exception:
                pass

        state = np.concatenate([enu, vel])
        self.current_target_id = getattr(best_unit, "id", None)
        return (None, state, missile_pos, 1.0, 1, 1, best_unit)

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
