import numpy as np
from radar.core.utils import enu_to_geodetic
from simulator.core.helpers import Position

class GuidanceTargetProvider:

    def __init__(self, missile):
        self.missile = missile
        self.last_confirmed_target_pos = None
        self.last_confirmed_target_vel = None
        self.current_target_id = getattr(missile, "designated_target_id", None)
        self._seeded = False

    def update(self, sim, tick_secs):
        self._maybe_seed_from_shooter()
        radar = self.missile.radar
        tracks = radar.get_tracks() or []

        # Build a quick lookup of which TIDs are missiles or countermeasures (non-targetable)
        # NOTE: get_tracks() format changed from 11 to 15 values
        # Old: tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt
        # New: tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count,
        #      is_deception, suspect_deception, engagement_id, jammer_id, engageable
        non_targetable_track_ids = set()
        for track in tracks:
            if len(track) >= 15:
                # New format (15 values)
                tid, state, cov, tgt, utype, ref, conf, n_obs, lifetime, upd_cnt, is_deception, suspect_deception, engagement_id, jammer_id, engageable = track
            else:
                # Old format (11 values) - for backward compatibility
                tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt = track
                is_deception = is_false

            # Skip missiles and countermeasures - they are not valid targets
            if getattr(tgt, "is_missile", False) or getattr(tgt, "is_countermeasure", False):
                non_targetable_track_ids.add(tid)

        # 1) Gather own locks and drop any that are missiles or countermeasures
        locked_ids = set()
        if hasattr(radar, "get_locked_target"):
            lid = radar.get_locked_target()
            if lid is not None:
                locked_ids.add(lid)
        if not locked_ids and hasattr(radar, "get_locked_targets"):
            locked_ids = set(radar.get_locked_targets() or [])
        # Filter out non-targetable entities (missiles and countermeasures)
        locked_ids -= non_targetable_track_ids

        # Helper: best track by quality, but never missiles or countermeasures
        def _best_track(filter_fn):
            best = None; best_key = None
            for track in tracks:
                if len(track) >= 15:
                    # New format (15 values)
                    tid, state, cov, tgt, utype, ref, conf, n_obs, lifetime, upd_cnt, is_deception, suspect_deception, engagement_id, jammer_id, engageable = track
                else:
                    # Old format (11 values)
                    tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt = track
                    is_deception = is_false

                if is_deception:
                    continue
                # Skip non-targetable entities (missiles and countermeasures)
                if getattr(tgt, "is_missile", False) or getattr(tgt, "is_countermeasure", False):
                    continue
                if not filter_fn(tid, tgt):
                    continue
                key = (upd_cnt, conf, n_obs)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (tid, state, ref, conf, n_obs, upd_cnt, tgt)  # Include tgt in return
            return best

        # 2) Primary: designated ID (if it isn't a missile or countermeasure)
        tid0 = self.current_target_id
        if tid0 in non_targetable_track_ids:
            # Drop designated target if it's a missile or countermeasure
            tid0 = None
            self.current_target_id = None

        best = None
        retargeted = False
        
        if tid0 is not None:
            # Try to find designated target by unit ID (not track ID)
            best = _best_track(lambda tid, tgt: getattr(tgt, "id", None) == tid0)

        # 3) Retargeting: if we don't have the designated target, look for alternatives
        if best is None:
            allow_retarget = getattr(self.missile, "retarget_policy", "locked_override") == "locked_override"
            
            if allow_retarget:
                # First priority: any locked target
                if locked_ids:
                    # Try to find a locked target by unit ID
                    for locked_id in locked_ids:
                        cand = _best_track(lambda tid, tgt: getattr(tgt, "id", None) == locked_id)
                        if cand is not None:
                            best = cand
                            retargeted = True
                            new_tid = getattr(cand[6] if len(cand) > 6 else None, "id", None)
                            if new_tid is not None:
                                self.current_target_id = new_tid
                            break
                
                # Second priority: best available track (if no locks found)
                if best is None and tracks:
                    # Get any valid target track
                    cand = _best_track(lambda tid, tgt: True)  # Accept any valid track
                    if cand is not None:
                        best = cand
                        retargeted = True
                        new_tid = getattr(cand[6] if len(cand) > 6 else None, "id", None)
                        if new_tid is not None:
                            self.current_target_id = new_tid

        # 4) Dead-reckoning fallback with improved ENU velocity handling
        if best is None:
            if self.last_confirmed_target_pos is not None and self.last_confirmed_target_vel is not None:
                vx, vy, vz = self.last_confirmed_target_vel  # ENU m/s
                dt = float(tick_secs)
                ref = self.last_confirmed_target_pos  # propagate around last known pos
                
                # Enhanced velocity handling with bounds checking
                max_reasonable_vel = 800.0  # m/s - reasonable aircraft limit
                vel_magnitude = float(np.linalg.norm([vx, vy, vz]))
                if vel_magnitude > max_reasonable_vel:
                    # Scale down unreasonable velocities
                    scale = max_reasonable_vel / vel_magnitude
                    vx, vy, vz = vx * scale, vy * scale, vz * scale
                
                d_enu = np.array([vx*dt, vy*dt, vz*dt], dtype=float)
                lat, lon, alt = enu_to_geodetic(d_enu, ref.lat, ref.lon, ref.alt)
                self.last_confirmed_target_pos = Position(lat, lon, alt)
            return

        # 5) State -> geodetic and store with enhanced velocity processing
        # Extract target object from best track
        if len(best) >= 7:
            tid, state, ref, conf, n_obs, upd_cnt, target_obj = best
        else:
            # Fallback if format doesn't include target object
            tid, state, ref, conf, n_obs, upd_cnt = best[:6]
            target_obj = None
        
        # Update missile.target if retargeting occurred and we have a valid target object
        if retargeted and target_obj is not None:
            self.missile.target = target_obj
        
        ref_pos = ref if ref is not None else self.missile.position
        pos = self._state_to_position(state, ref_pos)
        
        # Enhanced velocity extraction and validation
        vel = state[3:6] if len(state) >= 6 else None
        if vel is not None:
            vel = np.array(vel, dtype=float)
            max_reasonable_vel = 800.0  # m/s - reasonable aircraft limit
            vel_magnitude = float(np.linalg.norm(vel))
            
            # Apply velocity bounds and smoothing
            if vel_magnitude > max_reasonable_vel:
                # Scale down unreasonable velocities
                scale = max_reasonable_vel / vel_magnitude
                vel = vel * scale
            elif vel_magnitude < 50.0 and self.last_confirmed_target_vel is not None:
                # If velocity seems too low, blend with previous estimate
                alpha = 0.3  # trust new measurement 30%, previous 70%
                vel = alpha * vel + (1.0 - alpha) * np.array(self.last_confirmed_target_vel, dtype=float)
        
        self.last_confirmed_target_pos = pos
        self.last_confirmed_target_vel = vel.tolist() if vel is not None else None

    def get_guidance_target(self):
        return self.last_confirmed_target_pos

    def get_guidance_velocity(self):
        return self.last_confirmed_target_vel

    def _state_to_position(self, state, ref):        
        enu = state[:3]
        lat, lon, alt = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)        
        return Position(lat, lon, alt)
    
    def _maybe_seed_from_shooter(self):
        if self._seeded:
            return
        enu = getattr(self.missile, "initial_tracked_position_enu", None)
        ref = getattr(self.missile, "tracker_reference_pos", None)
        vel = getattr(self.missile, "initial_tracked_velocity_enu", None)
        if enu is not None and ref is not None:
            lat, lon, alt = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)
            self.last_confirmed_target_pos = Position(lat, lon, alt)
            self.last_confirmed_target_vel = vel
        self._seeded = True