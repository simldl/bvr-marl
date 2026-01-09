import numpy as np
from radar.radar import Radar
from radar.lock.missile import MissileLockController
from simulator.utils.geodesics import geodetic_distance_km
from radar.core.utils import _angles_dist
from simulator.utils.angles import yaw_geo_to_math, signed_yaw_deg_diff

class MissileRadar(Radar):
    def __init__(
        self, *args, lock_ctrl=None, owner=None,
        target_provider=None, initial_datalink_mode=None,
        original_data_link_mode=None, **kwargs
    ):
        lock_ctrl = lock_ctrl or MissileLockController()
        # Missile seekers are NOT susceptible to jamming
        kwargs.setdefault('jam_susceptible', False)
        super().__init__(*args, lock_ctrl=lock_ctrl, owner=owner, **kwargs)

        # Store references
        self.owner = owner
        self.target_provider = target_provider
        # If no explicit initial mode was provided, mirror 'original'
        if initial_datalink_mode is None:
            initial_datalink_mode = original_data_link_mode

        # If a data_link component exists, apply initial mode
        try:
            if hasattr(self, 'data_link') and self.data_link is not None and initial_datalink_mode:
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

    def update(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None
    ):
        self._manage_datalink_mode()
        self.param_policy.update_dynamic_params()
        self._update_beam_steering(tick_secs, steer_h, steer_p)

        own_detections = self._generate_and_transform_detections(owner_position, targets)
        self.cached_detections = own_detections
        if group_radars is None and sim is not None:
            group_radars = self.get_default_group_radars(sim)
        fused_measurements = self._get_fused_measurements(owner_position, group_radars, own_detections)
        filtered_measurements = self._filter_measurements_for_lock(fused_measurements)
        self.cached_tracks = self.tracker_manager.update_tracks(filtered_measurements, tick_secs, owner_position)
        self._update_lock_ctrl(self.cached_tracks)
        return self.cached_tracks

    def _manage_datalink_mode(self):
        if not self._has_switched:
            if self.should_activate_own_radar():
                self.data_link.set_mode(self._original_data_link_mode)
                self._has_switched = True
            else:
                self.data_link.set_mode("other")
        elif self.data_link.get_mode() == "other":
            self.data_link.set_mode(self._original_data_link_mode)

    def _get_fused_measurements(self, owner_position, group_radars, own_detections):
        if self.data_link is not None:
            return self.data_link.get_fused_clusters(
                own_radar=self,
                group_radars=group_radars,
                own_position=owner_position,
                clusterer=self.clusterer
            )
        else:
            return self.clusterer.cluster(own_detections)

    def _filter_measurements_for_lock(self, fused_measurements):
        # Filter out suspected deception (ghosts) - missile seekers prioritize real targets
        valid_clusters = [c for c in fused_measurements
                         if c.get('T', None) is not None
                         and not c.get('is_deception', False)
                         and not c.get('suspect_deception', False)
                         and not getattr(c.get('T', None), 'is_missile', False)
                         and not getattr(c.get('T', None), 'is_countermeasure', False)]
        
        if not valid_clusters:
            return []
        
        # If we have a designated lock target, prioritize it but keep tracking others too
        if self.locked_target is not None:
            designated_cluster = [c for c in valid_clusters 
                                 if getattr(c.get('T', None), "id", None) == self.locked_target]
            if designated_cluster:
                # Return designated target first, then other valid targets
                other_clusters = [c for c in valid_clusters 
                                if getattr(c.get('T', None), "id", None) != self.locked_target]
                return designated_cluster + other_clusters
        
        # No designated target or it's not visible - return all valid clusters
        # Select the best one as the primary lock
        if valid_clusters:
            best = max(valid_clusters, key=lambda c: (c['n_obs'], getattr(c.get('T', None), "rcs", 0)))
            self.locked_target = getattr(best.get('T', None), "id", None)
        
        return valid_clusters

    def _update_lock_ctrl(self, tracks):
        detected_target_ids = set()
        for tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count, is_deception, suspect_deception, engagement_id, jammer_id, engageable in tracks:
            # Missile seekers only lock onto engageable targets (no ghosts/deception)
            if engageable and tgt is not None and hasattr(tgt, "id") and not getattr(tgt, "is_missile", False):
                detected_target_ids.add(tgt.id)
        self.lock_ctrl.update_locks(detected_target_ids)
        self.locked_target = self.lock_ctrl.get_locked()
        self.mode = self.lock_ctrl.get_mode()

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
        distance = geodetic_distance_km(
            missile_pos.lat, missile_pos.lon, missile_pos.alt,
            target_pos.lat, target_pos.lon, target_pos.alt
        ) * 1000.0
        in_terminal = (
            hasattr(self.owner, "phase_manager") and
            getattr(self.owner.phase_manager, "current_phase", "") == "terminal"
        )
        return distance < self.max_range_m or in_terminal
    
    def get_tracker_info(self):
        locked_id = self.get_locked_target()
        for tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count, is_deception, suspect_deception, engagement_id, jammer_id, engageable in self.get_tracks():
            if tgt is not None and hasattr(tgt, "id") and str(tgt.id) == str(locked_id) and engageable:
                tracker = self.tracker_manager.tracks.get(tid, None)
                velocity = self._extract_velocity(state, tracker)
                position = self._state_to_position(state, ref if ref is not None else self.owner.position)
                return {
                    "tid": tid,
                    "state": state,
                    "cov": cov,
                    "tgt": tgt,
                    "ref": ref,
                    "confidence": confidence,
                    "n_obs": n_obs,
                    "lifetime": lifetime,
                    "update_count": update_count,
                    "tracker": tracker,
                    "velocity": velocity,
                    "position": position,
                    # ECM fields
                    "is_deception": is_deception,
                    "suspect_deception": suspect_deception,
                    "engagement_id": engagement_id,
                    "jammer_id": jammer_id,
                    "engageable": engageable
                }
        return None
    
    def _extract_velocity(self, state, tracker=None):
        # Falls Tracker/Filter eine eigene get_velocity() Methode hat, nutze sie
        if tracker is not None and hasattr(tracker, "get_velocity"):
            return tracker.get_velocity()
        # Standardannahme: state[3:6] = vx, vy, vz
        if len(state) >= 6:
            return state[3:6]
        return np.zeros(3)

    def _state_to_position(self, state, ref):
        from radar.core.utils import enu_to_geodetic
        from simulator.core.helpers import Position
        enu = state[:3]
        lat, lon, alt = enu_to_geodetic(enu, ref.lat, ref.lon, ref.alt)
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
                self.yaw_offset_deg  += max(-max_step, min(max_step, -getattr(self, "yaw_offset_deg", 0.0)))
                self.pitch_offset_deg += max(-max_step, min(max_step, -getattr(self, "pitch_offset_deg", 0.0)))
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