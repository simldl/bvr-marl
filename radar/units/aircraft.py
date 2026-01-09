from radar.radar import Radar
from radar.lock.aircraft import AircraftLockController

class AircraftRadar(Radar):

    def __init__(
        self,
        *args,
        lock_ctrl=None,
        data_link=None,
        owner=None,
        **kwargs
    ):
        lock_ctrl = lock_ctrl or AircraftLockController()
        # Aircraft radars are susceptible to jamming
        kwargs.setdefault('jam_susceptible', True)
        super().__init__(*args, lock_ctrl=lock_ctrl, data_link=data_link, owner=owner, **kwargs)
        self.data_link = data_link
        self.owner = owner
        self.locked_targets = set()

    def update(self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None):
        # Call parent update (which calls _update_lock_ctrl)
        tracks = super().update(tick_secs, sim, targets, owner_position, steer_h, steer_p, group_radars)
        # Sync locked targets from lock controller (set by _update_lock_ctrl override)
        self.locked_targets = set(self.lock_ctrl.locked_target_ids())
        return tracks

    def _update_lock_ctrl(self, tracks):
        """
        Use engagement_id priority and allow HOJ:
        - If a track is not 'engageable' but has an engagement_id (jammer),
        we still consider it for locking (HOJ).
        - Still skip missiles as lock candidates.
        """
        detected_target_ids = []
        for tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count, is_deception, suspect_deception, engagement_id, jammer_id, engageable in tracks:
            # Skip missiles
            if tgt is not None and getattr(tgt, "is_missile", False):
                continue

            # If there's no engagement_id and track isn't engageable, skip
            if (engagement_id is None) and (not engageable):
                continue

            det_id = engagement_id if engagement_id is not None else getattr(tgt, "id", None)
            if det_id is not None:
                detected_target_ids.append(det_id)

        self.lock_ctrl.update_locks(detected_target_ids)

    def has_radar_lock(self, target):
        return hasattr(target, "id") and target.id in self.locked_targets

    def get_locked_targets(self):
        return set(self.locked_targets)

    def get_locked_target(self):
        owner_tgt_id = getattr(getattr(self.owner, "target", None), "id", None)
        if owner_tgt_id in self.locked_targets:
            return owner_tgt_id
        return next(iter(self.locked_targets), None)
    
    def update_for_sensors(self, tick_secs, sim, owner_position, steer_h=0.0, steer_p=0.0):
        parent = getattr(self, "owner", None)
        if parent is None:
            return []
        targets = [
            u for u in sim.active_units.values()
            if u.id != parent.id and u.group != parent.group
        ]

        tracks = self.update(
            tick_secs,
            sim,
            targets=targets,
            owner_position=owner_position,
            steer_h=steer_h,
            steer_p=steer_p,
        )
        return tracks