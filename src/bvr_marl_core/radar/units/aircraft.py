from bvr_marl_core.domain.sensing_visibility import sensible_hostiles
from bvr_marl_core.radar.lock.aircraft import AircraftLockController
from bvr_marl_core.radar.radar import Radar


def radar_target_candidates(sim, parent):
    """Hostile units this radar may attempt to detect.

    Sensor-invisible units (AWACS and other pure support assets) are dropped HERE, at
    enumeration, before any per-target work happens. That is the point of doing it here
    rather than filtering detections or tracks later: everything downstream -- RCS
    lookup, SNR, line-of-sight, false-alarm merging, association, track maintenance --
    is per-candidate, so a unit removed at this line costs nothing for the rest of the
    tick.

    One definition for both radar entry points. They used to build the same list
    independently, which is exactly how the launch gate and the feasibility gate came to
    disagree about rules of engagement. The hostility+invisibility half of that rule now
    lives in `domain.sensing_visibility` so the RWR and seeker paths share it.
    """
    parent_id = getattr(parent, "id", None)
    parent_group = getattr(parent, "group", None)
    return [
        unit
        for unit in sensible_hostiles(sim.active_units.values(), parent_group)
        if unit.id != parent_id
    ]


class AircraftRadar(Radar):
    def __init__(self, *args, lock_ctrl=None, data_link=None, owner=None, **kwargs):
        lock_ctrl = lock_ctrl or AircraftLockController()
        # Aircraft radars are susceptible to jamming
        kwargs.setdefault("jam_susceptible", True)
        kwargs.setdefault("use_beam_steering", True)
        super().__init__(*args, lock_ctrl=lock_ctrl, data_link=data_link, owner=owner, **kwargs)
        self.data_link = data_link
        self.owner = owner
        self.locked_targets = set()

    def update(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None
    ):
        # Call parent update (which calls _update_lock_ctrl)
        tracks = super().update(
            tick_secs, sim, targets, owner_position, steer_h, steer_p, group_radars
        )
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
        for track in tracks:
            # Skip contacts classified as missiles without consulting a target object.
            if "missile" in track.classification:
                continue

            # If there's no engagement_id and track isn't engageable, skip
            if track.emitter_hypothesis_id is None and not track.engageable:
                continue

            detected_target_ids.append(track.track_id)

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
        targets = radar_target_candidates(sim, parent)

        tracks = self.update(
            tick_secs,
            sim,
            targets=targets,
            owner_position=owner_position,
            steer_h=steer_h,
            steer_p=steer_p,
        )
        return tracks

    def stage_reports_for_sensors(self, tick_secs, sim, owner_position, steer_h=0.0, steer_p=0.0):
        parent = getattr(self, "owner", None)
        if parent is None:
            self.cached_detections = ()
            return ()
        targets = radar_target_candidates(sim, parent)
        return self.stage_reports(
            tick_secs,
            sim,
            targets,
            owner_position,
            steer_h,
            steer_p,
        )

    def update_from_staged_sensor_reports(self, tick_secs, sim, owner_position):
        tracks = self.update_from_staged_reports(tick_secs, sim, owner_position)
        self.locked_targets = set(self.lock_ctrl.locked_target_ids())
        return tracks
