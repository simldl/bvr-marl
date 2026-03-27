from air_to_air_rl.radar.lock.base import BaseLockController


class MissileLockController(BaseLockController):
    def __init__(self):
        super().__init__()
        self._active_target = None

    def update_locks(self, detected_target_ids):
        super().update_locks(detected_target_ids)
        locked = self.locked_target_ids()
        if self._active_target not in locked:
            self._active_target = locked[0] if locked else None

    def get_mode(self):
        return "track" if self._active_target else "search"

    def get_locked(self):
        return self._active_target

    def is_locked(self):
        return self._active_target is not None

    def force_active(self, tid):
        """Set active target explicitly (called by radar after score+hysteresis selection).
        No-op if tid is not currently confirmed."""
        if self.is_confirmed(tid):
            self._active_target = tid

    def set_mode(self, mode, target_id=None):
        if mode == "track" and target_id is not None:
            self._active_target = target_id
            self._lock_state = {target_id: [self._confirm_needed, 0]}
        elif mode == "search":
            self._active_target = None
            self._lock_state.clear()
