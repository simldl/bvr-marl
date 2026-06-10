from bvr_marl_core.radar.lock.base import BaseLockController


class AircraftLockController(BaseLockController):
    def __init__(self):
        super().__init__()

    def get_mode(self):
        return "search"

    def is_locked(self, tid):
        return self.is_confirmed(tid)

    def get_all_locked(self):
        return self.locked_target_ids()
