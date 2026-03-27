class BaseLockController:
    """Base Lock Controller."""

    def __init__(self):
        # Dict: target_id -> [n_confirm, n_miss]
        self._lock_state = {}
        self._confirm_needed = 2
        self._miss_needed = 5

    def update_locks(self, detected_target_ids):

        detected_set = set(detected_target_ids)
        for tid in list(self._lock_state.keys()):
            if tid in detected_set:
                self._lock_state[tid][0] += 1
                self._lock_state[tid][1] = 0
            else:
                self._lock_state[tid][1] += 1
        for tid in detected_target_ids:
            if tid not in self._lock_state:
                self._lock_state[tid] = [1, 0]
        for tid in list(self._lock_state.keys()):
            if self._lock_state[tid][1] >= self._miss_needed:
                del self._lock_state[tid]

    def is_confirmed(self, tid):
        entry = self._lock_state.get(tid)
        return (
            entry is not None and entry[0] >= self._confirm_needed and entry[1] < self._miss_needed
        )

    def locked_target_ids(self):
        return [
            tid
            for tid, (nconf, nmiss) in self._lock_state.items()
            if nconf >= self._confirm_needed and nmiss < self._miss_needed
        ]

    def clear(self):
        self._lock_state.clear()
