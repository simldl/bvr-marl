import numpy as np


class MissileWarner:
    def __init__(self, parent, detection_delay_s: float = 1.0, detection_delay_std: float = 0.5):

        self.parent = parent
        self.detection_delay_s = detection_delay_s
        self.detection_delay_std = detection_delay_std
        self._pending_warnings = []
        self._active_warnings = set()

    def check_for_new_missiles(self, sim_time: float, sim):
        """Check for new missile threats and clean up stale warnings."""
        active_missile_ids = set()

        for unit in getattr(sim, "active_units", {}).values():
            if getattr(unit, "is_missile", False) and getattr(unit, "target", None) is self.parent:
                active_missile_ids.add(unit.id)

                if unit.id not in self._active_warnings and not self._is_pending(unit.id):
                    delay = float(
                        max(
                            np.random.normal(self.detection_delay_s, self.detection_delay_std), 0.05
                        )
                    )
                    self._pending_warnings.append((float(sim_time + delay), unit.id, unit))

        self._active_warnings = self._active_warnings.intersection(active_missile_ids)
        self._pending_warnings = [
            (wt, mid, m) for (wt, mid, m) in self._pending_warnings if mid in active_missile_ids
        ]

    def _is_pending(self, missile_id):
        return any(mid == missile_id for (_, mid, _) in self._pending_warnings)

    def update(self, sim_time: float):
        """Process pending warnings that are now due."""
        due = [m for (wt, mid, m) in self._pending_warnings if sim_time >= wt]
        self._pending_warnings = [
            (wt, mid, m) for (wt, mid, m) in self._pending_warnings if sim_time < wt
        ]
        for missile in due:
            self._active_warnings.add(missile.id)
        return due

    def get_current_warning_ids(self):
        """Get list of currently active warning IDs."""
        return list(self._active_warnings)

    def get_warning_count(self):
        """Get count of active warnings."""
        return len(self._active_warnings)
