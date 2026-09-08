import numpy as np

from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km

# A boosting missile's motor plume is a bright IR source detectable at long range
# and from any aspect (a spherical missile-approach/launch warner). After motor
# burnout the plume dies, so a missile is only newly *detected* while it is burning;
# once detected it is tracked for the rest of its (visible) life.
PLUME_DETECTION_RANGE_M = 120_000.0


class MissileWarner:
    """IR-plume-based missile launch / approach warner.

    Detects enemy missiles by their boost-phase motor plume (any aspect, within
    ``PLUME_DETECTION_RANGE_M``) rather than by ground-truth "is targeting me".
    A launch is therefore seen early, during boost, and gives a bearing; a detected
    missile is kept as an active warning while it remains alive even after burnout.
    """

    def __init__(self, parent, detection_delay_s: float = 1.0, detection_delay_std: float = 0.5):
        self.parent = parent
        self.detection_delay_s = detection_delay_s
        self.detection_delay_std = detection_delay_std
        self._pending_warnings = []
        self._active_warnings = set()

    def _plume_detectable(self, missile) -> bool:
        """True while the missile's motor is burning and it is within plume range."""
        engine = getattr(missile, "engine", None)
        burning = float(getattr(engine, "fuel_s", 0.0)) > 0.0
        if not burning:
            return False
        try:
            dist_m = (
                geodetic_distance_km(
                    self.parent.position.lat,
                    self.parent.position.lon,
                    self.parent.position.alt,
                    missile.position.lat,
                    missile.position.lon,
                    missile.position.alt,
                )
                * 1000.0
            )
        except Exception:
            return False
        return dist_m <= PLUME_DETECTION_RANGE_M

    def check_for_new_missiles(self, sim_time: float, sim):
        """Detect new burning enemy missiles by plume; prune warnings for dead ones."""
        alive_enemy_missiles = {
            u.id: u
            for u in getattr(sim, "active_units", {}).values()
            if getattr(u, "is_missile", False)
            and getattr(u, "group", None) != getattr(self.parent, "group", None)
        }

        for mid, missile in alive_enemy_missiles.items():
            if mid in self._active_warnings or self._is_pending(mid):
                continue
            if self._plume_detectable(missile):
                streams = getattr(sim, "random_streams", None)
                rng = (
                    streams.generator(
                        "missile_warning",
                        f"{getattr(self.parent, 'id', 0)}:{mid}",
                    )
                    if streams is not None
                    else np.random.default_rng(0)
                )
                delay = float(
                    max(rng.normal(self.detection_delay_s, self.detection_delay_std), 0.05)
                )
                self._pending_warnings.append((float(sim_time + delay), mid, missile))

        # Keep warnings only for still-alive enemy missiles (a tracked missile stays
        # warned after burnout; a dead/expired one drops off).
        self._active_warnings = self._active_warnings.intersection(alive_enemy_missiles)
        self._pending_warnings = [
            (wt, mid, m) for (wt, mid, m) in self._pending_warnings if mid in alive_enemy_missiles
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
