from aircrafts.systems.passive_radar import PassiveRadar
from aircrafts.core.nez import NoEscapeZoneCalculator
from aircrafts.systems.missile_warner import MissileWarner
from aircrafts.core.target_prio import TrackPrioritySystem
from simulator.core.helpers import units_distance_km

class AircraftSensorSystem:
    """
    Orchestrates all onboard sensor systems:
    - Radar (detection & tracking)
    - Passive radar observations
    - Track prioritization
    - No-Escape Zone (active & passive)
    - Missile warning
    """

    def __init__(self, parent):
        self.parent = parent

        # 1) Radar subsystem
        if hasattr(parent, 'radar') and parent.radar is not None:
            self.radar = parent.radar
        else:
            self.radar = None

        # State containers
        self.sensor_tracks = []
        self.prioritized_tracks = []
        self.active_nez = {}
        self.passive_nez = {}
        self.warnings = []

        # 2) Passive radar
        conf = getattr(parent, 'config', {}) or {}
        ang_err = conf.get('passive_radar_angular_error_deg', getattr(parent, 'passive_radar_angular_error_deg', 5.0))
        rng_err = conf.get('passive_radar_range_error_m',     getattr(parent, 'passive_radar_range_error_m', 2000.0))
        max_age = conf.get('passive_radar_max_age_s',         getattr(parent, 'passive_radar_max_age_s', 3.0))

        self.passive_radar = PassiveRadar(
            angular_error_deg=float(ang_err),
            range_error_m=float(rng_err),
            max_age_s=float(max_age),
        )

        # 3) No-Escape Zone calculator
        self.nez_calc = NoEscapeZoneCalculator(parent)

        # 4) Missile warner
        mw_delay = conf.get('missile_warning_delay_s', 0.0)
        mw_std   = conf.get('missile_warning_delay_std', 0.0)
        self.missile_warner = MissileWarner(
            parent,
            detection_delay_s=mw_delay,
            detection_delay_std=mw_std
        )

        # 5) Track prioritizer
        self.track_prioritizer = TrackPrioritySystem(parent)

    def update_sensor_data(self, sim, tick_secs):
        sim_time = getattr(sim, 'elapsed_time_s', None)

        if self.radar:
            self.sensor_tracks = self.radar.update_for_sensors(
                tick_secs,
                sim,
                owner_position=self.parent.position,
                steer_h=getattr(self.parent, "beam_rate_hz", 5.0),
                steer_p=getattr(self.parent, "beam_rate_p_hz", 3.0),
            )
        else:
            self.sensor_tracks = []

        for unit in sim.active_units.values():
            if unit is not self.parent and hasattr(unit, 'emit_radar') and unit.group != self.parent.group:
                _ = self.passive_radar.receive_emission(unit, self.parent, sim.elapsed_time_s)

        # OPTIMIZATION: Reuse dictionaries and cache NEZ calculations
        enemies = [u for u in sim.active_units.values() if u is not self.parent]

        # Initialize dicts on first call
        if not hasattr(self, '_last_nez_positions'):
            self.active_nez = {}
            self.passive_nez = {}
            self._last_nez_positions = {}  # Track positions for cache invalidation

        # Remove entries for units that no longer exist
        current_enemy_ids = {u.id for u in enemies}
        if hasattr(self, 'active_nez'):
            self.active_nez = {k: v for k, v in self.active_nez.items() if k in current_enemy_ids}
            self.passive_nez = {k: v for k, v in self.passive_nez.items() if k in current_enemy_ids}
            self._last_nez_positions = {k: v for k, v in self._last_nez_positions.items() if k in current_enemy_ids}

        # Update NEZ only for units that moved significantly or are new
        for u in enemies:
            needs_update = False
            if u.id not in self.active_nez:
                needs_update = True
            elif u.id in self._last_nez_positions:
                # Check if position changed significantly (>200m or speed changed >50 m/s)
                last_pos = self._last_nez_positions[u.id]
                pos_changed = (abs(u.position.lat - last_pos[0]) > 0.002 or  # ~200m
                              abs(u.position.lon - last_pos[1]) > 0.002 or
                              abs(u.position.alt - last_pos[2]) > 200 or
                              abs(getattr(u, 'speed', 0) - last_pos[3]) > 50)
                needs_update = pos_changed
            else:
                needs_update = True

            if needs_update:
                self.active_nez[u.id] = self.nez_calc.active_nez(u)
                self.passive_nez[u.id] = self.nez_calc.passive_nez(u)
                self._last_nez_positions[u.id] = (u.position.lat, u.position.lon,
                                                   u.position.alt, getattr(u, 'speed', 0))

        if sim_time is not None:
            self.missile_warner.check_for_new_missiles(sim_time, sim)
            self.warnings = self.missile_warner.update(sim_time)
        else:
            self.warnings = []

        raw_tracks = [(state, cov) for tid, state, cov, *_ in getattr(self, 'sensor_tracks', [])]
        self.prioritized_tracks = self.track_prioritizer.prioritize(raw_tracks)

    def get_locked_targets(self):
        """Returns set of locked target_id(s) (Multi-Lock!)."""
        return self.radar.get_locked_targets() if self.radar else set()

    def has_radar_lock(self, target):
        """True if radar lock exists on target."""
        return self.radar.has_radar_lock(target) if self.radar else False

    def select_target(self, candidates, selection_value=None):
        """Returns selected target or None."""
        if not self.radar:
            return None
        locked_candidates = [t for t in candidates if self.has_radar_lock(t)]
        sorted_targets = (
            sorted(locked_candidates, key=lambda t: units_distance_km(self.parent, t))
            if locked_candidates
            else sorted(candidates, key=lambda t: units_distance_km(self.parent, t))
        )
        n = len(sorted_targets)
        if n == 0:
            return None
        selection_value = 0.0 if selection_value is None else max(0.0, min(selection_value, 1.0))
        index = int(selection_value * n)
        if index >= n:
            index = n - 1
        return sorted_targets[index]

    def get_prio_tracks(self):
        return list(self.prioritized_tracks)

    def get_active_nez(self):
        return dict(self.active_nez)

    def get_passive_nez(self):
        return dict(self.passive_nez)

    def get_missile_warnings(self):
        return list(self.warnings)

    def get_nez_features(self, sim, selected_target_id):
        """
        Get NEZ features for selected target.

        Returns dict with:
        - active_nez: scalar (backward compat)
        - active_dlz: full DLZ object for the selected target (or None)
        - passive_nez: dict {target_id: scalar}
        """
        # Find selected target
        selected_target = None
        for unit in sim.active_units.values():
            if unit.id == selected_target_id:
                selected_target = unit
                break

        # Compute full DLZ for selected target
        active_dlz = None
        if selected_target:
            try:
                active_dlz = self.nez_calc.compute_dlz(selected_target)
            except:
                pass

        # Backward-compatible scalar
        active_val = self.active_nez.get(selected_target_id, 0.0)

        recognized_ids = set(self.active_nez.keys()) | set(self.passive_nez.keys())
        passive_dict = {tid: self.passive_nez.get(tid, 0.0) for tid in recognized_ids}

        return {
            "active_nez": active_val,           # scalar (backward compat)
            "active_dlz": active_dlz,           # full DLZ object
            "passive_nez": passive_dict,        # dict of scalars
        }