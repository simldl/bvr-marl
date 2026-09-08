from bvr_marl_core.missiles.missile import Missile


class Fox1Missile(Missile):
    def __init__(
        self,
        name: str,
        firing_time_s,
        target,
        source,
        map_limits,
        group,
        config,
        data_link_mode="illumination",
    ):
        super().__init__(
            name, firing_time_s, target, source, map_limits, group, config, data_link_mode
        )
        self.fox_type = 1
        self.requires_illumination = True
        self.parent_radar_link = True

    def _init_radar(self, rc: dict, data_link_mode: str):
        from bvr_marl_core.missiles.fox1.sarh_seeker import SemiActiveRadarSeeker

        return SemiActiveRadarSeeker(
            fov_deg=rc.get("fov_deg", 30.0),
            max_range_m=rc.get("max_range_m", 50000.0),
            sensitivity=rc.get("sensitivity", 1.0),
            owner=self,
            parent_aircraft=self.source,
        )

    def update(self, tick_secs: float, sim) -> list:
        if not self._parent_has_target_lock(sim):
            self.radar.lose_lock()

        return super().update(tick_secs, sim)

    def _parent_has_target_lock(self, sim) -> bool:
        if not hasattr(self.source, "radar"):
            return False

        if hasattr(self.source.radar, "get_locked_targets"):
            locked_ids = set(self.source.radar.get_locked_targets() or [])
            return (self.target is not None) and (getattr(self.target, "id", None) in locked_ids)

        get_one = getattr(self.source.radar, "get_locked_target", None)
        if callable(get_one):
            one_id = get_one()
            return (self.target is not None) and (getattr(self.target, "id", None) == one_id)

        return False
