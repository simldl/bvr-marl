from missiles.missile import Missile
import numpy as np

class Fox2Missile(Missile):
    def __init__(self, name: str, firing_time_s, target, source, map_limits, group, config, data_link_mode="none"):
        super().__init__(name, firing_time_s, target, source, map_limits, group, config, data_link_mode)
        self.fox_type = 2
        self.requires_illumination = False
        self.parent_radar_link = False
        
    def _init_radar(self, rc: dict, data_link_mode: str):
        from missiles.fox2.ir_seeker import InfraredSeeker
        return InfraredSeeker(
            fov_deg=rc.get("fov_deg", 20.0),
            max_range_m=rc.get("max_range_m", 18000.0),
            sensitivity=rc.get("sensitivity", 1.0),
            owner=self
        )