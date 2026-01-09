import numpy as np
from simulator.utils.geodesics import geodetic_bearing_deg
from simulator.utils.angles import signed_yaw_deg_diff

class SemiActiveRadarSeeker:
    def __init__(self, fov_deg, max_range_m, sensitivity, owner, parent_aircraft):
        self.fov_deg = fov_deg
        self.max_range_m = max_range_m
        self.sensitivity = sensitivity
        self.owner = owner
        self.parent_aircraft = parent_aircraft
        self.locked_target = None
        self.target_provider = None
        
    def update(self, tick_secs, sim, targets=None, owner_position=None):
        if not self._parent_illuminating_target():
            self.locked_target = None
            return
            
        if targets and len(targets) > 0:
            target = targets[0]
            if self._can_see_target(target, owner_position or self.owner.position):
                self.locked_target = target
                self._update_tracker_info(target)
            else:
                self.locked_target = None
    
    def _parent_illuminating_target(self) -> bool:
        if not hasattr(self.parent_aircraft, 'radar'):
            return False
        
        parent_locked_id = getattr(self.parent_aircraft.radar, 'get_locked_target', lambda: None)()
        return parent_locked_id is not None and parent_locked_id == self.owner.target.id
    
    def _can_see_target(self, target, missile_pos) -> bool:
        if not self._parent_illuminating_target():
            return False
        distance = self._calculate_distance(missile_pos, target.position)
        sigma = getattr(target, 'rcs', 0.5 * getattr(target, 'reference_area_m2', 1.0))
        eff_range = self.max_range_m * self._get_parent_range_scale(sigma_m2=float(sigma))
        if distance > eff_range:
            return False
            
        bearing_to_target = geodetic_bearing_deg(
            missile_pos.lat, missile_pos.lon,
            target.position.lat, target.position.lon
        )
        relative_bearing = abs(signed_yaw_deg_diff(self.owner.yaw_deg, bearing_to_target))
        
        return relative_bearing <= self.fov_deg / 2.0
    
    def _get_parent_range_scale(self, sigma_m2=1.0) -> float:
        if not hasattr(self.parent_aircraft, 'radar'):
            return 0.8  # conservative default
        pr = self.parent_aircraft.radar
        Pt  = float(getattr(pr, 'tx_power_w', 10_000.0))
        G   = 10.0 ** (float(getattr(pr, 'antenna_gain_db', 25.0)) / 10.0)
        # Reference terms (same as your previous normalization, but physics-aware)
        Pt0 = 10_000.0
        G0  = 10.0 ** (25.0 / 10.0)
        sigma0 = 1.0  # 1 m^2
        ratio = (Pt * (G**2) * sigma_m2) / (Pt0 * (G0**2) * sigma0)
        return float(np.clip(ratio ** 0.25, 0.5, 1.7))
    
    def _calculate_distance(self, pos1, pos2) -> float:
        R_earth = 6_371_000.0
        deg2rad = np.pi / 180.0
        
        dlat = (pos2.lat - pos1.lat) * deg2rad
        dlon = (pos2.lon - pos1.lon) * deg2rad
        lat_avg = 0.5 * (pos1.lat + pos2.lat) * deg2rad
        
        dx = R_earth * dlon * np.cos(lat_avg)
        dy = R_earth * dlat
        dz = pos2.alt - pos1.alt
        
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    
    def _update_tracker_info(self, target):
        self._tracker_info = {
            "position": [target.position.lat, target.position.lon, target.position.alt],
            "velocity": [
                getattr(target, 'velocity_x', 0.0),
                getattr(target, 'velocity_y', 0.0),
                getattr(target, 'velocity_z', 0.0)
            ]
        }
    
    def get_locked_target(self):
        return getattr(self.locked_target, 'id', None) if self.locked_target else None
    
    def get_tracker_info(self):
        return getattr(self, '_tracker_info', None)
    
    def lose_lock(self):
        self.locked_target = None
        
    def get_tracks(self):
        """Return empty list - Fox 1 missiles don't generate tracks"""
        return []