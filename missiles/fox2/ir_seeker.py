import numpy as np
from simulator.utils.geodesics import geodetic_bearing_deg
from simulator.utils.angles import signed_yaw_deg_diff

class InfraredSeeker:
    def __init__(self, fov_deg, max_range_m, sensitivity, owner):
        self.fov_deg = fov_deg
        self.max_range_m = max_range_m
        self.sensitivity = sensitivity
        self.owner = owner
        self.locked_target = None
        self.target_provider = None
        
    def update(self, tick_secs, sim, targets=None, owner_position=None):
        if targets and len(targets) > 0:
            target = targets[0]
            if self._can_detect_target(target, owner_position or self.owner.position):
                self.locked_target = target
                self._update_tracker_info(target)
            else:
                self.locked_target = None
    
    def _can_detect_target(self, target, missile_pos) -> bool:
        distance = self._calculate_distance(missile_pos, target.position)
        
        if distance > self.max_range_m:
            return False
            
        bearing_to_target = geodetic_bearing_deg(
            missile_pos.lat, missile_pos.lon,
            target.position.lat, target.position.lon
        )
        relative_bearing = abs(signed_yaw_deg_diff(self.owner.yaw_deg, bearing_to_target))
        
        if relative_bearing > self.fov_deg / 2.0:
            return False
            
        ir_signature = self._get_ir_signature(target, missile_pos)
        detection_threshold = 0.3
        
        return ir_signature > detection_threshold
    
    def _get_ir_signature(self, target, missile_pos) -> float:
        distance = self._calculate_distance(missile_pos, target.position)
        
        base_signature = 1.0
        distance_factor = max(0.1, self.max_range_m / max(distance, 100.0))
        
        aspect_angle = self._get_aspect_angle(target, missile_pos)
        if aspect_angle < 45:  # Rear aspect
            aspect_factor = 2.0
        elif aspect_angle < 135:  # Side aspect
            aspect_factor = 1.0
        else:  # Front aspect
            aspect_factor = 0.5
            
        speed_factor = 1.0 + (getattr(target, 'speed', 250) / 1000.0)
        
        return base_signature * distance_factor * aspect_factor * speed_factor * self.sensitivity
    
    def _get_aspect_angle(self, target, missile_pos) -> float:
        # Bearing from TARGET to MISSILE
        bearing_tgt_to_msl = geodetic_bearing_deg(
            target.position.lat, target.position.lon,
            missile_pos.lat, missile_pos.lon
        )
        # Tail axis = heading + 180°
        tail_heading = (getattr(target, 'yaw_deg', 0.0) + 180.0) % 360.0
        return abs(signed_yaw_deg_diff(tail_heading, bearing_tgt_to_msl))
    
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
        """Return empty list - Fox 2 missiles don't generate tracks"""
        return []