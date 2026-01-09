#!/usr/bin/env python3
"""
Mock missile classes for testing.
Provides lightweight mock implementations of missile components.
"""

import numpy as np
import math
from typing import Optional, Dict, List, Any


class MockMissileRadar:
    """Mock missile radar system."""
    def __init__(self):
        self.max_range_m = 75000.0
        self.h_fov_deg = 30.0
        self.v_fov_deg = 30.0
        self.locked_target_id = None
        self.tracking_mode = "search"
        
        # Lock control
        self.lock_ctrl = type('LockCtrl', (), {
            'set_mode': self._set_mode
        })()
        
    def _set_mode(self, mode, target_id=None):
        self.tracking_mode = mode
        if mode == "track" and target_id:
            self.locked_target_id = target_id
            
    def has_lock(self, target_id):
        return self.locked_target_id == target_id
        
    def get_locked_target_id(self):
        return self.locked_target_id


class MockMissileGuidance:
    """Mock missile guidance system."""
    def __init__(self, missile):
        self.missile = missile
        self.guidance_mode = "proportional_navigation"
        self.last_commands = {"yaw": 0.0, "pitch": 0.0}
        
    def compute(self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs):
        """Mock guidance computation."""
        if not target_position:
            return current_yaw_deg, current_pitch_deg
            
        # Simple mock guidance - point toward target
        try:
            # Calculate bearing to target
            dlat = target_position.lat - missile_position.lat
            dlon = target_position.lon - missile_position.lon
            
            target_bearing = math.degrees(math.atan2(dlon, dlat))
            target_elevation = math.degrees(math.atan2(
                target_position.alt - missile_position.alt,
                math.sqrt(dlat**2 + dlon**2) * 111000  # Rough conversion to meters
            ))
            
            # Gradually steer toward target
            yaw_error = target_bearing - current_yaw_deg
            if yaw_error > 180:
                yaw_error -= 360
            elif yaw_error < -180:
                yaw_error += 360
                
            pitch_error = target_elevation - current_pitch_deg
            
            # Apply proportional control
            new_yaw = current_yaw_deg + yaw_error * 0.1
            new_pitch = current_pitch_deg + pitch_error * 0.1
            
            # Store commands
            self.last_commands = {"yaw": new_yaw, "pitch": new_pitch}
            
            return new_yaw, new_pitch
            
        except Exception:
            return current_yaw_deg, current_pitch_deg


class MockMissileEngine:
    """Mock missile engine."""
    def __init__(self):
        self.thrust_n = 15000.0
        self.burn_time_s = 30.0
        self.fuel_remaining = 1.0
        self.is_burning = True
        self.start_time = 0.0
        
    def update(self, current_time, dt):
        """Update engine state."""
        burn_duration = current_time - self.start_time
        if burn_duration > self.burn_time_s:
            self.is_burning = False
            self.fuel_remaining = 0.0
        else:
            self.fuel_remaining = 1.0 - (burn_duration / self.burn_time_s)
            
    def get_thrust(self):
        return self.thrust_n if self.is_burning else 0.0
        
    def is_fuel_depleted(self):
        return self.fuel_remaining <= 0.0


class MockMissilePhysics:
    """Mock missile physics."""
    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.call_count = 0
        self.last_call_params = None
        
    def _default_params(self):
        """Default missile physics parameters."""
        return type('Params', (), {
            'mass_kg': 180,
            'reference_area_m2': 0.2,
            'max_speed_mps': 1200,
            'max_g': 50.0,
            'n_max': 50.0  # Same as max_g, for backwards compatibility
        })()
        
    def compute_movement(self, position, yaw_deg, pitch_deg, speed, thrust, dt):
        """Mock missile movement computation."""
        self.call_count += 1
        self.last_call_params = {
            'position': position,
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'speed': speed,
            'thrust': thrust,
            'dt': dt
        }
        
        # Simple ballistic motion with thrust
        new_speed = min(speed + thrust * dt / self.params.mass_kg, self.params.max_speed_mps)
        
        # Move in direction of current attitude
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        velocity_x = new_speed * math.cos(pitch_rad) * math.sin(yaw_rad)  # East
        velocity_y = new_speed * math.cos(pitch_rad) * math.cos(yaw_rad)  # North
        velocity_z = new_speed * math.sin(pitch_rad)  # Up
        
        new_lat = position.lat + velocity_y * dt / 111000  # Rough conversion
        new_lon = position.lon + velocity_x * dt / 111000
        new_alt = position.alt + velocity_z * dt
        
        return new_lat, new_lon, new_alt, new_speed


class MockMissile:
    """Mock missile with all major systems."""
    def __init__(self, firing_time_s=0.0, target=None, source=None, map_limits=None, group="BLUE", **kwargs):
        self.id = f"missile_{np.random.randint(1000, 9999)}"
        self.name = f"Mock Missile {self.id}"
        self.type = "Missile"
        self.is_missile = True
        
        # Basic properties
        self.firing_time_s = firing_time_s
        self.target = target
        self.source = source
        self.map_limits = map_limits
        self.group = group
        
        # Position and motion
        if source and hasattr(source, 'position'):
            self.position = MockPosition(source.position.lat, source.position.lon, source.position.alt)
        else:
            self.position = MockPosition()
            
        self.yaw_deg = source.yaw_deg if source and hasattr(source, 'yaw_deg') else 0.0
        self.pitch_deg = 0.0
        self.speed = 300.0  # Initial speed
        
        # Missile characteristics
        self.max_range_m = 75000.0
        self.min_range_m = 1500.0
        self.fox_type = 3  # Active radar homing
        
        # Tracking and guidance
        self.designated_target_id = target.id if target else None
        self.retarget_policy = "locked_override"
        self.initial_tracked_position_enu = None
        self.initial_tracked_velocity_enu = None
        self.tracker_reference_pos = None
        self.tracking_lock = False
        
        # Systems
        self.radar = MockMissileRadar()
        self.guidance = MockMissileGuidance(self)
        self.engine = MockMissileEngine()
        self.physics = MockMissilePhysics()
        
        # State
        self.is_active = True
        self.hit_target = False
        self.fuel_depleted = False
        self.time_of_flight = 0.0
        
    def update(self, tick_secs, sim):
        """Mock missile update."""
        self.time_of_flight += tick_secs
        current_time = sim.utc_time if hasattr(sim, 'utc_time') else self.time_of_flight
        
        # Update engine
        self.engine.update(current_time, tick_secs)
        self.fuel_depleted = self.engine.is_fuel_depleted()
        
        # Update guidance if we have a target
        if self.target and hasattr(self.target, 'position'):
            new_yaw, new_pitch = self.guidance.compute(
                self.yaw_deg, self.pitch_deg,
                self.position, self.target.position,
                tick_secs
            )
            self.yaw_deg = new_yaw
            self.pitch_deg = new_pitch
            
        # Update physics
        thrust = self.engine.get_thrust()
        result = self.physics.compute_movement(
            self.position, self.yaw_deg, self.pitch_deg, 
            self.speed, thrust, tick_secs
        )
        
        # Unpack physics results
        self.position.lat, self.position.lon, self.position.alt, self.speed = result
        
        # Check for intercept
        if self.target and hasattr(self.target, 'position'):
            distance = self._distance_to_target()
            if distance < 50.0:  # 50m proximity fuse
                self.hit_target = True
                self.is_active = False
                return [type('Event', (), {'type': 'missile_hit', 'missile': self, 'target': self.target})()]
                
        # Check for fuel depletion
        if self.fuel_depleted:
            self.is_active = False
            
        # Check for range limits
        if self.source and hasattr(self.source, 'position'):
            range_to_source = self._distance_to(self.source.position) * 1000  # Convert to meters
            if range_to_source > self.max_range_m:
                self.is_active = False
                
        return []
        
    def substep_update(self, dt, sim):
        """Mock substep update."""
        return self.update(dt, sim)
        
    def _distance_to_target(self):
        """Calculate distance to target in meters."""
        if not self.target or not hasattr(self.target, 'position'):
            return float('inf')
        return self._distance_to(self.target.position) * 1000  # Convert to meters
        
    def _distance_to(self, other_position):
        """Calculate distance to another position."""
        return math.sqrt(
            (self.position.lat - other_position.lat) ** 2 +
            (self.position.lon - other_position.lon) ** 2 +
            (self.position.alt - other_position.alt) ** 2
        )
        
    def get_state_representation(self):
        """Mock missile state representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "position": {
                "lat": self.position.lat,
                "lon": self.position.lon,
                "alt": self.position.alt
            },
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "speed": self.speed,
            "target_id": self.designated_target_id,
            "time_of_flight": self.time_of_flight,
            "fuel_remaining": self.engine.fuel_remaining,
            "is_active": self.is_active,
            "hit_target": self.hit_target,
            "tracking_lock": self.tracking_lock
        }


class MockPosition:
    """Mock position class (duplicated here for missile module independence)."""
    def __init__(self, lat=0.0, lon=0.0, alt=1000.0):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        
    def copy(self):
        return MockPosition(self.lat, self.lon, self.alt)
        
    def distance_to(self, other):
        return math.sqrt(
            (self.lat - other.lat) ** 2 +
            (self.lon - other.lon) ** 2 +
            (self.alt - other.alt) ** 2
        )


# Common missile types for testing
class MockAIM120(MockMissile):
    """Mock AIM-120 AMRAAM."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AIM-120 AMRAAM"
        self.max_range_m = 105000.0
        self.min_range_m = 2000.0
        self.fox_type = 3
        self.speed = 1200.0
        

class MockAIM9(MockMissile):
    """Mock AIM-9 Sidewinder."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AIM-9 Sidewinder"
        self.max_range_m = 35000.0
        self.min_range_m = 1000.0
        self.fox_type = 2
        self.speed = 850.0


class MockAIM7(MockMissile):
    """Mock AIM-7 Sparrow."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AIM-7 Sparrow"
        self.max_range_m = 50000.0
        self.min_range_m = 1500.0
        self.fox_type = 1
        self.speed = 1200.0