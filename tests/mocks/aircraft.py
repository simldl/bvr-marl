#!/usr/bin/env python3
"""
Mock aircraft classes for testing.
Provides lightweight mock implementations of aircraft components.
"""

import numpy as np
import math
from typing import Optional, Dict, List, Any


class MockPosition:
    """Mock position class."""
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


class MockVelocity:
    """Mock velocity class."""
    def __init__(self, vx=0.0, vy=0.0, vz=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz


class MockMapLimits:
    """Mock map limits class."""
    def __init__(self):
        self.bottom_lat = -90
        self.top_lat = 90
        self.left_lon = -180
        self.right_lon = 180
        self.min_alt = 0
        self.max_alt = 20000


class MockPhysics:
    """Mock physics engine for predictable testing."""
    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.call_count = 0
        self.last_call_params = None
        
    def _default_params(self):
        """Default physics parameters."""
        return type('Params', (), {
            'mass_kg': 15000,
            'reference_area_m2': 25,
            'aspect_ratio': 4.0,
            'oswald_e': 0.8,
            'max_speed_mps': 400,
            'n_max': 8.0,
            'stall0_mps': 60
        })()
        
    def compute_movement(self, position, current_yaw, desired_yaw, current_pitch, desired_pitch, speed, throttle, dt):
        """Mock movement computation with predictable results."""
        self.call_count += 1
        self.last_call_params = {
            'position': position,
            'current_yaw': current_yaw,
            'desired_yaw': desired_yaw,
            'current_pitch': current_pitch,
            'desired_pitch': desired_pitch,
            'speed': speed,
            'throttle': throttle,
            'dt': dt
        }
        
        # Simple mock physics - move forward and adjust attitude
        new_lat = position.lat + 0.001 * dt * math.cos(math.radians(current_yaw))
        new_lon = position.lon + 0.001 * dt * math.sin(math.radians(current_yaw))
        new_alt = position.alt + 100 * dt * math.sin(math.radians(current_pitch))
        
        # Gradually adjust attitude toward desired
        yaw_diff = desired_yaw - current_yaw
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360
        new_yaw = current_yaw + 0.1 * yaw_diff
        
        pitch_diff = desired_pitch - current_pitch
        new_pitch = current_pitch + 0.1 * pitch_diff
        
        # Adjust speed based on throttle
        new_speed = speed + (throttle - 0.5) * 20 * dt
        new_speed = max(50, min(new_speed, self.params.max_speed_mps))
        
        # Mock roll based on turn rate
        new_roll = yaw_diff * 0.1
        
        return new_lat, new_lon, new_alt, new_speed, new_yaw, new_pitch, new_roll


class MockRadar:
    """Mock radar system."""
    def __init__(self):
        self.h_fov_deg = 90.0
        self.v_fov_deg = 40.0
        self.max_range_m = 70000.0
        self.locked_targets = {}
        self.tracks = []
        
    def has_radar_lock(self, target):
        if target is None:
            return False
        return target.id in self.locked_targets
        
    def get_locked_targets(self):
        return self.locked_targets
        
    def update_for_sensors(self, tick_secs, sim, owner_position, steer_h=5.0, steer_p=3.0):
        """Mock sensor update."""
        return self.tracks


class MockSensor:
    """Mock sensor system."""
    def __init__(self, parent):
        self.parent = parent
        self.radar = parent.radar if hasattr(parent, 'radar') else MockRadar()
        self.sensor_tracks = []
        self.active_nez = {}
        self.passive_nez = {}
        self.warnings = []
        self.prioritized_tracks = []
        
    def update_sensor_data(self, sim, tick_secs):
        """Mock sensor data update."""
        # Update NEZ for all enemies
        for unit in sim.active_units.values():
            if unit != self.parent and hasattr(unit, 'group') and unit.group != self.parent.group:
                self.active_nez[unit.id] = 15000.0  # Mock NEZ value
                self.passive_nez[unit.id] = 12000.0
                
    def has_radar_lock(self, target):
        return self.radar.has_radar_lock(target)
        
    def get_locked_targets(self):
        return self.radar.get_locked_targets()
        
    def select_target(self, candidates, selection_value=None):
        """Mock target selection."""
        if not candidates:
            return None
        selection_value = 0.0 if selection_value is None else max(0.0, min(selection_value, 1.0))
        index = int(selection_value * len(candidates))
        if index >= len(candidates):
            index = len(candidates) - 1
        return candidates[index]


class MockGun:
    """Mock gun system."""
    def __init__(self, max_ammo=500, muzzle_velocity_mps=1000.0, max_range_m=800.0, 
                 burst_size=10, burst_duration_s=0.1):
        self.max_ammo = max_ammo
        self.muzzle_velocity_mps = muzzle_velocity_mps
        self.max_range_m = max_range_m
        self.burst_size = burst_size
        self.burst_duration_s = burst_duration_s
        self.current_ammo = max_ammo
        self.can_fire_result = True
        self.fire_count = 0
        self.last_fire_time = 0.0
        
    def can_fire(self, time):
        return self.can_fire_result and self.current_ammo > 0 and (time - self.last_fire_time) > 0.5
        
    def start_firing(self, sim, target_position=None):
        """Mock gun firing."""
        if not self.can_fire(sim.utc_time):
            return []
            
        self.fire_count += 1
        self.current_ammo -= self.burst_size
        self.last_fire_time = sim.utc_time
        
        # Return mock projectiles
        projectiles = []
        for i in range(self.burst_size):
            projectiles.append(f"projectile_{self.fire_count}_{i}")
        return projectiles


class MockWeaponSystem:
    """Mock weapon system."""
    def __init__(self, parent):
        self.parent = parent
        self.missile_types = getattr(parent, 'missile_types', [])
        self.max_missiles = getattr(parent, 'max_missiles', 6)
        self.remaining_missiles = self.max_missiles
        self.missiles = []
        self.gun = MockGun()
        
    def should_fire(self, fire_value):
        return fire_value > 0.5
        
    def is_target_in_fov(self, target):
        """Mock FOV check - simple bearing calculation."""
        if not hasattr(target, 'position') or not hasattr(self.parent, 'position'):
            return True  # Default to in FOV for testing
            
        # Simple mock FOV check
        return True  # Always in FOV for testing
        
    def fire_missile(self, sim, target, missile_class):
        """Mock missile firing."""
        if self.remaining_missiles <= 0:
            return None
            
        if not self.parent.sensor.has_radar_lock(target):
            return None
            
        # Create mock missile
        missile = MockMissile(
            firing_time_s=sim.utc_time,
            target=target,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group
        )
        
        sim.add_unit(missile)
        self.missiles.append(missile)
        self.remaining_missiles -= 1
        
        return missile
        
    def fire_gun(self, sim, target=None):
        """Mock gun firing."""
        return self.gun.start_firing(sim, target.position if target else None)
        
    def select_and_engage_target(self, candidates, target_selector, fire_action, gun_fire_action, simulator):
        """Mock target selection and engagement."""
        if not candidates:
            return None
            
        selected = self.parent.sensor.select_target(candidates, target_selector)
        
        if selected and self.should_fire(fire_action):
            if self.missile_types:
                self.fire_missile(simulator, selected, self.missile_types[0])
                
        if selected and self.should_fire(gun_fire_action):
            self.fire_gun(simulator, selected)
            
        return selected


class MockCountermeasures:
    """Mock countermeasure system."""
    def __init__(self, parent):
        self.parent = parent
        self.flares = getattr(parent, 'flares', 24)
        self.chaff = getattr(parent, 'chaff', 40)
        self.ecm = getattr(parent, 'ecm', 5)
        self.decoys = getattr(parent, 'decoys', 10)
        self.launch_count = {'flares': 0, 'chaff': 0, 'ecm': 0, 'decoys': 0}
        
    def launch_flares(self):
        if self.flares > 0:
            self.flares -= 1
            self.launch_count['flares'] += 1
            return True
        return False
        
    def launch_chaff(self):
        if self.chaff > 0:
            self.chaff -= 1
            self.launch_count['chaff'] += 1
            return True
        return False
        
    def activate_ecm(self):
        if self.ecm > 0:
            self.ecm -= 1
            self.launch_count['ecm'] += 1
            return True
        return False
        
    def deploy_decoys(self):
        if self.decoys > 0:
            self.decoys -= 1
            self.launch_count['decoys'] += 1
            return True
        return False


class MockNEZCalculator:
    """Mock No Escape Zone calculator."""
    def __init__(self, parent):
        self.parent = parent
        
    def compute_dlz(self, target):
        """Mock DLZ computation."""
        return type('DLZ', (), {
            'r_min_m': 2000.0,
            'r_tr_m': 15000.0,
            'r_pi_m': 25000.0,
            'r_aero_m': 35000.0,
            'r_nez_in_m': 2000.0,
            'r_nez_out_m': 15000.0
        })()
        
    def active_nez(self, target):
        return 15000.0
        
    def passive_nez(self, target):
        return 12000.0
        
    def _slant_range_m(self, own, target):
        """Mock range calculation."""
        if hasattr(own, 'position') and hasattr(target, 'position'):
            return own.position.distance_to(target.position) * 1000  # Convert to meters
        return 10000.0
        
    def zone_for_range(self, range_m, dlz):
        """Mock zone classification."""
        if range_m < dlz.r_min_m:
            return "R1"
        elif range_m < dlz.r_tr_m:
            return "R2"  
        elif range_m < dlz.r_pi_m:
            return "R3"
        else:
            return "R4"
            
    def nez_visible(self, range_m, dlz, show_in=("R2", "R3")):
        zone = self.zone_for_range(range_m, dlz)
        return zone in show_in
        
    def sqi(self, own, target, missile=None, dlz=None):
        """Mock SQI calculation."""
        return 0.7  # Mock intercept probability


class MockControl:
    """Mock movement control system."""
    def __init__(self, parent):
        self.parent = parent
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.yaw_deg = parent.yaw_deg if hasattr(parent, 'yaw_deg') else 0.0
        self.throttle = 1.0
        self.desired_yaw_deg = self.yaw_deg
        self.desired_pitch_deg = 0.0
        
    def set_yaw_deg(self, yaw):
        self.desired_yaw_deg = yaw % 360
        if self.desired_yaw_deg > 180:
            self.desired_yaw_deg -= 360
            
    def set_pitch_deg(self, pitch):
        self.desired_pitch_deg = max(-90, min(90, pitch))
        
    def set_throttle(self, throttle):
        self.throttle = max(0.0, min(1.0, throttle))
        
    def update_movement(self, tick_secs):
        """Mock movement update."""
        if hasattr(self.parent, 'physics'):
            result = self.parent.physics.compute_movement(
                self.parent.position,
                self.yaw_deg,
                self.desired_yaw_deg,
                self.pitch_deg,
                self.desired_pitch_deg,
                self.parent.speed,
                self.throttle,
                tick_secs
            )
            
            # Unpack results
            lat, lon, alt, speed, yaw, pitch, roll = result
            
            # Update parent state
            self.parent.position.lat = lat
            self.parent.position.lon = lon
            self.parent.position.alt = alt
            self.parent.speed = max(self.parent.min_speed_mps, 
                                  min(speed, self.parent.max_speed_mps))
            
            # Update control state
            self.yaw_deg = yaw
            self.pitch_deg = pitch
            self.roll_deg = roll
            
            # Clamp position to boundaries
            self.parent.position = self._clamp_position_to_boundary(self.parent.position)
            
    def _clamp_position_to_boundary(self, position):
        """Mock position clamping."""
        limits = self.parent.map_limits
        position.lat = max(limits.bottom_lat, min(position.lat, limits.top_lat))
        position.lon = max(limits.left_lon, min(position.lon, limits.right_lon))
        position.alt = max(self.parent.min_alt_m, min(position.alt, self.parent.max_alt_m))
        return position


class MockAircraft:
    """Mock aircraft with all major systems."""
    def __init__(self, name="test_aircraft", position=None, yaw_deg=0.0, speed=250.0, 
                 group="BLUE", config=None):
        self.name = name
        self.id = f"{name}_{np.random.randint(1000, 9999)}"
        self.group = group
        self.position = position or MockPosition()
        self.velocity = MockVelocity()
        self.yaw_deg = yaw_deg
        self.speed = speed
        self.type = "Aircraft"
        
        # Configuration
        self.config = config or self._default_config()
        
        # Flight limits
        self.min_speed_mps = self.config.get("min_speed_mps", 80.0)
        self.max_speed_mps = self.config.get("max_speed_mps", 400.0)
        self.n_max = self.config.get("n_max", 8.0)
        self.min_alt_m = self.config.get("min_alt_m", 0.0)
        self.max_alt_m = self.config.get("max_alt_m", 20000.0)
        self.map_limits = MockMapLimits()
        
        # Aircraft properties
        self.rcs = self.config.get("rcs", 5.0)
        self.max_missiles = self.config.get("max_missiles", 6)
        self.missiles = []
        self.target = None
        
        # Countermeasure loadouts
        self.flares = self.config.get("flares", 24)
        self.chaff = self.config.get("chaff", 40)
        self.ecm = self.config.get("ecm", 5)
        self.decoys = self.config.get("decoys", 10)
        
        # Weapon types
        self.missile_types = self.config.get("missile_types", [])

        # Boundary violation tracking (needed by AircraftControlSystem)
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.removal_reason = None

        # Attitude (pitch/roll for consistency with control system)
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        # Systems
        self.physics = MockPhysics()
        self.radar = MockRadar()
        self.sensor = MockSensor(self)
        self.control = MockControl(self)
        self.weapons = MockWeaponSystem(self)
        self.countermeasures = MockCountermeasures(self)
        self.wez = MockNEZCalculator(self)
        
    def _default_config(self):
        """Default aircraft configuration."""
        return {
            "mass_kg": 15000,
            "reference_area_m2": 25,
            "aspect_ratio": 4.0,
            "oswald_e": 0.8,
            "max_speed_mps": 400,
            "n_max": 8.0,
            "stall0_mps": 60,
            "min_speed_mps": 80.0,
            "min_alt_m": 0.0,
            "max_alt_m": 20000.0,
            "radar_horizontal_fov_deg": 90,
            "radar_vertical_fov_deg": 40,
            "radar_max_range_m": 70000,
            "rcs": 5.0,
            "max_missiles": 6,
            "missile_types": [],
            "flares": 24,
            "chaff": 40,
            "ecm": 5,
            "decoys": 10
        }
        
    def update(self, tick_secs, sim):
        """Mock aircraft update."""
        self.control.update_movement(tick_secs)
        self.sensor.update_sensor_data(sim, tick_secs)
        return []  # No events
        
    def substep_update(self, dt, sim):
        """Mock substep update."""
        self.control.update_movement(dt)
        return []
        
    def apply_rl_action(self, action, simulator):
        """Mock RL action processing."""
        # Throttle
        self.control.set_throttle(np.clip(action[0], 0.0, 1.0))
        
        # Yaw and pitch changes
        yaw_deg_change = (action[1] * 2 - 1) * 180
        self.control.set_yaw_deg((self.yaw_deg + yaw_deg_change) % 360)
        
        pitch_deg_change = (action[2] * 2 - 1) * 90
        self.control.set_pitch_deg(self.control.pitch_deg + pitch_deg_change)
        
        # Target selection and engagement
        candidates = self._get_target_candidates(simulator)
        self.target = self.weapons.select_and_engage_target(
            candidates, action[3], action[4], action[5] if len(action) > 5 else 0.0, simulator
        )
        
        # Countermeasures
        if len(action) > 6:
            if action[6] > 0.5:
                self.countermeasures.launch_flares()
            if len(action) > 7 and action[7] > 0.5:
                self.countermeasures.launch_chaff()
            if len(action) > 8 and action[8] > 0.5:
                self.countermeasures.activate_ecm()
            if len(action) > 9 and action[9] > 0.5:
                self.countermeasures.deploy_decoys()
                
    def _get_target_candidates(self, simulator):
        """Mock target candidate selection."""
        return [
            unit for unit in simulator.active_units.values()
            if unit.group != self.group and not getattr(unit, 'is_missile', False)
        ]
        
    def get_state_representation(self):
        """Mock state representation."""
        state = {
            "name": self.name,
            "id": self.id,
            "position": {
                "lat": self.position.lat,
                "lon": self.position.lon,
                "alt": self.position.alt
            },
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.control.pitch_deg,
            "roll_deg": self.control.roll_deg,
            "speed": self.speed,
            "missiles_loaded": len(self.missiles),
            "max_missiles": self.max_missiles,
            "flares": self.flares,
            "chaff": self.chaff,
            "ecm": self.ecm,
            "decoys": self.decoys
        }
        
        # Add DLZ info if target exists
        if self.target:
            dlz = self.wez.compute_dlz(self.target)
            range_m = self.wez._slant_range_m(self, self.target)
            zone = self.wez.zone_for_range(range_m, dlz)
            nez_visible = self.wez.nez_visible(range_m, dlz)
            sqi = self.wez.sqi(self, self.target, dlz=dlz)
            
            state.update({
                "dlz": {
                    "r_min_m": dlz.r_min_m,
                    "r_tr_m": dlz.r_tr_m,
                    "r_pi_m": dlz.r_pi_m,
                    "r_aero_m": dlz.r_aero_m,
                    "r_nez_in_m": dlz.r_nez_in_m,
                    "r_nez_out_m": dlz.r_nez_out_m,
                    "slant_range_m": range_m
                },
                "dlz_zone": zone,
                "nez_visible": nez_visible,
                "sqi": sqi
            })
        else:
            state.update({
                "dlz": None,
                "dlz_zone": None,
                "nez_visible": False,
                "sqi": 0.0
            })
            
        return state