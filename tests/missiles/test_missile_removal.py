#!/usr/bin/env python3

import math
from simulator.simulator import Simulator
from simulator.core.helpers import Position
from aircrafts.aircraft import Aircraft
from missiles.fox3.amraam import AIM120_AMRAAM

class MapLimitsConfig:
    def __init__(self):
        self.bottom_lat = -1.0
        self.top_lat = 1.0
        self.left_lon = -1.0
        self.right_lon = 1.0
        self.min_alt = 0
        self.max_alt = 20000

def calculate_distance(pos1, pos2):
    R_earth = 6_371_000.0
    deg2rad = math.pi / 180.0
    
    dlat = (pos2.lat - pos1.lat) * deg2rad
    dlon = (pos2.lon - pos1.lon) * deg2rad
    lat_avg = 0.5 * (pos1.lat + pos2.lat) * deg2rad
    
    dx = R_earth * dlon * math.cos(lat_avg)
    dy = R_earth * dlat
    dz = pos2.alt - pos1.alt
    
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def test_missile_removal():
    """Test why missiles are being removed early"""
    print("=== MISSILE REMOVAL TEST ===")
    
    sim = Simulator()
    map_limits = MapLimitsConfig()
    
    # Simple setup
    shooter_config = {
        "mass_kg": 18000, "reference_area_m2": 27.87, "max_speed_mps": 680, "n_max": 9.0,
        "stall0_mps": 60.0, "radar_max_range_m": 150000.0, "radar_tx_power_w": 50000.0,
        "radar_antenna_gain_db": 45.0, "radar_frequency_hz": 10e9,
        "radar_snr_threshold_db": 5.0, "rcs": 1.0
    }
    
    target_config = {
        "mass_kg": 18000, "reference_area_m2": 27.87, "max_speed_mps": 680, "n_max": 9.0,
        "stall0_mps": 60.0, "radar_max_range_m": 150000.0, "radar_tx_power_w": 50000.0,
        "radar_antenna_gain_db": 45.0, "radar_frequency_hz": 10e9,
        "radar_snr_threshold_db": 5.0, "rcs": 10000.0
    }
    
    shooter = Aircraft("Shooter", Position(0.0, 0.0, 10000), 0.0, 300.0, "BLUE", map_limits, shooter_config)
    target = Aircraft("Target", Position(0.05, 0.0, 10000), 180.0, 0.0, "RED", map_limits, target_config)
    
    sim.add_unit(shooter)
    sim.add_unit(target)
    
    # Wait for lock
    for i in range(50):
        sim.do_tick()
        if hasattr(shooter.sensor, 'has_radar_lock') and shooter.sensor.has_radar_lock(target):
            print(f"Lock at t={i*0.1:.1f}s")
            break
    
    # Fire missile
    missile = AIM120_AMRAAM(
        firing_time_s=sim.elapsed_time_s,
        target=target,
        source=shooter,
        map_limits=map_limits,
        group="BLUE"
    )
    sim.add_unit(missile)
    
    print(f"Missile fired at t={sim.elapsed_time_s:.1f}s")
    print(f"Launch distance: {calculate_distance(missile.position, target.position):.0f}m")
    
    # Check missile properties
    print(f"Missile life_time_s: {getattr(missile, 'life_time_s', 'N/A')}")
    print(f"Missile motor_burn_s: {getattr(missile, 'motor_burn_s', 'N/A')}")
    
    # Track missile with detailed removal analysis
    for t in range(200):  # 20 seconds max
        sim.do_tick()
        time_s = (t + 1) * 0.1
        
        if missile.id not in sim.active_units:
            print(f"MISSILE REMOVED at t={time_s:.1f}s")
            
            # Analyze why it was removed
            print("Checking removal conditions:")
            print(f"  Elapsed time: {getattr(missile, 'elapsed_time_s', 0):.1f}s")
            print(f"  Life time limit: {getattr(missile, 'life_time_s', 0):.1f}s")
            print(f"  Engine empty: {getattr(missile.engine, 'fuel_s', 0) == 0.0}")
            print(f"  Current speed: {getattr(missile, 'speed', 'Unknown'):.1f}m/s")
            print(f"  Altitude: {getattr(missile.position, 'alt', 'Unknown'):.0f}m")
            
            # Check radar lock
            locked_target = None
            if hasattr(missile, 'radar') and hasattr(missile.radar, 'get_locked_target'):
                try:
                    locked_target = missile.radar.get_locked_target()
                except:
                    pass
            print(f"  Has radar lock: {locked_target is not None}")
            
            # Check distance to target
            dist_km = calculate_distance(missile.position, target.position) / 1000.0
            print(f"  Distance to target: {dist_km:.1f}km")
            
            break
            
        if target.id not in sim.active_units:
            print(f"TARGET HIT at t={time_s:.1f}s!")
            break
        
        # Print status every 2 seconds
        if t % 20 == 0:
            distance = calculate_distance(missile.position, target.position)
            speed = getattr(missile, 'speed', 0)
            elapsed = getattr(missile, 'elapsed_time_s', 0)
            engine_empty = getattr(missile.engine, 'fuel_s', 1.0) == 0.0
            
            print(f"t={time_s:4.1f}s: {distance:6.0f}m, speed={speed:4.0f}m/s, elapsed={elapsed:.1f}s, engine_empty={engine_empty}")

if __name__ == "__main__":
    test_missile_removal()