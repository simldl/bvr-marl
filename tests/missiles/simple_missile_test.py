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

def simple_test():
    print("=== SIMPLE MISSILE TEST ===")
    
    # Create the most basic scenario possible
    sim = Simulator()
    map_limits = MapLimitsConfig()
    
    # Create shooter at origin
    shooter_config = {
        "mass_kg": 18000,
        "reference_area_m2": 27.87,
        "max_speed_mps": 680,
        "n_max": 9.0,
        "stall0_mps": 60.0,
        "radar_max_range_m": 150000.0,
        "radar_tx_power_w": 50000.0,
        "radar_antenna_gain_db": 45.0,
        "radar_frequency_hz": 10e9,
        "radar_snr_threshold_db": 5.0,
        "rcs": 1.0
    }
    
    shooter = Aircraft(
        name="Shooter",
        position=Position(0.0, 0.0, 10000),
        yaw_deg=0.0,
        speed_mps=300.0,
        group="BLUE",
        map_limits=map_limits,
        config=shooter_config
    )
    
    # Create target close by with massive RCS
    target_config = {
        "mass_kg": 18000,
        "reference_area_m2": 27.87,
        "max_speed_mps": 680,
        "n_max": 9.0,
        "stall0_mps": 60.0,
        "radar_max_range_m": 150000.0,
        "radar_tx_power_w": 50000.0,
        "radar_antenna_gain_db": 45.0,
        "radar_frequency_hz": 10e9,
        "radar_snr_threshold_db": 5.0,
        "rcs": 10000.0  # Massive RCS
    }
    
    target = Aircraft(
        name="Target",
        position=Position(0.05, 0.0, 10000),  # ~5km north - closer target
        yaw_deg=180.0,  # Facing shooter
        speed_mps=0.0,  # Stationary
        group="RED",
        map_limits=map_limits,
        config=target_config
    )
    
    sim.add_unit(shooter)
    sim.add_unit(target)
    
    initial_distance = calculate_distance(shooter.position, target.position)
    print(f"Initial separation: {initial_distance:.0f}m")
    
    # Wait longer for radar lock
    print("Establishing radar lock...")
    for i in range(100):  # 10 seconds
        sim.do_tick()
        if hasattr(shooter.sensor, 'has_radar_lock') and shooter.sensor.has_radar_lock(target):
            print(f"OK Lock at t={i*0.1:.1f}s")
            break
        elif i % 20 == 0:
            print(f"t={i*0.1:.1f}s: Still seeking...")
    else:
        print("FAIL No lock - firing anyway")
    
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
    
    # Track for 30 seconds max
    distances = []
    for t in range(300):  # 30 seconds
        sim.do_tick()
        time_s = t * 0.1
        
        if missile.id not in sim.active_units:
            print(f"Missile removed at t={time_s:.1f}s")
            break
            
        if target.id not in sim.active_units:
            print(f"TARGET HIT at t={time_s:.1f}s!")
            break
        
        distance = calculate_distance(missile.position, target.position)
        distances.append(distance)
        
        # Print every 3 seconds
        if t % 30 == 0:
            print(f"t={time_s:4.1f}s: {distance:6.0f}m")
        
        # Check if missile is getting closer
        if len(distances) > 50:  # After 5 seconds
            recent_min = min(distances[-50:])
            if recent_min > initial_distance * 0.8:
                print(f"FAIL Missile not approaching effectively - stopping at t={time_s:.1f}s")
                break
    
    if distances:
        min_distance = min(distances)
        final_distance = distances[-1] if distances else initial_distance
        
        print(f"\nResults:")
        print(f"  Initial distance: {initial_distance:.0f}m")
        print(f"  Minimum distance: {min_distance:.0f}m")
        print(f"  Final distance: {final_distance:.0f}m")
        
        if min_distance < 1000:
            print("OK SUCCESS: Close approach achieved")
        elif min_distance < initial_distance * 0.5:
            print("WARN PARTIAL: Some approach but not close")
        else:
            print("FAIL FAILURE: Little to no approach")

if __name__ == "__main__":
    simple_test()