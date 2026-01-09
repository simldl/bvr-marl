#!/usr/bin/env python3

import math
import numpy as np
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

def create_test_aircraft(name, lat, lon, alt, yaw, group="TEST"):
    config = {
        "mass_kg": 18000,
        "reference_area_m2": 27.87,
        "max_speed_mps": 680,
        "n_max": 9.0,
        "stall0_mps": 60.0,
        "radar_max_range_m": 120000.0,
        "radar_tx_power_w": 15000.0,
        "radar_antenna_gain_db": 40.0,
        "radar_frequency_hz": 10e9,
        "radar_snr_threshold_db": 8.0,
        "rcs": 1000.0 if group == "RED" else 10.0
    }
    aircraft = Aircraft(
        name=name,
        position=Position(lat, lon, alt),
        yaw_deg=yaw,
        speed_mps=300.0,
        group=group,
        map_limits=MapLimitsConfig(),
        config=config
    )
    return aircraft

def calculate_distance(pos1, pos2):
    """Calculate distance between two positions"""
    R_earth = 6_371_000.0
    deg2rad = math.pi / 180.0
    
    dlat = (pos2.lat - pos1.lat) * deg2rad
    dlon = (pos2.lon - pos1.lon) * deg2rad
    lat_avg = 0.5 * (pos1.lat + pos2.lat) * deg2rad
    
    dx = R_earth * dlon * math.cos(lat_avg)
    dy = R_earth * dlat
    dz = pos2.alt - pos1.alt
    
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def calculate_bearing(pos1, pos2):
    """Calculate bearing from pos1 to pos2"""
    dlat = pos2.lat - pos1.lat
    dlon = pos2.lon - pos1.lon
    
    bearing_rad = math.atan2(dlon, dlat)
    bearing_deg = math.degrees(bearing_rad)
    
    if bearing_deg < 0:
        bearing_deg += 360
    
    return bearing_deg

def debug_missile_guidance():
    """Debug why missiles aren't approaching targets"""
    print("=== DEBUGGING MISSILE GUIDANCE ===")
    
    sim = Simulator()
    map_limits = MapLimitsConfig()
    
    # Create aircraft 10km apart
    shooter = create_test_aircraft("Shooter", 0.0, 0.0, 10000, 0, "BLUE")
    target = create_test_aircraft("Target", 0.0, 0.09, 10000, 180, "RED")  # ~10km north
    
    sim.add_unit(shooter)
    sim.add_unit(target)
    
    initial_distance = calculate_distance(shooter.position, target.position)
    print(f"Initial separation: {initial_distance:.0f}m")
    print(f"Shooter position: {shooter.position.lat:.6f}, {shooter.position.lon:.6f}, {shooter.position.alt:.0f}")
    print(f"Target position: {target.position.lat:.6f}, {target.position.lon:.6f}, {target.position.alt:.0f}")
    
    # Wait for radar lock
    print("\n--- RADAR LOCK PHASE ---")
    for i in range(50):  # 5 seconds
        sim.do_tick()
        if hasattr(shooter.sensor, 'has_radar_lock') and shooter.sensor.has_radar_lock(target):
            print(f"✓ Radar lock established at t={i*0.1:.1f}s")
            break
        elif i % 10 == 0:
            print(f"t={i*0.1:.1f}s: Seeking lock...")
    else:
        print("❌ No radar lock established")
    
    # Check radar status
    print("\n--- RADAR STATUS ---")
    if hasattr(shooter, 'radar'):
        locked_target = shooter.radar.get_locked_target() if hasattr(shooter.radar, 'get_locked_target') else None
        tracks = shooter.radar.get_tracks() if hasattr(shooter.radar, 'get_tracks') else []
        print(f"Locked target: {locked_target}")
        print(f"Number of tracks: {len(tracks) if tracks else 0}")
    
    # Fire missile
    print("\n--- MISSILE LAUNCH ---")
    missile = AIM120_AMRAAM(
        firing_time_s=sim.elapsed_time_s,
        target=target,
        source=shooter,
        map_limits=map_limits,
        group="BLUE"
    )
    sim.add_unit(missile)
    
    launch_distance = calculate_distance(missile.position, target.position)
    print(f"Launch distance: {launch_distance:.0f}m")
    print(f"Missile position: {missile.position.lat:.6f}, {missile.position.lon:.6f}, {missile.position.alt:.0f}")
    print(f"Missile initial yaw: {missile.yaw_deg:.1f}°")
    print(f"Missile target ID: {getattr(missile, 'target', 'None')}")
    
    # Track missile for first 20 seconds with detailed info
    print("\n--- MISSILE TRACKING ---")
    max_time_debug = 20  # 20 seconds of detailed tracking
    dt = 0.1
    
    for t in range(int(max_time_debug / dt)):
        sim.do_tick()
        time_s = t * dt
        
        if missile.id not in sim.active_units:
            print(f"❌ Missile removed at t={time_s:.1f}s")
            break
        
        if target.id not in sim.active_units:
            print(f"✓ Target hit at t={time_s:.1f}s")
            break
        
        # Detailed tracking every second
        if t % 10 == 0:
            distance = calculate_distance(missile.position, target.position)
            missile_speed = getattr(missile, 'speed', 0)
            missile_yaw = getattr(missile, 'yaw_deg', 0)
            missile_pitch = getattr(missile, 'pitch_deg', 0)
            
            # Get guidance target info
            target_pos = None
            target_vel = None
            if hasattr(missile, 'target_provider'):
                try:
                    target_pos = missile.target_provider.get_guidance_target()
                    target_vel = missile.target_provider.get_guidance_velocity()
                except:
                    pass
            
            # Get missile radar status
            locked_target = None
            if hasattr(missile, 'radar') and hasattr(missile.radar, 'get_locked_target'):
                try:
                    locked_target = missile.radar.get_locked_target()
                except:
                    pass
            
            # Get guidance phase
            guidance_phase = "unknown"
            if hasattr(missile, 'guidance'):
                try:
                    guidance_phase = missile.guidance._select_guidance_phase(missile.position, target.position)
                except:
                    pass
            
            # Calculate bearing to target
            bearing_to_target = calculate_bearing(missile.position, target.position)
            bearing_error = abs(missile_yaw - bearing_to_target)
            if bearing_error > 180:
                bearing_error = 360 - bearing_error
            
            print(f"t={time_s:4.1f}s: dist={distance:6.0f}m, speed={missile_speed:4.0f}m/s, yaw={missile_yaw:5.1f}°, pitch={missile_pitch:4.1f}°")
            print(f"         bearing_to_tgt={bearing_to_target:5.1f}°, error={bearing_error:4.1f}°, phase={guidance_phase}")
            print(f"         locked={locked_target is not None}, has_target_pos={target_pos is not None}, has_target_vel={target_vel is not None}")
            
            # Check if missile is moving in right direction
            if t > 0:  # Skip first iteration
                if distance > launch_distance * 0.9:  # If not getting much closer
                    print(f"         ⚠ WARNING: Missile not approaching target effectively!")
        
        # Early exit if missile clearly not working
        if t > 50 and calculate_distance(missile.position, target.position) > launch_distance * 0.8:
            print(f"❌ Missile guidance appears ineffective, stopping debug at t={time_s:.1f}s")
            break
    
    final_distance = calculate_distance(missile.position, target.position) if missile.id in sim.active_units else 0
    print(f"\nFinal distance: {final_distance:.0f}m")
    
    # Analyze the results
    print("\n--- ANALYSIS ---")
    if final_distance > launch_distance * 0.5:
        print("❌ PROBLEM: Missile did not effectively approach target")
        print("Possible issues:")
        print("  1. Guidance system not receiving target position")
        print("  2. Guidance algorithm not working correctly") 
        print("  3. Missile not responding to guidance commands")
        print("  4. Target provider not functioning")
    elif final_distance < 1000:
        print("✓ SUCCESS: Missile guidance working effectively")
    else:
        print("⚠ PARTIAL: Missile approached but may have missed")

if __name__ == "__main__":
    debug_missile_guidance()