#!/usr/bin/env python3

import math
import numpy as np
from typing import List, Dict, Tuple
from simulator.simulator import Simulator
from simulator.core.helpers import Position
from aircrafts.aircraft import Aircraft
from missiles.fox1.sparrow import AIM7_Sparrow
from missiles.fox2.sidewinder import AIM9_Sidewinder  
from missiles.fox3.amraam import AIM120_AMRAAM

class MapLimitsConfig:
    def __init__(self):
        self.bottom_lat = -1.0
        self.top_lat = 1.0
        self.left_lon = -1.0
        self.right_lon = 1.0
        self.min_alt = 0
        self.max_alt = 20000

def create_test_aircraft(name: str, lat: float, lon: float, alt: float, yaw: float, speed: float = 300.0, group: str = "TEST") -> Aircraft:
    """Create standardized test aircraft"""
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
        "radar_snr_threshold_db": 10.0,
        "rcs": 10 if group == "BLUE" else 1000
    }
    aircraft = Aircraft(
        name=name,
        position=Position(lat, lon, alt),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=MapLimitsConfig(),
        config=config
    )
    return aircraft

def calculate_distance_km(pos1: Position, pos2: Position) -> float:
    """Calculate 3D distance between two positions in kilometers"""
    R_earth = 6_371_000.0
    deg2rad = math.pi / 180.0
    
    dlat = (pos2.lat - pos1.lat) * deg2rad
    dlon = (pos2.lon - pos1.lon) * deg2rad
    lat_avg = 0.5 * (pos1.lat + pos2.lat) * deg2rad
    
    dx = R_earth * dlon * math.cos(lat_avg)
    dy = R_earth * dlat
    dz = pos2.alt - pos1.alt
    
    return math.sqrt(dx*dx + dy*dy + dz*dz) / 1000.0

def run_missile_test(missile_class, scenario_name: str, shooter_pos: Tuple, target_pos: Tuple, 
                    shooter_speed: float = 300.0, target_speed: float = 300.0, 
                    target_heading: float = None) -> Dict:
    """Run a single missile engagement test"""
    
    sim = Simulator(tick_secs=0.1)
    map_limits = MapLimitsConfig()
    
    # Create aircraft
    shooter = create_test_aircraft(
        "Shooter", shooter_pos[0], shooter_pos[1], shooter_pos[2], 
        0, shooter_speed, "BLUE"
    )
    
    # Calculate target heading if not specified (face shooter)
    if target_heading is None:
        dlat = shooter_pos[0] - target_pos[0]
        dlon = shooter_pos[1] - target_pos[1]
        target_heading = math.degrees(math.atan2(dlon, dlat)) % 360
    
    target = create_test_aircraft(
        "Target", target_pos[0], target_pos[1], target_pos[2], 
        target_heading, target_speed, "RED"
    )
    
    # Point shooter at target
    dlat = target_pos[0] - shooter_pos[0]
    dlon = target_pos[1] - shooter_pos[1]
    shooter.yaw_deg = math.degrees(math.atan2(dlon, dlat)) % 360
    
    sim.add_unit(shooter)
    sim.add_unit(target)
    
    # Wait for radar lock (up to 5 seconds)
    lock_achieved = False
    for i in range(50):
        sim.do_tick()
        if hasattr(shooter.sensor, 'has_radar_lock') and shooter.sensor.has_radar_lock(target):
            lock_achieved = True
            break
    
    if not lock_achieved:
        return {
            "scenario": scenario_name,
            "missile": missile_class.__name__,
            "result": "NO_LOCK",
            "initial_distance_km": calculate_distance_km(Position(*shooter_pos), Position(*target_pos)),
            "ticks": 0,
            "final_distance_km": None,
            "hit": False
        }
    
    # Fire missile
    missile = missile_class(
        firing_time_s=sim.elapsed_time_s,
        target=target,
        source=shooter,
        map_limits=map_limits,
        group="BLUE"
    )
    sim.add_unit(missile)
    
    initial_distance = calculate_distance_km(missile.position, target.position)
    
    # Run simulation
    max_ticks = 3000  # 5 minutes max
    final_distance = None
    hit = False
    
    for tick in range(max_ticks):
        sim.do_tick()
        
        # Check if missile is still active
        if missile.id not in sim.active_units:
            final_distance = calculate_distance_km(missile.position, target.position) if hasattr(missile, 'position') else None
            break
            
        # Check if target is still active (hit)
        if target.id not in sim.active_units:
            hit = True
            final_distance = 0.0
            break
        
        # Update final distance
        final_distance = calculate_distance_km(missile.position, target.position)
        
        # Safety check - if missile is moving away and far
        if tick > 100 and final_distance > initial_distance * 1.5:
            break
    
    return {
        "scenario": scenario_name,
        "missile": missile_class.__name__,
        "result": "HIT" if hit else "MISS",
        "initial_distance_km": initial_distance,
        "ticks": tick + 1,
        "time_s": (tick + 1) * 0.1,
        "final_distance_km": final_distance,
        "hit": hit
    }

def run_accuracy_tests():
    """Run comprehensive missile accuracy tests"""
    print("=== MISSILE ACCURACY TEST SUITE ===\n")
    
    # Define test scenarios
    scenarios = [
        # Short range scenarios
        {
            "name": "Short_Range_Head_On",
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.05, 10000),  # ~5km north
            "target_heading": 180,  # Heading toward shooter
            "target_speed": 300
        },
        {
            "name": "Short_Range_Beam", 
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.05, 10000),  # ~5km north
            "target_heading": 90,   # Perpendicular
            "target_speed": 300
        },
        {
            "name": "Short_Range_Tail_Chase",
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.05, 10000),  # ~5km north  
            "target_heading": 0,    # Same direction
            "target_speed": 250
        },
        
        # Medium range scenarios  
        {
            "name": "Medium_Range_Head_On",
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.2, 10000),   # ~20km north
            "target_heading": 180,
            "target_speed": 300
        },
        {
            "name": "Medium_Range_Beam",
            "shooter_pos": (0.0, 0.0, 10000), 
            "target_pos": (0.0, 0.2, 10000),   # ~20km north
            "target_heading": 90,
            "target_speed": 300
        },
        
        # Long range scenarios (mainly for Fox 3)
        {
            "name": "Long_Range_Head_On",
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.5, 10000),   # ~50km north
            "target_heading": 180,
            "target_speed": 300
        },
        {
            "name": "Long_Range_Beam", 
            "shooter_pos": (0.0, 0.0, 10000),
            "target_pos": (0.0, 0.5, 10000),   # ~50km north
            "target_heading": 90,
            "target_speed": 300
        },
        
        # High altitude scenarios
        {
            "name": "High_Alt_Head_On",
            "shooter_pos": (0.0, 0.0, 15000),
            "target_pos": (0.0, 0.2, 15000),   # ~20km north, high altitude
            "target_heading": 180,
            "target_speed": 350
        },
        
        # Low altitude scenarios
        {
            "name": "Low_Alt_Head_On", 
            "shooter_pos": (0.0, 0.0, 3000),
            "target_pos": (0.0, 0.1, 3000),    # ~10km north, low altitude
            "target_heading": 180,
            "target_speed": 300
        }
    ]
    
    # Missile types to test
    missile_types = [
        (AIM7_Sparrow, "Fox 1"),
        (AIM9_Sidewinder, "Fox 2"), 
        (AIM120_AMRAAM, "Fox 3")
    ]
    
    results = []
    
    for missile_class, fox_type in missile_types:
        print(f"\n{'='*20} TESTING {fox_type} ({missile_class.__name__}) {'='*20}")
        
        fox_results = []
        
        for scenario in scenarios:
            # Skip long range tests for Fox 2 (IR missiles)
            if fox_type == "Fox 2" and "Long_Range" in scenario["name"]:
                continue
                
            print(f"Testing {scenario['name']}... ", end="", flush=True)
            
            result = run_missile_test(
                missile_class, 
                scenario["name"],
                scenario["shooter_pos"],
                scenario["target_pos"],
                target_speed=scenario["target_speed"],
                target_heading=scenario["target_heading"]
            )
            
            fox_results.append(result)
            results.append(result)
            
            # Print immediate result
            status = "✓ HIT" if result["hit"] else "✗ MISS"
            if result["result"] == "NO_LOCK":
                status = "⚠ NO_LOCK"
            
            print(f"{status} ({result['initial_distance_km']:.1f}km -> {result.get('final_distance_km', 'N/A'):.1f if result.get('final_distance_km') else 'N/A'}km in {result.get('time_s', 0):.1f}s)")
        
        # Calculate Fox type summary
        hits = sum(1 for r in fox_results if r["hit"])
        total = len(fox_results)
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        print(f"\n{fox_type} Summary: {hits}/{total} hits ({hit_rate:.1f}% accuracy)")
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MISSILE ACCURACY SUMMARY")
    print(f"{'='*60}")
    
    for missile_class, fox_type in missile_types:
        fox_results = [r for r in results if r["missile"] == missile_class.__name__]
        
        if not fox_results:
            continue
            
        hits = sum(1 for r in fox_results if r["hit"])
        misses = sum(1 for r in fox_results if not r["hit"] and r["result"] != "NO_LOCK")
        no_locks = sum(1 for r in fox_results if r["result"] == "NO_LOCK")
        total = len(fox_results)
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        print(f"\n{fox_type} ({missile_class.__name__}):")
        print(f"  Total Tests: {total}")
        print(f"  Hits: {hits} ({hit_rate:.1f}%)")
        print(f"  Misses: {misses}")
        print(f"  No Lock: {no_locks}")
        
        # Scenario breakdown
        scenarios_tested = {}
        for r in fox_results:
            scenario_type = r["scenario"].split("_")[0] + "_" + r["scenario"].split("_")[1]  # e.g., "Short_Range"
            if scenario_type not in scenarios_tested:
                scenarios_tested[scenario_type] = {"hits": 0, "total": 0}
            scenarios_tested[scenario_type]["total"] += 1
            if r["hit"]:
                scenarios_tested[scenario_type]["hits"] += 1
        
        print("  By Range:")
        for scenario_type, stats in scenarios_tested.items():
            rate = (stats["hits"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"    {scenario_type.replace('_', ' ')}: {stats['hits']}/{stats['total']} ({rate:.1f}%)")
    
    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print(f"{'='*60}")
    
    # Group by scenario for comparison
    scenario_names = list(set(r["scenario"] for r in results))
    scenario_names.sort()
    
    for scenario in scenario_names:
        scenario_results = [r for r in results if r["scenario"] == scenario]
        print(f"\n{scenario}:")
        for r in scenario_results:
            status = "HIT" if r["hit"] else "MISS" if r["result"] != "NO_LOCK" else "NO_LOCK"
            missile_name = r["missile"].replace("AIM", "").replace("_", " ")
            print(f"  {missile_name:15} {status:8} {r['initial_distance_km']:5.1f}km -> {r.get('final_distance_km', 0):5.1f}km ({r.get('time_s', 0):5.1f}s)")

if __name__ == "__main__":
    run_accuracy_tests()