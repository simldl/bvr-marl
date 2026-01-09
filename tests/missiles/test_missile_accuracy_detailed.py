#!/usr/bin/env python3

import math
import numpy as np
from typing import List, Dict, Tuple
from simulator.simulator import Simulator
from simulator.core.helpers import Position
from aircrafts.aircraft import Aircraft
from missiles.fox3.default_missile import LongRangeMissile
from missiles.fox1.sparrow import AIM7_Sparrow
from missiles.fox2.sidewinder import AIM9_Sidewinder
from missiles.fox3.amraam import AIM120_AMRAAM

class MapLimitsConfig:
    def __init__(self):
        self.left_lon = -1
        self.right_lon = 1
        self.bottom_lat = -1
        self.top_lat = 1
        self.min_alt = 0
        self.max_alt = 20000

def create_aircraft(name, lat, lon, alt, yaw, speed, group, map_limits):
    config = {
        "mass_kg": 18000, 
        "reference_area_m2": 50, 
        "max_speed_mps": 600, 
        "n_max": 9,
        "radar_horizontal_fov_deg": 60, 
        "radar_vertical_fov_deg": 30,
        "radar_max_range_m": 120000, 
        "radar_frequency_hz": 10e9, 
        "radar_tx_power_w": 15e3,
        "radar_antenna_gain_db": 40, 
        "radar_snr_threshold_db": 10, 
        "rcs": 10 if group == "blue" else 1000
    }
    return Aircraft(
        name=name,
        position=Position(lat=lat, lon=lon, alt=alt),
        yaw_deg=yaw,
        speed_mps=speed,
        group=group,
        map_limits=map_limits,
        config=config
    )

def calculate_distance_km(pos1, pos2):
    R_earth = 6_371_000.0
    deg2rad = math.pi / 180.0
    
    dlat = (pos2.lat - pos1.lat) * deg2rad
    dlon = (pos2.lon - pos1.lon) * deg2rad
    lat_avg = 0.5 * (pos1.lat + pos2.lat) * deg2rad
    
    dx = R_earth * dlon * math.cos(lat_avg)
    dy = R_earth * dlat
    dz = pos2.alt - pos1.alt
    
    return math.sqrt(dx*dx + dy*dy + dz*dz) / 1000.0

def calculate_relative_velocity(pos1, pos2, vel1, vel2):
    """Calculate relative velocity between two objects"""
    if vel1 is None or vel2 is None:
        return None
    
    # Simple relative velocity calculation
    rel_vel_x = vel2[0] - vel1[0] if len(vel1) >= 1 and len(vel2) >= 1 else 0
    rel_vel_y = vel2[1] - vel1[1] if len(vel1) >= 2 and len(vel2) >= 2 else 0
    rel_vel_z = vel2[2] - vel1[2] if len(vel1) >= 3 and len(vel2) >= 3 else 0
    
    return math.sqrt(rel_vel_x**2 + rel_vel_y**2 + rel_vel_z**2)

class DetailedMissileAnalysis:
    def __init__(self):
        self.position_history = []
        self.velocity_predictions = []
        self.actual_velocities = []
        self.target_positions = []
        self.guidance_phases = []
        self.distances = []
        
    def record_tick(self, missile, target, tick, time_s):
        """Record data for a single simulation tick"""
        missile_pos = getattr(missile, 'position', None)
        target_pos = getattr(target, 'position', None)
        
        if missile_pos and target_pos:
            distance = calculate_distance_km(missile_pos, target_pos)
            self.distances.append(distance)
            self.position_history.append({
                'tick': tick,
                'time': time_s,
                'missile_pos': (missile_pos.lat, missile_pos.lon, missile_pos.alt),
                'target_pos': (target_pos.lat, target_pos.lon, target_pos.alt),
                'distance_km': distance
            })
            
            # Try to get missile velocity
            missile_vel = [
                getattr(missile, 'velocity_x', getattr(missile, 'speed', 0) * math.cos(math.radians(getattr(missile, 'yaw_deg', 0)))),
                getattr(missile, 'velocity_y', getattr(missile, 'speed', 0) * math.sin(math.radians(getattr(missile, 'yaw_deg', 0)))),
                getattr(missile, 'velocity_z', 0)
            ]
            
            target_vel = [
                getattr(target, 'velocity_x', getattr(target, 'speed', 0) * math.cos(math.radians(getattr(target, 'yaw_deg', 0)))),
                getattr(target, 'velocity_y', getattr(target, 'speed', 0) * math.sin(math.radians(getattr(target, 'yaw_deg', 0)))),
                getattr(target, 'velocity_z', 0)
            ]
            
            missile_speed = getattr(missile, 'speed', math.sqrt(missile_vel[0]**2 + missile_vel[1]**2 + missile_vel[2]**2))
            target_speed = getattr(target, 'speed', math.sqrt(target_vel[0]**2 + target_vel[1]**2 + target_vel[2]**2))
            
            self.actual_velocities.append({
                'missile_speed': missile_speed,
                'target_speed': target_speed,
                'relative_speed': calculate_relative_velocity(missile_pos, target_pos, missile_vel, target_vel)
            })
            
            # Try to get guidance phase
            guidance_phase = "unknown"
            if hasattr(missile, 'guidance'):
                try:
                    guidance_phase = missile.guidance._select_guidance_phase(missile_pos, target_pos)
                except:
                    pass
            self.guidance_phases.append(guidance_phase)
            
            # Get predicted target velocity if available
            predicted_vel = None
            if hasattr(missile, 'target_provider'):
                predicted_vel = missile.target_provider.get_guidance_velocity()
            self.velocity_predictions.append(predicted_vel)
    
    def analyze_accuracy(self):
        """Analyze the accuracy of velocity predictions"""
        if not self.velocity_predictions or not self.actual_velocities:
            return {}
        
        valid_predictions = []
        prediction_errors = []
        
        for i, (pred, actual) in enumerate(zip(self.velocity_predictions, self.actual_velocities)):
            if pred is not None and len(pred) >= 3:
                pred_speed = math.sqrt(pred[0]**2 + pred[1]**2 + pred[2]**2)
                actual_speed = actual['target_speed']
                error = abs(pred_speed - actual_speed)
                prediction_errors.append(error)
                valid_predictions.append({
                    'tick': i,
                    'predicted_speed': pred_speed,
                    'actual_speed': actual_speed,
                    'error': error,
                    'error_percent': (error / actual_speed * 100) if actual_speed > 0 else 0
                })
        
        if not valid_predictions:
            return {"valid_predictions": 0}
        
        avg_error = np.mean(prediction_errors)
        max_error = np.max(prediction_errors)
        min_error = np.min(prediction_errors)
        
        avg_error_percent = np.mean([p['error_percent'] for p in valid_predictions])
        
        return {
            "valid_predictions": len(valid_predictions),
            "avg_error_mps": avg_error,
            "max_error_mps": max_error,
            "min_error_mps": min_error,
            "avg_error_percent": avg_error_percent,
            "first_10_predictions": valid_predictions[:10],
            "last_10_predictions": valid_predictions[-10:]
        }
    
    def analyze_guidance_phases(self):
        """Analyze which guidance phases were used"""
        if not self.guidance_phases:
            return {}
        
        phase_counts = {}
        for phase in self.guidance_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        total_ticks = len(self.guidance_phases)
        phase_percentages = {phase: count/total_ticks*100 for phase, count in phase_counts.items()}
        
        return {
            "phase_counts": phase_counts,
            "phase_percentages": phase_percentages,
            "total_ticks": total_ticks
        }

def run_detailed_missile_test(missile_class, scenario_name, shooter_pos, target_pos, target_heading, target_speed=300):
    """Run detailed missile test with analysis"""
    
    analysis = DetailedMissileAnalysis()
    
    try:
        sim = Simulator(tick_secs=0.1)
        map_limits = MapLimitsConfig()
        
        # Create aircraft
        shooter = create_aircraft("Shooter", shooter_pos[0], shooter_pos[1], shooter_pos[2], 0, 300, "blue", map_limits)
        target = create_aircraft("Target", target_pos[0], target_pos[1], target_pos[2], target_heading, target_speed, "red", map_limits)
        
        sim.add_unit(shooter)
        sim.add_unit(target)
        
        initial_distance = calculate_distance_km(shooter.position, target.position)
        
        # Point shooter toward target
        from simulator.utils.geodesics import geodetic_bearing_deg
        bearing = geodetic_bearing_deg(
            shooter.position.lat, shooter.position.lon,
            target.position.lat, target.position.lon
        )
        shooter.yaw_deg = bearing
        
        # Wait for radar lock
        lock_achieved = False
        for i in range(50):
            sim.do_tick()
            if shooter.sensor.has_radar_lock(target):
                lock_achieved = True
                break
        
        if not lock_achieved:
            return {
                "scenario": scenario_name,
                "missile": missile_class.__name__,
                "result": "NO_LOCK",
                "initial_distance_km": initial_distance,
                "analysis": analysis
            }
        
        # Fire missile
        missile = None
        if hasattr(shooter.weapons, 'fire_missile'):
            try:
                missile = shooter.weapons.fire_missile(sim, target, missile_class)
            except:
                pass
        
        if missile is None:
            try:
                missile = missile_class(
                    firing_time_s=sim.elapsed_time_s,
                    target=target,
                    source=shooter,
                    map_limits=map_limits,
                    group="blue"
                )
                sim.add_unit(missile)
            except Exception as e:
                return {
                    "scenario": scenario_name,
                    "missile": missile_class.__name__,
                    "result": f"CREATE_FAILED: {str(e)[:50]}",
                    "initial_distance_km": initial_distance,
                    "analysis": analysis
                }
        
        # Run detailed simulation
        max_ticks = 1200  # 2 minutes max
        hit = False
        final_distance = initial_distance
        
        for tick in range(max_ticks):
            sim.do_tick()
            time_s = (tick + 1) * 0.1
            
            # Record detailed analysis data
            if missile.id in sim.active_units and target.id in sim.active_units:
                analysis.record_tick(missile, target, tick + 1, time_s)
            
            # Check if target was hit
            if target.id not in sim.active_units:
                hit = True
                final_distance = 0.0
                break
            
            # Check if missile is still active
            if missile.id not in sim.active_units:
                if hasattr(missile, 'position'):
                    final_distance = calculate_distance_km(missile.position, target.position)
                break
            
            # Update final distance
            if hasattr(missile, 'position'):
                final_distance = calculate_distance_km(missile.position, target.position)
            
            # Safety: stop if missile is clearly diverging
            if tick > 100 and final_distance > initial_distance * 2:
                break
        
        return {
            "scenario": scenario_name,
            "missile": missile_class.__name__,
            "result": "HIT" if hit else "MISS",
            "hit": hit,
            "initial_distance_km": initial_distance,
            "final_distance_km": final_distance,
            "time_s": time_s,
            "ticks": tick + 1,
            "analysis": analysis
        }
        
    except Exception as e:
        return {
            "scenario": scenario_name,
            "missile": missile_class.__name__,
            "result": f"EXCEPTION: {str(e)[:50]}",
            "initial_distance_km": calculate_distance_km(Position(*shooter_pos), Position(*target_pos)),
            "analysis": analysis
        }

def test_detailed_missile_accuracy():
    """Test missile accuracy with detailed analysis"""
    
    print("=== DETAILED MISSILE ACCURACY TEST (NO MOVING AVERAGE FILTER) ===\n")
    
    # Test scenarios
    scenarios = [
        ("Short_Head_On", (0.0, 0.0, 10000), (0.0, 0.09, 10000), 180, 300, "10km head-on"),
        ("Medium_Head_On", (0.0, 0.0, 10000), (0.0, 0.18, 10000), 180, 300, "20km head-on"), 
        ("Medium_Beam", (0.0, 0.0, 10000), (0.0, 0.18, 10000), 90, 300, "20km beam"),
        ("Long_Head_On", (0.0, 0.0, 8000), (0.0, 0.45, 12000), 180, 350, "50km head-on"),
    ]
    
    # Missile types
    missile_types = [
        (AIM120_AMRAAM, "AIM-120 AMRAAM (Fox 3)"),
        (AIM7_Sparrow, "AIM-7 Sparrow (Fox 1)"),
        (AIM9_Sidewinder, "AIM-9 Sidewinder (Fox 2)"),
        (LongRangeMissile, "Default Long Range")
    ]
    
    all_results = []
    
    for missile_class, missile_name in missile_types:
        print(f"\n{'='*70}")
        print(f"TESTING: {missile_name}")
        print(f"{'='*70}")
        
        missile_results = []
        
        for scenario_name, shooter_pos, target_pos, target_heading, target_speed, description in scenarios:
            # Skip long range for Fox 2
            if "Fox 2" in missile_name and "Long_" in scenario_name:
                continue
            
            print(f"\nScenario: {description}")
            print("-" * 50)
            
            result = run_detailed_missile_test(
                missile_class, scenario_name, shooter_pos, target_pos, target_heading, target_speed
            )
            
            missile_results.append(result)
            all_results.append(result)
            
            # Print basic result
            if result.get("hit"):
                status = "[HIT]"
            elif "NO_LOCK" in result.get("result", ""):
                status = "[NO_LOCK]" 
            elif "FAILED" in result.get("result", ""):
                status = "[ERROR]"
            else:
                status = "[MISS]"
            
            print(f"Result: {status}")
            if result.get("initial_distance_km"):
                final_dist = result.get('final_distance_km', 0)
                final_dist_str = f"{final_dist:.1f}" if final_dist is not None else "N/A"
                print(f"Distance: {result['initial_distance_km']:.1f}km -> {final_dist_str}km")
            if result.get("time_s"):
                print(f"Time: {result['time_s']:.1f}s")
            
            # Analyze prediction accuracy
            analysis = result.get("analysis")
            if analysis:
                accuracy_stats = analysis.analyze_accuracy()
                guidance_stats = analysis.analyze_guidance_phases()
                
                if accuracy_stats.get("valid_predictions", 0) > 0:
                    print(f"Velocity Predictions: {accuracy_stats['valid_predictions']} valid")
                    print(f"  Avg Error: {accuracy_stats['avg_error_mps']:.1f} m/s ({accuracy_stats['avg_error_percent']:.1f}%)")
                    print(f"  Max Error: {accuracy_stats['max_error_mps']:.1f} m/s")
                
                if guidance_stats.get("phase_percentages"):
                    print(f"Guidance Phases:")
                    for phase, percentage in guidance_stats["phase_percentages"].items():
                        if percentage > 5:  # Only show phases used >5% of time
                            print(f"  {phase}: {percentage:.1f}%")
        
        # Calculate missile type summary
        successful_tests = [r for r in missile_results if "ERROR" not in r.get("result", "") and "NO_LOCK" not in r.get("result", "")]
        if successful_tests:
            hits = sum(1 for r in successful_tests if r.get("hit"))
            accuracy = (hits / len(successful_tests)) * 100
            print(f"\n{missile_name} Summary: {hits}/{len(successful_tests)} hits ({accuracy:.1f}% accuracy)")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (NO MOVING AVERAGE FILTER)")  
    print(f"{'='*70}")
    
    missile_summaries = {}
    for result in all_results:
        missile_name = result.get("missile", "Unknown")
        if missile_name not in missile_summaries:
            missile_summaries[missile_name] = {"total": 0, "hits": 0, "errors": 0}
        
        missile_summaries[missile_name]["total"] += 1
        if result.get("hit"):
            missile_summaries[missile_name]["hits"] += 1
        elif "ERROR" in result.get("result", "") or "NO_LOCK" in result.get("result", ""):
            missile_summaries[missile_name]["errors"] += 1
    
    for missile_name, stats in missile_summaries.items():
        successful = stats["total"] - stats["errors"]
        if successful > 0:
            accuracy = (stats["hits"] / successful) * 100
            print(f"{missile_name:25} {stats['hits']:2d}/{successful:2d} ({accuracy:5.1f}%) [{stats['errors']} errors]")

if __name__ == "__main__":
    test_detailed_missile_accuracy()