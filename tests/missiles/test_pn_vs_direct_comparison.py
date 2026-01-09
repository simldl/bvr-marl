#!/usr/bin/env python3
"""
Comprehensive test comparing PN vs Direct navigation performance.
Tests different scenarios: head-on, beam, maneuvering targets, etc.
"""

import sys
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from simulator.simulator import Simulator
from simulator.core.helpers import Position
from aircrafts.aircraft import Aircraft
from missiles.fox3.amraam import AIM120_AMRAAM

@dataclass
class ComparisonResult:
    scenario: str
    guidance_type: str
    hit: bool
    miss_distance: float
    time_to_intercept: float
    final_distance: float
    trajectory_efficiency: float  # ratio of direct path to actual path
    velocity_errors: List[float]  # velocity estimation errors over time

class NavigationTester:
    def __init__(self):
        self.results = []
        
    def run_scenario(self, scenario_name: str, shooter_pos: Tuple, target_pos: Tuple, 
                    target_heading: float, target_speed: float, target_maneuvers: bool = False,
                    guidance_override: str = None) -> ComparisonResult:
        """Run a single test scenario."""
        
        # Create simulation
        map_limits = {"min_lat": -1, "max_lat": 1, "min_lon": -1, "max_lon": 1}
        sim = Simulator(map_limits=map_limits, tick_rate_hz=10.0)
        
        # Create shooter aircraft
        shooter_position = Position(lat=shooter_pos[0], lon=shooter_pos[1], alt=shooter_pos[2])
        shooter = Aircraft('Shooter', shooter_position, yaw_deg=0.0, speed_mps=300.0, 
                          group='blue', map_limits=map_limits,
                          config={'mass_kg': 18000, 'reference_area_m2': 50, 'max_speed_mps': 600, 
                                 'n_max': 9, 'radar_max_range_m': 150000, 'radar_horizontal_fov_deg': 120,
                                 'radar_vertical_fov_deg': 60, 'radar_tx_power_w': 25000, 
                                 'radar_antenna_gain_db': 40, 'rcs': 0.1})
        sim.add_unit(shooter)
        
        # Create target aircraft
        target_position = Position(lat=target_pos[0], lon=target_pos[1], alt=target_pos[2])
        target = Aircraft('Target', target_position, yaw_deg=target_heading, speed_mps=target_speed,
                         group='red', map_limits=map_limits,
                         config={'mass_kg': 12000, 'reference_area_m2': 40, 'max_speed_mps': 500,
                                'n_max': 9, 'rcs': 5.0})
        sim.add_unit(target)
        
        # Wait for radar lock
        for _ in range(20):  # 2 seconds
            sim.tick()
            
        # Create and fire missile with guidance override  
        missile = AIM120_AMRAAM(
            source=shooter,
            target_id=target.id,
            map_limits=map_limits
        )
        # Set guidance override if specified
        if guidance_override:
            missile.guidance_mode_override = guidance_override
        sim.add_unit(missile)
        
        # Track simulation data
        start_time = sim.current_time
        trajectory_length = 0.0
        prev_missile_pos = missile.position
        velocity_errors = []
        min_distance = float('inf')
        
        # Run simulation
        for tick in range(2000):  # Max 200 seconds
            sim.tick()
            
            # Check if missile is destroyed or target hit
            if missile.is_destroyed:
                break
                
            if target.is_destroyed:
                break
                
            # Calculate trajectory metrics
            curr_pos = missile.position
            if prev_missile_pos:
                step_distance = self._calculate_distance(prev_missile_pos, curr_pos)
                trajectory_length += step_distance
            prev_missile_pos = curr_pos
            
            # Track minimum distance to target
            distance_to_target = self._calculate_distance(missile.position, target.position)
            min_distance = min(min_distance, distance_to_target)
            
            # Track velocity estimation accuracy if available
            if hasattr(missile, 'target_provider') and missile.target_provider:
                estimated_vel = missile.target_provider.get_guidance_velocity()
                if estimated_vel and hasattr(target, 'velocity'):
                    actual_vel = self._get_target_velocity_enu(target)
                    if actual_vel is not None:
                        error = np.linalg.norm(np.array(estimated_vel) - actual_vel)
                        velocity_errors.append(error)
            
            # Add target maneuvering if requested
            if target_maneuvers and tick % 100 == 0:  # Every 10 seconds
                target.yaw_deg += np.random.uniform(-45, 45)
                target.yaw_deg = target.yaw_deg % 360
        
        # Calculate results
        final_distance = self._calculate_distance(missile.position, target.position)
        hit = target.is_destroyed or final_distance < 50.0  # 50m proximity
        time_to_intercept = sim.current_time - start_time
        
        # Calculate direct path for efficiency
        direct_distance = self._calculate_distance(shooter_position, target_position)
        trajectory_efficiency = direct_distance / max(trajectory_length, 1.0)
        
        return ComparisonResult(
            scenario=scenario_name,
            guidance_type=guidance_override or "default",
            hit=hit,
            miss_distance=min_distance,
            time_to_intercept=time_to_intercept,
            final_distance=final_distance,
            trajectory_efficiency=trajectory_efficiency,
            velocity_errors=velocity_errors
        )
    
    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between two positions in meters."""
        R_earth = 6_371_000.0
        lat1, lon1, alt1 = math.radians(pos1.lat), math.radians(pos1.lon), pos1.alt
        lat2, lon2, alt2 = math.radians(pos2.lat), math.radians(pos2.lon), pos2.alt
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        horizontal_dist = R_earth * c
        vertical_dist = alt2 - alt1
        
        return math.sqrt(horizontal_dist**2 + vertical_dist**2)
    
    def _get_target_velocity_enu(self, target) -> np.ndarray:
        """Get target velocity in ENU coordinates."""
        if hasattr(target, 'velocity') and target.velocity:
            return np.array(target.velocity[:3])
        return None
    
    def run_comprehensive_test(self):
        """Run all test scenarios comparing PN vs Direct navigation."""
        
        scenarios = [
            # (name, shooter_pos, target_pos, heading, speed, maneuvers)
            ("Short_Head_On", (0.0, 0.0, 10000), (0.0, 0.09, 10000), 180, 300, False),
            ("Medium_Head_On", (0.0, 0.0, 10000), (0.0, 0.18, 10000), 180, 300, False),
            ("Long_Head_On", (0.0, 0.0, 10000), (0.0, 0.36, 10000), 180, 350, False),
            ("Short_Beam", (0.0, 0.0, 10000), (0.09, 0.0, 10000), 90, 300, False),
            ("Medium_Beam", (0.0, 0.0, 10000), (0.18, 0.0, 10000), 90, 300, False),
            ("Long_Beam", (0.0, 0.0, 10000), (0.36, 0.0, 10000), 90, 350, False),
            ("Maneuvering_Close", (0.0, 0.0, 10000), (0.18, 0.0, 10000), 45, 300, True),
            ("Maneuvering_Far", (0.0, 0.0, 10000), (0.36, 0.0, 10000), 45, 350, True),
        ]
        
        guidance_modes = ["pn", "direct"]
        
        print("=== Navigation Comparison Test ===")
        print(f"{'Scenario':<20} {'Guidance':<8} {'Hit':<5} {'Miss Dist':<10} {'Time':<8} {'Efficiency':<10} {'Avg Vel Err':<12}")
        print("-" * 85)
        
        for scenario_name, shooter_pos, target_pos, heading, speed, maneuvers in scenarios:
            for guidance_mode in guidance_modes:
                try:
                    result = self.run_scenario(
                        scenario_name, shooter_pos, target_pos, heading, speed, 
                        maneuvers, guidance_mode
                    )
                    self.results.append(result)
                    
                    avg_vel_error = np.mean(result.velocity_errors) if result.velocity_errors else 0.0
                    
                    print(f"{scenario_name:<20} {guidance_mode:<8} {str(result.hit):<5} "
                          f"{result.miss_distance:<10.1f} {result.time_to_intercept:<8.1f} "
                          f"{result.trajectory_efficiency:<10.3f} {avg_vel_error:<12.1f}")
                          
                except Exception as e:
                    print(f"{scenario_name:<20} {guidance_mode:<8} ERROR: {str(e)}")
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and compare results between PN and Direct navigation."""
        
        print("\n=== ANALYSIS ===")
        
        pn_results = [r for r in self.results if r.guidance_type == "pn"]
        direct_results = [r for r in self.results if r.guidance_type == "direct"]
        
        if not pn_results or not direct_results:
            print("Insufficient data for comparison")
            return
        
        # Hit rate comparison
        pn_hits = sum(1 for r in pn_results if r.hit)
        direct_hits = sum(1 for r in direct_results if r.hit)
        
        print(f"Hit Rate:")
        print(f"  PN: {pn_hits}/{len(pn_results)} ({100*pn_hits/len(pn_results):.1f}%)")
        print(f"  Direct: {direct_hits}/{len(direct_results)} ({100*direct_hits/len(direct_results):.1f}%)")
        
        # Average miss distance
        pn_avg_miss = np.mean([r.miss_distance for r in pn_results])
        direct_avg_miss = np.mean([r.miss_distance for r in direct_results])
        
        print(f"\nAverage Miss Distance:")
        print(f"  PN: {pn_avg_miss:.1f}m")
        print(f"  Direct: {direct_avg_miss:.1f}m")
        
        # Trajectory efficiency
        pn_avg_efficiency = np.mean([r.trajectory_efficiency for r in pn_results])
        direct_avg_efficiency = np.mean([r.trajectory_efficiency for r in direct_results])
        
        print(f"\nTrajectory Efficiency:")
        print(f"  PN: {pn_avg_efficiency:.3f}")
        print(f"  Direct: {direct_avg_efficiency:.3f}")
        
        # Velocity estimation accuracy
        pn_vel_errors = [e for r in pn_results for e in r.velocity_errors]
        direct_vel_errors = [e for r in direct_results for e in r.velocity_errors]
        
        if pn_vel_errors and direct_vel_errors:
            print(f"\nVelocity Estimation Error:")
            print(f"  PN: {np.mean(pn_vel_errors):.1f} ± {np.std(pn_vel_errors):.1f} m/s")
            print(f"  Direct: {np.mean(direct_vel_errors):.1f} ± {np.std(direct_vel_errors):.1f} m/s")
        
        # Recommendation
        print(f"\n=== RECOMMENDATION ===")
        better_guidance = "PN" if pn_hits > direct_hits else "Direct"
        if pn_hits == direct_hits:
            better_guidance = "PN" if pn_avg_miss < direct_avg_miss else "Direct"
        
        print(f"Better overall performance: {better_guidance}")
        
        if pn_hits > direct_hits:
            print("Reason: Higher hit rate")
        elif direct_hits > pn_hits:
            print("Reason: Higher hit rate")
        elif pn_avg_miss < direct_avg_miss:
            print("Reason: Lower average miss distance")
        else:
            print("Reason: Lower average miss distance")

def main():
    tester = NavigationTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()