"""
Beyond Visual Range (BVR) tactical maneuvers implementation.
Includes advanced BVR tactics like crank, skate, banzai, f-pole management, etc.
Uses ObservationHelper for all geometry calculations (consistent with BT and RL observations).
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.aircrafts.systems.observation_helper import ObservationHelper
from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus
from air_to_air_rl.simulator.utils.angles import normalize_angle, signed_yaw_deg_diff
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class BVRTactics:
    """Implementation of Beyond Visual Range combat tactics."""

    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.current_maneuver = None
        self.maneuver_start_time = 0.0
        self.target_bearing_history = []

        # ObservationHelper for consistent geometry calculations
        self.obs_helper = ObservationHelper(aircraft)

        # Get actual missile parameters from aircraft
        self.missile_params = self._get_missile_params()

    def _get_missile_params(self) -> dict[str, Any]:
        """Get actual missile parameters from aircraft configuration."""
        default_params = {
            "max_range_m": 100000,  # 100km BVR range
            "min_range_m": 1000,  # Default minimum range
            "nez_range_m": 60000,  # Default NEZ range (~60% of max)
            "active_range_m": 100000,  # 100km active seeker range
            "fox_type": 3,
        }

        # Try to get actual missile parameters
        if hasattr(self.aircraft, "missile_types") and self.aircraft.missile_types:
            try:
                missile_type = self.aircraft.missile_types[0]
                temp_missile = missile_type(
                    None, None, self.aircraft, self.aircraft.map_limits, self.aircraft.group
                )

                missile_range = getattr(temp_missile.radar, "max_range_m", 100000)
                default_params.update(
                    {
                        "max_range_m": missile_range,
                        "min_range_m": getattr(temp_missile, "min_range_m", 1000),
                        "nez_range_m": missile_range * 0.6,
                        "active_range_m": missile_range,
                        "fox_type": getattr(temp_missile, "fox_type", 3),
                    }
                )
                # Debug: Verify missile parameters for BVR tactics
                print(
                    f"BVRTactics: Missile params loaded - Max Range: {missile_range / 1000:.1f}km, NEZ: {missile_range * 0.6 / 1000:.1f}km"
                )
            except Exception:
                pass

        return default_params

    def execute_crank_maneuver(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute crank maneuver - maintain radar lock while reducing closure rate.

        Crank involves turning to place the target at the edge of radar FOV (60-70 degrees)
        to reduce closure rate while maintaining radar lock.
        """
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Get geometry from ObservationHelper (consistent with observations)
        geom = self.obs_helper.get_geometry_kinematics(target)
        if not geom["valid"]:
            return NodeStatus.FAILURE

        target_bearing = geom["bearing_deg"]
        target_range = geom["slant_range_m"] / 1000.0  # Convert to km

        # Determine crank direction (left or right)
        current_yaw = self.aircraft.yaw_deg
        bearing_diff = signed_yaw_deg_diff(current_yaw, target_bearing)

        # Choose crank direction to maintain radar lock
        radar_fov = getattr(self.aircraft.radar, "h_fov_deg", 60) / 2
        optimal_crank_angle = min(radar_fov - 10, 50)  # Stay within radar limits

        if abs(bearing_diff) > optimal_crank_angle:
            # Already at good crank angle, maintain
            desired_yaw = current_yaw
        else:
            # Turn to crank position
            if bearing_diff > 0:
                # Target is to the right, crank right
                desired_yaw = target_bearing - optimal_crank_angle
            else:
                # Target is to the left, crank left
                desired_yaw = target_bearing + optimal_crank_angle

        desired_yaw = normalize_angle(desired_yaw)

        # Set flight controls for crank
        context["desired_yaw"] = desired_yaw
        context["desired_pitch"] = 0.0  # Level flight during crank
        context["desired_throttle"] = 0.8  # Moderate throttle

        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(
            f"Aircraft {aircraft_id}: CRANKING - Range: {target_range:.1f}km, Desired Yaw: {desired_yaw:.1f}deg"
        )

        self.current_maneuver = "crank"
        return NodeStatus.SUCCESS

    def execute_launch_and_leave(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute launch-and-leave (skate) tactics.

        Fire missile then immediately turn away to exit before enemy launch range.
        """
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Check if we should fire using WeaponSystem + tactical criteria
        if not hasattr(self.aircraft, "weapons"):
            return NodeStatus.FAILURE

        # Get fire feasibility from WeaponSystem (lock, FOV, inventory checks)
        feasibility = self.aircraft.weapons.check_fire_feasibility(target)
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        if not feasibility["can_fire"]:
            print(f"Aircraft {aircraft_id}: L&L_VETO - {feasibility.get('veto_reason', 'unknown')}")
            return NodeStatus.FAILURE

        # WeaponSystem says OK - now check tactical criteria
        target_range = context.get("target_range_m", float("inf"))
        nez_out_range = context.get("nez_out_range_m", 0)
        closure_rate = context.get("closure_rate_mps", 0)
        r_aero = context.get("r_aero_m", 100000)

        # Tactical range envelope
        radar_max_range = (
            getattr(self.aircraft.radar, "max_range_m", 100000)
            if hasattr(self.aircraft, "radar")
            else 100000
        )
        max_fire_range = min(max(nez_out_range, r_aero * 0.9), radar_max_range * 0.95)

        # Tactical criteria
        range_ok = target_range <= max_fire_range
        closure_ok = closure_rate > 10

        in_launch_range = range_ok and closure_ok

        target_range_km = target_range / 1000
        print(
            f"Aircraft {aircraft_id}: LAUNCH_AND_LEAVE - Range: {target_range_km:.1f}km, MaxFire: {max_fire_range / 1000:.1f}km, "
            f"WpnSys: OK, Range: {'OK' if range_ok else 'FAIL'}, Closure: {'OK' if closure_ok else 'FAIL'}, "
            f"Tactical: {'OK' if in_launch_range else 'FAIL'}"
        )

        if in_launch_range:
            # Fire missile
            context["fire_missile"] = True
            print(f"Aircraft {aircraft_id}: FIRING MISSILE - Launch and Leave!")

            # Get target bearing from ObservationHelper
            geom = self.obs_helper.get_geometry_kinematics(target)
            if not geom["valid"]:
                return NodeStatus.FAILURE
            target_bearing = geom["bearing_deg"]

            # Check if we need to point at target first (from behavior tree)
            if context.get("pre_launch_pointing", False):
                # Still pointing at target - maintain target bearing for accurate launch
                context["desired_yaw"] = target_bearing
                context["desired_pitch"] = 0.0  # Level flight for stable launch
                context["desired_throttle"] = 0.75  # Maintain speed
                print(f"Aircraft {aircraft_id}: Pointing at target for accurate missile launch")
            else:
                # Already pointed at target - now skate away
                escape_bearing = normalize_angle(target_bearing + 180)
                context["desired_yaw"] = escape_bearing
                context["desired_pitch"] = -5.0  # Slight dive for speed
                context["desired_throttle"] = 1.0  # Full throttle
                print(f"Aircraft {aircraft_id}: Skating away after launch")

            self.current_maneuver = "skate"
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.FAILURE

    def execute_launch_and_decide(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute launch-and-decide (banzai) tactics.

        Fire missile and continue approach to assess results and potentially engage again.
        """
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Check if we should fire using the same logic as behavior tree
        target_range = context.get("target_range_m", float("inf"))
        nez_out_range = context.get("nez_out_range_m", 0)
        closure_rate = context.get("closure_rate_mps", 0)
        context.get("gimbal_ok", True)
        r_aero = context.get("r_aero_m", 100000)

        # Check if we should fire using WeaponSystem + tactical criteria
        if not hasattr(self.aircraft, "weapons"):
            return NodeStatus.FAILURE

        # Get fire feasibility from WeaponSystem (lock, FOV, inventory checks)
        feasibility = self.aircraft.weapons.check_fire_feasibility(target)
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        if not feasibility["can_fire"]:
            print(f"Aircraft {aircraft_id}: L&D_VETO - {feasibility.get('veto_reason', 'unknown')}")
            return NodeStatus.FAILURE

        # WeaponSystem says OK - now check tactical criteria
        nez_out_range = context.get("nez_out_range_m", 0)
        r_aero = context.get("r_aero_m", 100000)

        # Tactical range envelope
        radar_max_range = (
            getattr(self.aircraft.radar, "max_range_m", 100000)
            if hasattr(self.aircraft, "radar")
            else 100000
        )
        max_fire_range = min(max(nez_out_range, r_aero * 0.9), radar_max_range * 0.95)

        # Tactical criteria
        range_ok = target_range <= max_fire_range
        closure_ok = closure_rate > 10

        in_launch_range = range_ok and closure_ok

        target_range_km = target_range / 1000
        print(
            f"Aircraft {aircraft_id}: LAUNCH_AND_DECIDE - Range: {target_range_km:.1f}km, MaxFire: {max_fire_range / 1000:.1f}km, "
            f"WpnSys: OK, Range: {'OK' if range_ok else 'FAIL'}, Closure: {'OK' if closure_ok else 'FAIL'}, "
            f"Tactical: {'OK' if in_launch_range else 'FAIL'}"
        )

        if in_launch_range:
            # Fire missile
            context["fire_missile"] = True
            print(f"Aircraft {aircraft_id}: FIRING MISSILE - Launch and Decide!")

            # Get target bearing from ObservationHelper
            geom = self.obs_helper.get_geometry_kinematics(target)
            if not geom["valid"]:
                return NodeStatus.FAILURE
            target_bearing = geom["bearing_deg"]

            # Check if we need to point at target first (from behavior tree)
            if context.get("pre_launch_pointing", False):
                # Still pointing at target - maintain target bearing for accurate launch
                context["desired_yaw"] = target_bearing
                context["desired_pitch"] = 0.0  # Level flight for stable launch
                context["desired_throttle"] = 0.75  # Maintain speed
                print(f"Aircraft {aircraft_id}: Pointing at target for accurate missile launch")
            else:
                # Already pointed at target - continue approach with slight offset for flexibility
                approach_bearing = target_bearing + np.random.choice([-15, 15])
                approach_bearing = normalize_angle(approach_bearing)
                context["desired_yaw"] = approach_bearing
                context["desired_pitch"] = 2.0  # Slight climb for energy
                context["desired_throttle"] = 0.9  # High throttle
                print(f"Aircraft {aircraft_id}: Continuing approach after launch")

            self.current_maneuver = "banzai"
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.FAILURE

    def execute_approach(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute approach to target for BVR engagement.

        Includes tactical considerations for aspect angle and energy management.
        """
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Get approach geometry from ObservationHelper
        geom = self.obs_helper.get_geometry_kinematics(target)
        if not geom["valid"]:
            return NodeStatus.FAILURE

        target_bearing = geom["bearing_deg"]
        target_range = geom["slant_range_m"]
        aspect_angle = geom["aspect_deg"]

        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # Adjust approach based on aspect angle and actual missile ranges
        max_range = self.missile_params["max_range_m"]
        nez_range = self.missile_params["nez_range_m"]
        min_range = self.missile_params["min_range_m"]

        print(
            f"Aircraft {aircraft_id}: BVR_APPROACH - Range: {target_range / 1000:.1f}km, MaxRange: {max_range / 1000:.1f}km, "
            f"NEZRange: {nez_range / 1000:.1f}km, Bearing: {target_bearing:.1f}deg, Aspect: {aspect_angle:.1f}deg"
        )

        if target_range > max_range * 1.1:  # Beyond maximum missile range
            # Direct approach for very long range
            desired_yaw = target_bearing
            desired_pitch = self._calculate_optimal_climb_angle(target_range)
            throttle = 0.9
        elif target_range > nez_range:  # Outside NEZ but within missile range
            # Consider aspect angle for approach
            if abs(aspect_angle) < 30:  # Head-on
                # Slight offset to improve missile kinematics
                offset = 15 if aspect_angle >= 0 else -15
                desired_yaw = normalize_angle(target_bearing + offset)
            else:
                # Flanking approach for better missile performance
                desired_yaw = target_bearing

            desired_pitch = 2.0  # Slight climb for energy advantage
            throttle = 0.8
        elif target_range > min_range * 5:  # Within NEZ but not too close
            # More aggressive positioning
            desired_yaw = target_bearing
            desired_pitch = 0.0  # Level approach
            throttle = 0.85
        else:  # Close range - Within minimum effective range
            # Prepare for potential defensive maneuvers or WVR transition
            desired_yaw = target_bearing
            desired_pitch = -2.0  # Slight dive for energy
            throttle = 0.9

        context["desired_yaw"] = desired_yaw
        context["desired_pitch"] = desired_pitch
        context["desired_throttle"] = throttle

        self.current_maneuver = "approach"
        return NodeStatus.SUCCESS

    def execute_notch_maneuver(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute notch maneuver - turn 90 degrees to incoming missile.

        Used to defeat Doppler radar by presenting zero closure rate.
        """
        threat = context.get("primary_missile_threat")
        if not threat:
            return NodeStatus.FAILURE

        # Get threat geometry from ObservationHelper
        geom = self.obs_helper.get_geometry_kinematics(threat)
        if not geom["valid"]:
            return NodeStatus.FAILURE

        threat_bearing = geom["bearing_deg"]

        # Notch is 90 degrees perpendicular to threat
        current_yaw = self.aircraft.yaw_deg
        left_notch = normalize_angle(threat_bearing - 90)
        right_notch = normalize_angle(threat_bearing + 90)

        # Choose closer notch direction
        left_diff = abs(signed_yaw_deg_diff(current_yaw, left_notch))
        right_diff = abs(signed_yaw_deg_diff(current_yaw, right_notch))

        desired_yaw = left_notch if left_diff < right_diff else right_notch

        context["desired_yaw"] = desired_yaw
        context["desired_pitch"] = 0.0  # Level notch
        context["desired_throttle"] = 1.0  # Full throttle

        self.current_maneuver = "notch"
        return NodeStatus.SUCCESS

    def execute_beam_maneuver(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute beam maneuver - fly perpendicular to threat radar.

        Similar to notch but specifically for radar threats rather than missiles.
        """
        threat = context.get("primary_radar_threat")
        if not threat:
            return NodeStatus.FAILURE

        # Get threat geometry from ObservationHelper
        geom = self.obs_helper.get_geometry_kinematics(threat)
        if not geom["valid"]:
            return NodeStatus.FAILURE

        threat_bearing = geom["bearing_deg"]

        # Beam is perpendicular to threat radar
        current_yaw = self.aircraft.yaw_deg
        left_beam = normalize_angle(threat_bearing - 90)
        right_beam = normalize_angle(threat_bearing + 90)

        # Choose beam direction based on tactical situation
        # Prefer beaming away from other threats
        other_threats = context.get("threats", [])
        if len(other_threats) > 1:
            # Complex threat environment - choose safer beam direction
            # This would need more sophisticated threat analysis
            desired_yaw = right_beam  # Default choice
        else:
            # Simple threat - choose closer beam
            left_diff = abs(signed_yaw_deg_diff(current_yaw, left_beam))
            right_diff = abs(signed_yaw_deg_diff(current_yaw, right_beam))
            desired_yaw = left_beam if left_diff < right_diff else right_beam

        context["desired_yaw"] = desired_yaw
        context["desired_pitch"] = 0.0  # Level beam
        context["desired_throttle"] = 0.9  # High throttle

        self.current_maneuver = "beam"
        return NodeStatus.SUCCESS

    def _calculate_aspect_angle(self, target_bearing: float, target_heading: float) -> float:
        """
        Calculate aspect angle - angle between target's nose and our bearing to target.

        0 degrees = head-on, 90 degrees = beam aspect, 180 degrees = tail aspect
        """
        # Bearing from target to us
        reciprocal_bearing = normalize_angle(target_bearing + 180)

        # Aspect angle is difference between target heading and bearing to us
        aspect_angle = signed_yaw_deg_diff(target_heading, reciprocal_bearing)

        return abs(aspect_angle)

    def _calculate_optimal_climb_angle(self, target_range: float) -> float:
        """Calculate optimal climb angle for BVR approach based on range."""
        if target_range > 50000:  # Very long range
            return 10.0  # Steep climb for altitude advantage
        elif target_range > 30000:  # Long range
            return 5.0  # Moderate climb
        else:  # Medium range
            return 2.0  # Slight climb

    def _calculate_f_pole_distance(self, context: dict[str, Any]) -> float:
        """
        Calculate F-pole distance using actual missile active range.

        This is critical for determining when to turn defensive.
        """
        # Use actual missile active seeker range
        missile_active_range = self.missile_params["active_range_m"]
        target_range = context.get("target_range_m", float("inf"))

        # F-pole occurs when missile is within active seeker range of target
        f_pole_distance = max(0, target_range - missile_active_range)

        return f_pole_distance

    def get_current_maneuver(self) -> str:
        """Get the currently executing maneuver."""
        return self.current_maneuver or "none"

    def reset_maneuver(self):
        """Reset current maneuver state."""
        self.current_maneuver = None
        self.maneuver_start_time = 0.0
