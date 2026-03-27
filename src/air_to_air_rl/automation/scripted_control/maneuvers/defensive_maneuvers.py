"""
Defensive maneuvers for air-to-air combat.
Includes notch, beam, split-S, and other defensive tactics.
"""

from typing import Any

import numpy as np

from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus
from air_to_air_rl.simulator.utils.angles import normalize_angle


class DefensiveManeuvers:
    """Collection of defensive maneuvers for threat evasion."""

    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.maneuver_start_time = 0.0
        self.current_defensive_maneuver = None

    def execute_notch_90(self, threat_bearing: float, context: dict[str, Any]) -> NodeStatus:
        """
        Execute 90-degree notch maneuver.

        Turn 90 degrees perpendicular to incoming threat to defeat Doppler radar.
        """
        current_yaw = self.aircraft.yaw_deg

        # Calculate left and right notch headings
        left_notch = normalize_angle(threat_bearing - 90)
        right_notch = normalize_angle(threat_bearing + 90)

        # Choose the closer notch direction
        from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff

        left_diff = abs(signed_yaw_deg_diff(current_yaw, left_notch))
        right_diff = abs(signed_yaw_deg_diff(current_yaw, right_notch))

        notch_heading = left_notch if left_diff < right_diff else right_notch

        # Execute notch
        context["desired_yaw"] = notch_heading
        context["desired_pitch"] = 0.0  # Level notch
        context["desired_throttle"] = 1.0  # Full throttle

        self.current_defensive_maneuver = "notch_90"
        return NodeStatus.SUCCESS

    def execute_beam_maneuver(self, threat_bearing: float, context: dict[str, Any]) -> NodeStatus:
        """
        Execute beam maneuver - fly perpendicular to threat radar.

        Similar to notch but optimized for radar evasion rather than missile evasion.
        """
        # Beam maneuver is perpendicular to threat
        current_yaw = self.aircraft.yaw_deg

        left_beam = normalize_angle(threat_bearing - 90)
        right_beam = normalize_angle(threat_bearing + 90)

        # Choose beam direction based on terrain and other threats
        # For now, choose the direction that keeps us away from map boundaries
        map_limits = getattr(self.aircraft, "map_limits", None)
        if map_limits:
            # Simple boundary avoidance
            our_lat = self.aircraft.position.lat
            our_lon = self.aircraft.position.lon

            center_lat = (map_limits.top_lat + map_limits.bottom_lat) / 2
            center_lon = (map_limits.left_lon + map_limits.right_lon) / 2

            # Choose beam direction toward center of map
            if our_lat < center_lat and our_lon < center_lon:
                beam_heading = right_beam  # Prefer right beam
            else:
                beam_heading = left_beam  # Prefer left beam
        else:
            # Default: choose closer beam
            from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff

            left_diff = abs(signed_yaw_deg_diff(current_yaw, left_beam))
            right_diff = abs(signed_yaw_deg_diff(current_yaw, right_beam))
            beam_heading = left_beam if left_diff < right_diff else right_beam

        context["desired_yaw"] = beam_heading
        context["desired_pitch"] = 0.0
        context["desired_throttle"] = 0.95

        self.current_defensive_maneuver = "beam"
        return NodeStatus.SUCCESS

    def execute_split_s(self, escape_heading: float, context: dict[str, Any]) -> NodeStatus:
        """
        Execute split-S maneuver.

        Half roll followed by pull through to reverse direction and lose altitude.
        """
        current_altitude = self.aircraft.position.alt

        # Don't execute split-S if too low
        if current_altitude < 3000:  # 3km minimum altitude
            return NodeStatus.FAILURE

        # Split-S involves rolling inverted then pulling through
        # Simplified as: turn to escape heading with dive
        context["desired_yaw"] = escape_heading
        context["desired_pitch"] = -20.0  # Steep dive
        context["desired_throttle"] = 0.8  # Moderate throttle in dive

        self.current_defensive_maneuver = "split_s"
        return NodeStatus.SUCCESS

    def execute_barrel_roll(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute barrel roll defensive maneuver.

        Helical path that changes aspect angle while maintaining general direction.
        """
        # Simplified barrel roll: climb with turn
        current_yaw = self.aircraft.yaw_deg

        # Add 45-degree turn component to barrel roll
        roll_heading = normalize_angle(current_yaw + 45)

        context["desired_yaw"] = roll_heading
        context["desired_pitch"] = 10.0  # Climb component
        context["desired_throttle"] = 0.9

        self.current_defensive_maneuver = "barrel_roll"
        return NodeStatus.SUCCESS

    def execute_defensive_spiral(
        self, threat_bearing: float, context: dict[str, Any]
    ) -> NodeStatus:
        """
        Execute defensive spiral.

        Continuous turn with altitude changes to defeat tracking.
        """
        current_yaw = self.aircraft.yaw_deg

        # Spiral away from threat
        spiral_rate = 30  # degrees per second (simplified)
        spiral_heading = normalize_angle(current_yaw + spiral_rate)

        # Alternate climb and dive in spiral
        sim_time = context.get("sim_time", 0)
        if int(sim_time) % 10 < 5:  # 5-second intervals
            pitch = 15.0  # Climb phase
        else:
            pitch = -10.0  # Dive phase

        context["desired_yaw"] = spiral_heading
        context["desired_pitch"] = pitch
        context["desired_throttle"] = 0.85

        self.current_defensive_maneuver = "defensive_spiral"
        return NodeStatus.SUCCESS

    def execute_high_g_turn(self, escape_heading: float, context: dict[str, Any]) -> NodeStatus:
        """
        Execute high-G defensive turn.

        Maximum performance turn to defeat incoming missile.
        """
        # Check if we have energy for high-G turn
        current_speed = self.aircraft.speed
        if current_speed < 200:  # Need minimum speed for high-G
            return NodeStatus.FAILURE

        context["desired_yaw"] = escape_heading
        context["desired_pitch"] = 0.0  # Level turn for maximum G
        context["desired_throttle"] = 1.0  # Full throttle

        self.current_defensive_maneuver = "high_g_turn"
        return NodeStatus.SUCCESS

    def execute_terrain_masking(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute terrain masking maneuver.

        Descend to low altitude to break radar locks.
        """
        current_altitude = self.aircraft.position.alt
        min_safe_altitude = getattr(self.aircraft, "min_alt_m", 500)

        if current_altitude <= min_safe_altitude + 1000:  # Too low already
            return NodeStatus.FAILURE

        # Descend to minimum safe altitude
        target_altitude = min_safe_altitude + 500  # 500m buffer

        if current_altitude > target_altitude:
            context["desired_pitch"] = -15.0  # Descent
            context["desired_throttle"] = 0.7  # Reduced power in descent
        else:
            context["desired_pitch"] = 0.0  # Level off
            context["desired_throttle"] = 0.8

        self.current_defensive_maneuver = "terrain_masking"
        return NodeStatus.SUCCESS

    def execute_chaff_corridor(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute chaff corridor maneuver.

        Fly through previously deployed chaff for additional radar masking.
        """
        # This would require tracking chaff deployment locations
        # For now, implement as a straight-line maneuver with chaff deployment

        current_yaw = self.aircraft.yaw_deg

        context["desired_yaw"] = current_yaw  # Maintain heading
        context["desired_pitch"] = 0.0
        context["desired_throttle"] = 0.9

        # Request chaff deployment (handled by countermeasure system)
        context["deploy_chaff"] = True

        self.current_defensive_maneuver = "chaff_corridor"
        return NodeStatus.SUCCESS

    def execute_multiplane_maneuver(
        self, threat_bearing: float, context: dict[str, Any]
    ) -> NodeStatus:
        """
        Execute multi-plane defensive maneuver.

        Combines horizontal and vertical maneuvers to defeat missile tracking.
        """

        # Calculate perpendicular heading to threat
        perp_heading = normalize_angle(threat_bearing + 90)

        # Add vertical component based on current altitude
        current_altitude = self.aircraft.position.alt

        if current_altitude > 8000:  # High altitude - can dive
            pitch_component = -10.0
        elif current_altitude < 3000:  # Low altitude - must climb
            pitch_component = 10.0
        else:  # Medium altitude - slight climb
            pitch_component = 5.0

        context["desired_yaw"] = perp_heading
        context["desired_pitch"] = pitch_component
        context["desired_throttle"] = 0.9

        self.current_defensive_maneuver = "multiplane"
        return NodeStatus.SUCCESS

    def select_optimal_defensive_maneuver(
        self,
        threat_type: str,
        threat_bearing: float,
        threat_distance: float,
        context: dict[str, Any],
    ) -> NodeStatus:
        """
        Select and execute the optimal defensive maneuver based on threat parameters.
        """
        if threat_type == "missile":
            if threat_distance < 2000:  # Very close missile
                return self.execute_high_g_turn(normalize_angle(threat_bearing + 180), context)
            elif threat_distance < 5000:  # Close missile
                return self.execute_notch_90(threat_bearing, context)
            else:  # Distant missile
                return self.execute_beam_maneuver(threat_bearing, context)

        elif threat_type == "radar_lock":
            return self.execute_beam_maneuver(threat_bearing, context)

        elif threat_type == "aircraft":
            if threat_distance < 3000:  # WVR threat
                return self.execute_defensive_spiral(threat_bearing, context)
            else:  # BVR threat
                return self.execute_beam_maneuver(threat_bearing, context)

        else:
            # Unknown threat - default defensive action
            return self.execute_beam_maneuver(threat_bearing, context)

    def get_current_maneuver(self) -> str:
        """Get the currently executing defensive maneuver."""
        return self.current_defensive_maneuver or "none"

    def reset_maneuver(self):
        """Reset defensive maneuver state."""
        self.current_defensive_maneuver = None
        self.maneuver_start_time = 0.0
