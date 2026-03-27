"""
Geometry calculator for air-to-air combat engagement geometry.
Handles aspect angles, attack geometry, and engagement envelope calculations.
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.simulator.utils.angles import normalize_angle, signed_yaw_deg_diff
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class GeometryCalculator:
    """Calculates engagement geometry for air-to-air combat."""

    def __init__(self, aircraft):
        self.aircraft = aircraft

    def calculate_aspect_angle(self, target) -> float:
        """
        Calculate target aspect angle.

        Aspect angle is the angle between the target's heading and the line from target to us.
        0° = head-on, 90° = beam, 180° = tail aspect
        """
        if not target or not hasattr(target, "position") or not hasattr(target, "yaw_deg"):
            return 0.0

        # Calculate bearing from target to us
        bearing_to_us = geodetic_bearing_deg(
            target.position.lat,
            target.position.lon,
            self.aircraft.position.lat,
            self.aircraft.position.lon,
        )

        # Aspect angle is the difference between target heading and bearing to us
        target_heading = target.yaw_deg
        aspect_angle = abs(signed_yaw_deg_diff(target_heading, bearing_to_us))

        return aspect_angle

    def calculate_angle_off_tail(self, target) -> float:
        """
        Calculate angle off target's tail (AOT).

        0° = directly behind target, 180° = head-on
        """
        if not target:
            return 180.0

        aspect_angle = self.calculate_aspect_angle(target)
        angle_off_tail = 180.0 - aspect_angle

        return angle_off_tail

    def calculate_antenna_train_angle(self, target) -> float:
        """
        Calculate antenna train angle (ATA) - angle between our nose and target.

        Also known as angle off nose or bearing to target.
        """
        if not target:
            return 0.0

        target_bearing = geodetic_bearing_deg(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            target.position.lat,
            target.position.lon,
        )

        our_heading = self.aircraft.yaw_deg
        ata = abs(signed_yaw_deg_diff(our_heading, target_bearing))

        return ata

    def calculate_closure_rate(self, target) -> float:
        """
        Calculate closure rate with target.

        Positive = closing, negative = opening
        """
        if not target or not hasattr(target, "velocity"):
            return 0.0

        try:
            # Get velocity components
            our_vel = getattr(self.aircraft, "velocity", None)
            target_vel = getattr(target, "velocity", None)

            if not our_vel or not target_vel:
                return 0.0

            # Calculate relative velocity vector
            rel_vx = target_vel.vx - our_vel.vx
            rel_vy = target_vel.vy - our_vel.vy

            # Calculate line of sight vector
            target_bearing = geodetic_bearing_deg(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                target.position.lat,
                target.position.lon,
            )

            los_x = np.sin(np.radians(target_bearing))
            los_y = np.cos(np.radians(target_bearing))

            # Project relative velocity onto line of sight
            closure_rate = -(rel_vx * los_x + rel_vy * los_y)  # Negative for closing

            return closure_rate

        except Exception:
            return 0.0

    def calculate_line_of_sight_rate(self, target) -> float:
        """
        Calculate line of sight angular rate.

        Important for missile guidance and tracking.
        """
        # This would require tracking bearing history over time
        # For now, return 0 as placeholder
        return 0.0

    def is_in_radar_gimbal_limits(self, target) -> bool:
        """Check if target is within radar gimbal limits."""
        ata = self.calculate_antenna_train_angle(target)
        radar_fov = getattr(self.aircraft.radar, "h_fov_deg", 60) / 2

        return ata <= radar_fov

    def calculate_engagement_envelope(self, target) -> dict[str, Any]:
        """
        Calculate engagement envelope relative to target.

        Returns envelope information including optimal attack vectors.
        """
        if not target:
            return {}

        aspect_angle = self.calculate_aspect_angle(target)
        angle_off_tail = self.calculate_angle_off_tail(target)
        ata = self.calculate_antenna_train_angle(target)
        closure_rate = self.calculate_closure_rate(target)

        target_range = (
            geodetic_distance_km(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                self.aircraft.position.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            * 1000
        )  # Convert to meters

        # Classify engagement geometry
        if aspect_angle < 30:
            geometry_type = "head_on"
            tactical_advantage = "neutral"
        elif aspect_angle < 60:
            geometry_type = "oblique"
            tactical_advantage = "slight" if angle_off_tail < 90 else "slight_disadvantage"
        elif aspect_angle < 120:
            geometry_type = "beam"
            tactical_advantage = "good" if angle_off_tail < 90 else "disadvantage"
        else:
            geometry_type = "tail_aspect"
            tactical_advantage = "excellent" if angle_off_tail < 60 else "good"

        # Calculate optimal attack vector
        optimal_vector = self._calculate_optimal_attack_vector(target, aspect_angle, target_range)

        return {
            "aspect_angle_deg": aspect_angle,
            "angle_off_tail_deg": angle_off_tail,
            "antenna_train_angle_deg": ata,
            "closure_rate_mps": closure_rate,
            "target_range_m": target_range,
            "geometry_type": geometry_type,
            "tactical_advantage": tactical_advantage,
            "optimal_attack_vector": optimal_vector,
            "in_radar_gimbal": self.is_in_radar_gimbal_limits(target),
        }

    def _calculate_optimal_attack_vector(
        self, target, aspect_angle: float, target_range: float
    ) -> dict[str, float]:
        """Calculate optimal attack vector for engagement."""

        # Base optimal approach
        if target_range > 30000:  # Long range BVR
            if aspect_angle < 45:  # Head-on
                # Slight offset for better missile kinematics
                optimal_offset = 15.0
            else:
                # Use current geometry
                optimal_offset = 0.0
        elif target_range > 10000:  # Medium range
            if aspect_angle > 135:  # Tail aspect
                # Direct stern attack
                optimal_offset = 0.0
            else:
                # Beam attack for better missile performance
                optimal_offset = 20.0
        else:  # Short range WVR
            # Position for guns or short-range missiles
            optimal_offset = 0.0

        # Calculate optimal bearing
        current_bearing = geodetic_bearing_deg(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            target.position.lat,
            target.position.lon,
        )

        optimal_bearing = normalize_angle(current_bearing + optimal_offset)

        return {
            "optimal_bearing_deg": optimal_bearing,
            "offset_from_direct_deg": optimal_offset,
            "recommended_range_m": min(target_range, 25000),  # Optimal engagement range
        }

    def calculate_sqi_geometry(self, target, missile_target=None) -> dict[str, Any]:
        """
        Calculate SQI (Target, Interceptor, Predicted Waypoint Intersection) geometry.

        This is used for missile guidance and engagement planning.
        """
        if not target:
            return {}

        # Use existing SQI calculation if available
        if hasattr(self.aircraft, "wez") and hasattr(self.aircraft.wez, "sqi"):
            try:
                sqi_value = self.aircraft.wez.sqi(self.aircraft, target, missile_target, None)
                return {"sqi": sqi_value}
            except Exception:
                pass

        # Fallback calculation
        aspect_angle = self.calculate_aspect_angle(target)
        closure_rate = self.calculate_closure_rate(target)

        # Simplified SQI approximation
        if closure_rate > 0:  # Closing
            sqi_factor = min(1.0, closure_rate / 300.0)  # Normalize to typical closure rates
        else:
            sqi_factor = 0.0

        # Adjust for aspect angle (head-on is better for SQI)
        aspect_factor = 1.0 - (aspect_angle / 180.0)

        sqi_estimate = sqi_factor * aspect_factor

        return {
            "sqi": sqi_estimate,
            "closure_component": sqi_factor,
            "aspect_component": aspect_factor,
        }

    def calculate_lead_pursuit_angle(self, target) -> float:
        """
        Calculate lead pursuit angle for intercept geometry.

        Used for guns and close-in combat.
        """
        if not target:
            return 0.0

        closure_rate = self.calculate_closure_rate(target)
        target_range = (
            geodetic_distance_km(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                self.aircraft.position.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            * 1000
        )

        if target_range <= 0 or not hasattr(target, "speed"):
            return 0.0

        # Calculate time to intercept
        if closure_rate > 0:
            time_to_intercept = target_range / closure_rate
        else:
            return 0.0  # Opening, no intercept

        # Calculate target angular velocity (simplified)
        target_speed = getattr(target, "speed", 0)
        target_angular_velocity = target_speed / target_range  # rad/s

        # Lead angle = angular velocity * time to intercept
        lead_angle_rad = target_angular_velocity * time_to_intercept
        lead_angle_deg = np.degrees(lead_angle_rad)

        # Limit to reasonable values
        return np.clip(lead_angle_deg, -45.0, 45.0)

    def get_geometry_summary(self, target) -> dict[str, Any]:
        """Get comprehensive geometry summary for target."""
        if not target:
            return {"has_target": False}

        envelope = self.calculate_engagement_envelope(target)
        sqi = self.calculate_sqi_geometry(target)
        lead_angle = self.calculate_lead_pursuit_angle(target)

        return {
            "has_target": True,
            "engagement_envelope": envelope,
            "sqi_geometry": sqi,
            "lead_pursuit_angle_deg": lead_angle,
            "radar_gimbal_status": "in_limits"
            if envelope.get("in_radar_gimbal", False)
            else "out_of_limits",
        }
