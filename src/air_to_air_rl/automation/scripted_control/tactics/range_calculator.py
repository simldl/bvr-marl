"""
Range calculator for missile engagement envelopes and critical ranges.
Handles NEZ, MAR, F-pole, A-pole calculations for BVR combat.
Now uses the proper NoEscapeZoneCalculator and TrackPrioritySystem.
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.aircrafts.core.nez import NoEscapeZoneCalculator
from air_to_air_rl.aircrafts.core.target_prio import TrackPrioritySystem
from air_to_air_rl.simulator.utils.geodesics import geodetic_distance_km


class RangeCalculator:
    """Calculates critical ranges for missile engagement using proper fighter systems."""

    def __init__(self, aircraft):
        self.aircraft = aircraft

        # Initialize proper fighter systems
        self.nez_calculator = NoEscapeZoneCalculator(aircraft)
        self.track_priority = TrackPrioritySystem(aircraft)

        # Get actual missile parameters from aircraft configuration
        self.missile_params = self._get_missile_params_from_aircraft()

    def _get_missile_params_from_aircraft(self) -> dict[str, Any]:
        """Get missile parameters from the aircraft's actual missile types."""
        default_params = {
            "max_range_m": 100000,  # 100km BVR range
            "min_range_m": 1000,  # Default minimum range
            "active_range_m": 100000,  # 100km active seeker range
            "guidance_type": "active_radar",
            "fox_type": 3,
        }

        # Try to get actual missile parameters from aircraft missile types
        if hasattr(self.aircraft, "missile_types") and self.aircraft.missile_types:
            try:
                # Get the first missile type and extract its parameters
                missile_type = self.aircraft.missile_types[0]
                # Create temporary missile instance to get parameters
                temp_missile = missile_type(
                    None, None, self.aircraft, self.aircraft.map_limits, self.aircraft.group
                )

                missile_range = getattr(temp_missile.radar, "max_range_m", 100000)
                default_params.update(
                    {
                        "max_range_m": missile_range,
                        "min_range_m": getattr(temp_missile, "min_range_m", 1000),
                        "active_range_m": missile_range,
                        "fox_type": getattr(temp_missile, "fox_type", 3),
                        "seeker_sensitivity": getattr(temp_missile, "seeker_sensitivity", 1.0),
                    }
                )
                # Debug: Verify missile parameters are loaded correctly
                print(
                    f"RangeCalculator: Missile params loaded - Max Range: {missile_range / 1000:.1f}km, Fox Type: {getattr(temp_missile, 'fox_type', 3)}"
                )
            except Exception:
                # Fall back to defaults if missile creation fails
                pass

        return default_params

    def calculate_nez_ranges(self, target, missile_type: str | None = None) -> dict[str, float]:
        """
        Calculate No Escape Zone (NEZ) ranges using the proper NoEscapeZoneCalculator.

        NEZ is the range at which a missile will hit even if target goes defensive immediately.
        """
        if not target:
            return {"nez_in_m": 0.0, "nez_out_m": 0.0, "nez_valid": False}

        try:
            # Use the proper NoEscapeZoneCalculator to compute DLZ (Dynamic Launch Zone)
            dlz = self.nez_calculator.compute_dlz(target)

            return {
                "nez_in_m": dlz.r_nez_in_m,
                "nez_out_m": dlz.r_nez_out_m,
                "nez_valid": True,
                "r_min_m": dlz.r_min_m,
                "r_tr_m": dlz.r_tr_m,  # Turning target range
                "r_pi_m": dlz.r_pi_m,  # Non-maneuvering target range
                "r_aero_m": dlz.r_aero_m,  # Aerodynamic limit range
                "has_dlz": True,
            }
        except Exception:
            # Fallback to basic calculation if DLZ computation fails
            return self._calculate_fallback_nez(target)

    def _calculate_fallback_nez(self, target) -> dict[str, float]:
        """Fallback NEZ calculation using missile parameters."""
        geodetic_distance_km(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            self.aircraft.position.alt,
            target.position.lat,
            target.position.lon,
            target.position.alt,
        ) * 1000

        # Use actual missile parameters
        max_range = self.missile_params["max_range_m"]
        min_range = self.missile_params["min_range_m"]

        # Calculate aspect angle for NEZ adjustment
        aspect_angle = self._calculate_aspect_angle(target)
        altitude_advantage = self.aircraft.position.alt - getattr(target.position, "alt", 0)
        speed_advantage = self.aircraft.speed - getattr(target, "speed", 300)

        # Base NEZ ranges based on actual missile range
        base_nez_out = max_range * 0.6  # 60% of max range for NEZ outer
        base_nez_in = min_range * 10  # 10x min range for NEZ inner

        # Adjust for aspect angle
        if aspect_angle < 30:  # Head-on
            nez_multiplier = 0.8  # Reduced NEZ head-on
        elif aspect_angle > 150:  # Tail aspect
            nez_multiplier = 1.3  # Increased NEZ tail aspect
        else:  # Beam
            nez_multiplier = 1.0

        # Adjust for energy advantages
        altitude_factor = 1.0 + (altitude_advantage / 10000) * 0.2
        speed_factor = 1.0 + (speed_advantage / 100) * 0.1
        energy_multiplier = np.clip(altitude_factor * speed_factor, 0.5, 2.0)

        nez_out = base_nez_out * nez_multiplier * energy_multiplier
        nez_in = base_nez_in * nez_multiplier * 0.5

        return {
            "nez_in_m": max(min_range, nez_in),
            "nez_out_m": min(max_range, nez_out),
            "nez_valid": True,
            "has_dlz": False,
            # Add generous fallback ranges for aggressive firing
            "r_aero_m": max_range,
            "r_tr_m": max(nez_out * 0.8, 50000),  # At least 50km turning target range
            "r_pi_m": max(nez_out * 1.2, 70000),  # At least 70km non-maneuvering range
        }

    def calculate_mar_range(self, threat_aircraft) -> float:
        """
        Calculate Minimum Abort Range (MAR).

        MAR is the minimum range at which we can successfully escape an enemy missile
        if we turn cold (away) immediately.
        """
        if not threat_aircraft:
            return 0.0

        # Get threat missile parameters
        threat_missile_range = 40000  # Assume modern BVR missile

        # Calculate range to threat
        geodetic_distance_km(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            self.aircraft.position.alt,
            threat_aircraft.position.lat,
            threat_aircraft.position.lon,
            threat_aircraft.position.alt,
        ) * 1000

        # MAR calculation factors
        our_speed = self.aircraft.speed
        threat_aspect = self._calculate_aspect_angle(threat_aircraft)

        # Base MAR is roughly 60-70% of threat missile range
        base_mar = threat_missile_range * 0.65

        # Adjust for our speed (faster = smaller MAR)
        speed_factor = max(0.7, 300 / our_speed)  # Normalize to 300 m/s baseline

        # Adjust for aspect angle
        if threat_aspect < 30:  # Head-on threat
            aspect_factor = 1.2  # Need more distance to escape head-on
        else:
            aspect_factor = 1.0

        mar = base_mar * speed_factor * aspect_factor

        return min(mar, 45000)  # Cap at 45km

    def calculate_f_pole_range(self, target, missile_type: str | None = None) -> float:
        """
        Calculate F-pole range using actual missile parameters.

        F-pole is the range between aircraft when the launched missile goes active (pitbull).
        """
        # Use actual missile active seeker range
        active_range = self.missile_params["active_range_m"]

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

        # F-pole occurs when missile is within active seeker range of target
        f_pole = max(0, target_range - active_range)

        return f_pole

    def calculate_a_pole_range(self, target, missile_type: str | None = None) -> float:
        """
        Calculate A-pole range using actual missile guidance type.

        A-pole is the range at which the missile impacts the target.
        Most relevant for SARH missiles that require continuous illumination.
        """
        # Use actual missile guidance type and fox type
        guidance_type = self.missile_params["guidance_type"]
        self.missile_params["fox_type"]

        if "sarh" in guidance_type.lower() or "semi" in guidance_type.lower():
            return 0.0  # Must maintain lock until impact
        else:
            return 0.0  # Active missiles don't require lock at impact

    def calculate_launch_envelope(self, target) -> dict[str, Any]:
        """
        Calculate complete launch envelope for target.

        Returns all critical ranges and engagement parameters.
        """
        if not target:
            return {"valid": False}

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

        # Calculate all critical ranges using proper systems
        nez_ranges = self.calculate_nez_ranges(target)
        f_pole = self.calculate_f_pole_range(target)
        a_pole = self.calculate_a_pole_range(target)

        # DLZ data is now included in NEZ ranges if available
        dlz_data = {
            "r_min_m": nez_ranges.get("r_min_m", self.missile_params["min_range_m"]),
            "r_tr_m": nez_ranges.get("r_tr_m", nez_ranges["nez_out_m"]),
            "r_pi_m": nez_ranges.get("r_pi_m", nez_ranges["nez_out_m"] * 1.2),
            "r_aero_m": nez_ranges.get("r_aero_m", self.missile_params["max_range_m"]),
            "has_dlz": nez_ranges.get("has_dlz", False),
        }

        # Determine engagement zones
        if target_range < nez_ranges["nez_in_m"]:
            engagement_zone = "inside_nez"
            shot_quality = "poor"  # Too close
        elif target_range <= nez_ranges["nez_out_m"]:
            engagement_zone = "in_nez"
            shot_quality = "excellent"
        elif target_range <= dlz_data.get("r_aero_m", 50000):
            engagement_zone = "in_launch_envelope"
            shot_quality = "good"
        elif target_range <= dlz_data.get("r_aero_m", 60000) * 1.2:
            engagement_zone = "extended_range"
            shot_quality = "marginal"
        else:
            engagement_zone = "out_of_range"
            shot_quality = "no_shot"

        return {
            "valid": True,
            "target_range_m": target_range,
            "nez_ranges": nez_ranges,
            "f_pole_m": f_pole,
            "a_pole_m": a_pole,
            "dlz_data": dlz_data,
            "engagement_zone": engagement_zone,
            "shot_quality": shot_quality,
            "recommended_launch": shot_quality in ["excellent", "good"],
        }

    def calculate_desired_out_range(self, threats) -> float:
        """
        Calculate Desired Out Range (DOR) based on threat situation.

        DOR is the range at which we want to be "out" (turned away) from threats.
        """
        if not threats:
            return 20000  # Default 20km

        # Find the most threatening missile
        closest_missile_range = float("inf")
        highest_threat_range = float("inf")

        for threat in threats:
            if threat.threat_type == "missile":
                closest_missile_range = min(closest_missile_range, threat.distance_m)
            elif threat.threat_level == "high":
                highest_threat_range = min(highest_threat_range, threat.distance_m)

        # DOR should be outside the threat NEZ but not too far
        if closest_missile_range < float("inf"):
            # Active missile threat - need to be out soon
            dor = max(15000, closest_missile_range * 1.5)
        elif highest_threat_range < float("inf"):
            # High threat aircraft - plan exit range
            dor = max(20000, highest_threat_range * 0.6)
        else:
            dor = 25000  # Standard DOR

        return min(dor, 40000)  # Cap at 40km

    def _calculate_aspect_angle(self, target) -> float:
        """Calculate aspect angle to target."""
        if not target or not hasattr(target, "yaw_deg"):
            return 0.0

        from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff
        from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg

        bearing_to_us = geodetic_bearing_deg(
            target.position.lat,
            target.position.lon,
            self.aircraft.position.lat,
            self.aircraft.position.lon,
        )

        aspect_angle = abs(signed_yaw_deg_diff(target.yaw_deg, bearing_to_us))
        return aspect_angle

    def get_range_summary(self, target, threats=None) -> dict[str, Any]:
        """Get comprehensive range summary."""
        if not target:
            return {"has_target": False}

        envelope = self.calculate_launch_envelope(target)
        dor = self.calculate_desired_out_range(threats or [])

        # Calculate MAR for primary threat
        mar = 0.0
        if threats:
            primary_threat = max(threats, key=lambda t: 1 if t.threat_level == "high" else 0)
            if primary_threat and hasattr(primary_threat, "source"):
                mar = self.calculate_mar_range(primary_threat.source)

        return {
            "has_target": True,
            "launch_envelope": envelope,
            "desired_out_range_m": dor,
            "minimum_abort_range_m": mar,
            "tactical_situation": self._assess_tactical_situation(envelope, dor, mar),
        }

    def _assess_tactical_situation(self, envelope: dict, dor: float, mar: float) -> str:
        """Assess overall tactical situation based on ranges."""
        if not envelope.get("valid", False):
            return "no_target"

        target_range = envelope["target_range_m"]
        shot_quality = envelope["shot_quality"]

        if target_range < mar and mar > 0:
            return "defensive_immediate"  # Inside MAR
        elif shot_quality == "excellent":
            return "offensive_optimal"  # In NEZ
        elif shot_quality in ["good", "marginal"]:
            return "offensive_viable"  # Can shoot
        elif target_range > dor:
            return "approach"  # Need to close
        else:
            return "positioning"  # Maneuvering for advantage
