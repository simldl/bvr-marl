"""
Countermeasure deployment controller for the semi-automated helper.
Intelligently deploys flares, chaff, ECM, and decoys based on threat assessment.
Uses missile tracking data to deploy countermeasures at optimal timing.
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.simulator.utils.geodesics import geodetic_distance_km


class CountermeasureController:
    """Controls automatic deployment of countermeasures based on threat assessment and missile tracking."""

    def __init__(self, aircraft, config):
        self.aircraft = aircraft
        self.config = config

        # Deployment state tracking
        self.last_flare_time = 0.0
        self.last_chaff_time = 0.0
        self.last_ecm_time = 0.0
        self.last_decoy_time = 0.0

        # Minimum intervals between deployments (seconds)
        self.flare_interval = 2.0
        self.chaff_interval = 3.0
        self.ecm_interval = 5.0
        self.decoy_interval = 10.0

        # Threat tracking for smart deployment
        self.active_missile_threats = {}
        self.radar_lock_duration = {}

        # Missile impact timing thresholds (seconds before estimated impact)
        self.optimal_cm_window_min_s = 1.5  # Deploy no earlier than 1.5s before impact
        self.optimal_cm_window_max_s = 4.0  # Deploy no later than 4s before impact

    def evaluate_countermeasures(self, threats: list[Any], simulator) -> dict[str, bool]:
        """
        Evaluate whether to deploy countermeasures based on current threats.

        Args:
            threats: list of assessed threats
            simulator: The simulation environment

        Returns:
            dict with countermeasure deployment decisions
        """
        current_time = getattr(simulator, "elapsed_time", 0.0)

        # Initialize deployment decisions
        deploy = {"flares": False, "chaff": False, "ecm": False, "decoys": False}

        if not self.config.enable_countermeasures:
            return deploy

        # Analyze threats and determine appropriate countermeasures
        missile_threats = self._get_missile_threats(threats)
        radar_threats = self._get_radar_threats(threats)

        # Deploy countermeasures based on threat assessment
        deploy.update(self._evaluate_flare_deployment(missile_threats, current_time))
        deploy.update(self._evaluate_chaff_deployment(radar_threats, current_time))
        deploy.update(self._evaluate_ecm_deployment(radar_threats, current_time))
        deploy.update(self._evaluate_decoy_deployment(threats, current_time))

        # Update deployment timestamps
        if deploy["flares"]:
            self.last_flare_time = current_time
        if deploy["chaff"]:
            self.last_chaff_time = current_time
        if deploy["ecm"]:
            self.last_ecm_time = current_time
        if deploy["decoys"]:
            self.last_decoy_time = current_time

        return deploy

    def _get_missile_threats(self, threats: list[Any]) -> list[Any]:
        return [t for t in threats if getattr(t, "threat_type", "") == "missile"]

    def _get_radar_threats(self, threats: list[Any]) -> list[Any]:
        """Extract radar lock threats from the threat list."""
        return [threat for threat in threats if getattr(threat, "threat_type", "") == "radar_lock"]

    def _estimate_time_to_impact(self, missile) -> float:
        """
        Estimate time to impact from missile tracking data.

        Uses missile warner tracking data and missile kinematics to estimate
        when the missile will reach the aircraft.

        Returns:
            Time to impact in seconds, or float('inf') if cannot estimate
        """
        try:
            # Get missile position and velocity from tracking
            if hasattr(missile, "position") and hasattr(missile, "velocity"):
                missile_pos = np.array(
                    [missile.position.lat, missile.position.lon, missile.position.alt]
                )
                aircraft_pos = np.array(
                    [
                        self.aircraft.position.lat,
                        self.aircraft.position.lon,
                        self.aircraft.position.alt,
                    ]
                )

                # Calculate range
                range_m = (
                    geodetic_distance_km(
                        missile.position.lat,
                        missile.position.lon,
                        missile.position.alt,
                        self.aircraft.position.lat,
                        self.aircraft.position.lon,
                        self.aircraft.position.alt,
                    )
                    * 1000.0
                )

                # Get missile speed
                if hasattr(missile, "airspeed_mps"):
                    missile_speed = missile.airspeed_mps
                elif hasattr(missile, "velocity"):
                    missile_speed = np.linalg.norm(
                        [missile.velocity.vx, missile.velocity.vy, missile.velocity.vz]
                    )
                else:
                    # Assume typical missile speed
                    missile_speed = 600.0  # m/s

                # Get closure rate (relative velocity towards us)
                if hasattr(self.aircraft, "velocity"):
                    aircraft_vel = np.array(
                        [
                            self.aircraft.velocity.vx,
                            self.aircraft.velocity.vy,
                            self.aircraft.velocity.vz,
                        ]
                    )
                    missile_vel = (
                        np.array([missile.velocity.vx, missile.velocity.vy, missile.velocity.vz])
                        if hasattr(missile, "velocity")
                        else np.zeros(3)
                    )

                    relative_vel = missile_vel - aircraft_vel
                    # Project onto line connecting missile to aircraft
                    direction = aircraft_pos - missile_pos
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        direction /= direction_norm
                        closure_rate = np.dot(relative_vel, direction)
                    else:
                        closure_rate = missile_speed
                else:
                    closure_rate = missile_speed

                # Time to impact = range / closure rate (if closing)
                if closure_rate > 0:
                    time_to_impact = range_m / closure_rate
                    return max(0.0, time_to_impact)

            return float("inf")
        except Exception:
            # If estimation fails, use distance-based fallback
            try:
                distance = (
                    geodetic_distance_km(
                        missile.position.lat,
                        missile.position.lon,
                        missile.position.alt,
                        self.aircraft.position.lat,
                        self.aircraft.position.lon,
                        self.aircraft.position.alt,
                    )
                    * 1000.0
                )
                # Assume typical missile speed for fallback
                return distance / 600.0
            except Exception:
                return float("inf")

    def _get_missile_type(self, missile) -> str:
        """Determine missile guidance type from missile object."""
        if hasattr(missile, "guidance_type"):
            guidance = str(getattr(missile, "guidance_type", "")).lower()
            if "ir" in guidance or "heat" in guidance:
                return "fox2"  # IR-guided
            elif "radar" in guidance or "arh" in guidance:
                return "fox3"  # Active radar homing
            elif "sarh" in guidance or "semi" in guidance:
                return "fox1"  # Semi-active radar homing

        if hasattr(missile, "type"):
            missile_type = str(getattr(missile, "type", "")).lower()
            if "fox2" in missile_type or "aim-9" in missile_type or "sidewinder" in missile_type:
                return "fox2"
            elif "fox3" in missile_type or "aim-120" in missile_type or "amraam" in missile_type:
                return "fox3"
            elif "fox1" in missile_type or "aim-7" in missile_type or "sparrow" in missile_type:
                return "fox1"

        # Default assumption based on attributes
        if hasattr(missile, "seeker"):
            seeker_type = str(type(missile.seeker).__name__).lower()
            if "ir" in seeker_type:
                return "fox2"
            elif "arh" in seeker_type or "active" in seeker_type:
                return "fox3"
            elif "sarh" in seeker_type or "semi" in seeker_type:
                return "fox1"

        return "unknown"

    def _evaluate_flare_deployment(
        self, missile_threats: list[Any], current_time: float
    ) -> dict[str, bool]:
        """
        Evaluate whether to deploy flares against IR-guided missiles.
        Uses missile tracking data to deploy at optimal timing.
        """
        deploy = {"flares": False}

        if current_time - self.last_flare_time < self.flare_interval:
            return deploy

        # Check missile warner data for precise timing
        if hasattr(self.aircraft, "sensor"):
            missile_warnings = self.aircraft.sensor.get_missile_warnings()

            for missile in missile_warnings:
                missile_type = self._get_missile_type(missile)

                # Deploy flares for IR-guided missiles
                if missile_type == "fox2":
                    # Calculate time to impact
                    time_to_impact = self._estimate_time_to_impact(missile)

                    # Deploy if within optimal window
                    if (
                        self.optimal_cm_window_min_s
                        <= time_to_impact
                        <= self.optimal_cm_window_max_s
                    ):
                        deploy["flares"] = True
                        return deploy

        # Fallback to old distance-based logic if tracking unavailable
        for threat in missile_threats:
            missile = getattr(threat, "source", None)
            if not missile:
                continue

            missile_type = self._get_missile_type(missile)

            if missile_type == "fox2":
                distance = getattr(threat, "distance_m", float("inf"))

                # Deploy flares if IR missile is close
                if distance < self.config.threat_response_thresholds["incoming_missile_distance_m"]:
                    deploy["flares"] = True
                    break

        return deploy

    def _evaluate_chaff_deployment(
        self, radar_threats: list[Any], current_time: float
    ) -> dict[str, bool]:
        """
        Evaluate whether to deploy chaff against radar-guided threats.
        Uses missile tracking data to deploy at optimal timing.
        """
        deploy = {"chaff": False}

        if current_time - self.last_chaff_time < self.chaff_interval:
            return deploy

        # Check missile warner data for precise timing
        if hasattr(self.aircraft, "sensor"):
            missile_warnings = self.aircraft.sensor.get_missile_warnings()

            for missile in missile_warnings:
                missile_type = self._get_missile_type(missile)

                # Deploy chaff for radar-guided missiles (Fox-1 and Fox-3)
                if missile_type in ["fox1", "fox3"]:
                    # Calculate time to impact
                    time_to_impact = self._estimate_time_to_impact(missile)

                    # Deploy if within optimal window
                    if (
                        self.optimal_cm_window_min_s
                        <= time_to_impact
                        <= self.optimal_cm_window_max_s
                    ):
                        deploy["chaff"] = True
                        return deploy

        # Fallback: Deploy chaff for radar-guided missile threats
        for threat in radar_threats:
            missile = getattr(threat, "source", None)
            if not missile:
                continue

            missile_type = self._get_missile_type(missile)

            if missile_type in ["fox1", "fox3"]:
                distance = getattr(threat, "distance_m", float("inf"))

                # Deploy chaff if radar missile is close
                if distance < self.config.threat_response_thresholds["incoming_missile_distance_m"]:
                    deploy["chaff"] = True
                    break

        # Also deploy chaff if we're being radar locked for too long
        lock_time = getattr(radar_threats[0] if radar_threats else None, "lock_duration", 0.0)
        if lock_time > self.config.threat_response_thresholds["radar_lock_time_s"]:
            deploy["chaff"] = True

        return deploy

    def _evaluate_ecm_deployment(
        self, radar_threats: list[Any], current_time: float
    ) -> dict[str, bool]:
        """Evaluate whether to activate ECM against radar threats."""
        deploy = {"ecm": False}

        if current_time - self.last_ecm_time < self.ecm_interval:
            return deploy

        # Activate ECM for multiple radar threats or persistent locks
        if len(radar_threats) >= self.config.threat_response_thresholds["multiple_threats"]:
            deploy["ecm"] = True

        # Or if we have a high-priority radar threat
        for threat in radar_threats:
            threat_level = getattr(threat, "threat_level", "low")
            if threat_level == "high":
                deploy["ecm"] = True
                break

        return deploy

    def _evaluate_decoy_deployment(
        self, threats: list[Any], current_time: float
    ) -> dict[str, bool]:
        """Evaluate whether to deploy decoys for complex threat scenarios."""
        deploy = {"decoys": False}

        if current_time - self.last_decoy_time < self.decoy_interval:
            return deploy

        # Deploy decoys in high-threat situations
        high_threat_count = sum(
            1 for threat in threats if getattr(threat, "threat_level", "low") == "high"
        )

        if high_threat_count >= 2:  # Multiple high-priority threats
            deploy["decoys"] = True

        # Or if we're in a defensive automation mode and under serious threat
        if (
            self.config.automation_level == "defensive"
            and len(threats) >= self.config.threat_response_thresholds["multiple_threats"]
        ):
            deploy["decoys"] = True

        return deploy

    def get_countermeasure_status(self) -> dict[str, Any]:
        """Get current status of countermeasure systems."""
        status = {
            "flares_available": getattr(self.aircraft, "flares", 0),
            "chaff_available": getattr(self.aircraft, "chaff", 0),
            "ecm_available": getattr(self.aircraft, "ecm", 0) > 0,
            "decoys_available": getattr(self.aircraft, "decoys", 0),
            "last_deployments": {
                "flares": self.last_flare_time,
                "chaff": self.last_chaff_time,
                "ecm": self.last_ecm_time,
                "decoys": self.last_decoy_time,
            },
        }
        return status

    def update_config(self, config):
        """Update configuration."""
        self.config = config
