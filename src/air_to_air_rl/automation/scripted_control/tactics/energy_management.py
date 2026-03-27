"""
Energy management system for air-to-air combat.
Handles altitude, speed, and energy state optimization for tactical advantage.
"""

from typing import Any

import numpy as np

from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus


class EnergyManager:
    """Manages aircraft energy state for optimal combat performance."""

    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.preferred_altitude_m = 12000  # 12km optimal BVR altitude
        self.preferred_speed_mps = 400  # Optimal BVR speed
        self.min_combat_speed_mps = 200  # Minimum speed for maneuvering
        self.max_climb_rate = 50  # Maximum climb rate m/s

        # Energy state tracking
        self.energy_history = []
        self.last_altitude = aircraft.position.alt
        self.last_speed = aircraft.speed

    def calculate_specific_energy(self) -> float:
        """
        Calculate specific energy (energy per unit mass).

        Specific Energy = (V²/2g) + h
        Where V = velocity, g = gravity, h = altitude
        """
        velocity_energy = (self.aircraft.speed**2) / (2 * 9.81)  # Kinetic energy term
        altitude_energy = self.aircraft.position.alt  # Potential energy term

        return velocity_energy + altitude_energy

    def calculate_energy_rate(self) -> float:
        """Calculate rate of energy change."""
        current_energy = self.calculate_specific_energy()

        if len(self.energy_history) < 2:
            self.energy_history.append(current_energy)
            return 0.0

        previous_energy = self.energy_history[-1]
        energy_rate = current_energy - previous_energy

        # Update history (keep last 10 values)
        self.energy_history.append(current_energy)
        if len(self.energy_history) > 10:
            self.energy_history.pop(0)

        return energy_rate

    def assess_energy_advantage(self, target) -> float:
        """
        Assess energy advantage relative to target.

        Returns:
            Positive value = we have energy advantage
            Negative value = target has energy advantage
        """
        our_energy = self.calculate_specific_energy()

        # Estimate target energy (if we have the data)
        if hasattr(target, "position") and hasattr(target, "speed"):
            target_velocity_energy = (target.speed**2) / (2 * 9.81)
            target_altitude_energy = target.position.alt
            target_energy = target_velocity_energy + target_altitude_energy

            return our_energy - target_energy
        else:
            # If no target data, compare to optimal energy state
            optimal_energy = (self.preferred_speed_mps**2) / (2 * 9.81) + self.preferred_altitude_m
            return our_energy - optimal_energy

    def calculate_optimal_flight_profile(
        self, context: dict[str, Any]
    ) -> tuple[float, float, float]:
        """
        Calculate optimal altitude, speed, and throttle for current tactical situation.

        Returns:
            tuple of (desired_altitude, desired_speed, throttle_setting)
        """
        target = context.get("target")
        threats = context.get("threats", [])
        tactical_situation = context.get("tactical_situation", "neutral")

        # Base preferences
        desired_altitude = self.preferred_altitude_m
        desired_speed = self.preferred_speed_mps
        throttle = 0.8

        # Adjust based on tactical situation
        if tactical_situation == "defensive":
            # Defensive: prioritize speed and maneuverability
            desired_speed = min(self.preferred_speed_mps * 1.1, 450)
            desired_altitude = max(
                self.aircraft.position.alt - 1000, 8000
            )  # Lower altitude for terrain masking
            throttle = 0.9

        elif tactical_situation == "offensive":
            # Offensive: prioritize altitude advantage
            desired_altitude = min(self.preferred_altitude_m + 2000, 15000)
            desired_speed = self.preferred_speed_mps
            throttle = 0.85

        elif tactical_situation == "approach":
            # Approaching target: balance speed and altitude
            if target:
                target_altitude = getattr(target.position, "alt", self.preferred_altitude_m)
                # Try to maintain altitude advantage
                desired_altitude = max(target_altitude + 1000, self.preferred_altitude_m)
                desired_speed = self.preferred_speed_mps * 0.95  # Slightly conservative
                throttle = 0.8

        elif tactical_situation == "escape":
            # Escaping: maximize speed
            desired_speed = min(self.aircraft.max_speed_mps * 0.9, 500)
            desired_altitude = self.aircraft.position.alt  # Maintain current altitude initially
            throttle = 1.0

        # Adjust for immediate threats
        immediate_threats = [
            t for t in threats if t.distance_m < 10000 and t.threat_level == "critical"
        ]
        if immediate_threats:
            # Emergency energy management
            desired_speed = min(self.preferred_speed_mps * 1.15, 480)
            throttle = 1.0

        return desired_altitude, desired_speed, throttle

    def execute_energy_climb(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute energy climb - trade speed for altitude.

        Used when we have excess speed and want altitude advantage.
        """
        current_speed = self.aircraft.speed

        if current_speed > self.preferred_speed_mps * 1.1:  # Have excess speed
            # Calculate climb angle based on excess speed
            excess_speed = current_speed - self.preferred_speed_mps
            climb_angle = min(excess_speed * 0.1, 20.0)  # Max 20 degree climb

            context["desired_pitch"] = climb_angle
            context["desired_throttle"] = 0.7  # Reduce throttle during climb

            return NodeStatus.SUCCESS
        else:
            return NodeStatus.FAILURE

    def execute_energy_dive(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute energy dive - trade altitude for speed.

        Used when we need speed and have altitude to spare.
        """
        current_speed = self.aircraft.speed
        current_altitude = self.aircraft.position.alt

        if (
            current_speed < self.preferred_speed_mps * 0.9  # Need more speed
            and current_altitude > 8000
        ):  # Have altitude to trade
            # Calculate dive angle based on speed deficit
            speed_deficit = self.preferred_speed_mps - current_speed
            dive_angle = min(speed_deficit * 0.1, -15.0)  # Max 15 degree dive

            context["desired_pitch"] = dive_angle
            context["desired_throttle"] = 0.9  # Higher throttle during dive

            return NodeStatus.SUCCESS
        else:
            return NodeStatus.FAILURE

    def execute_energy_sustain(self, context: dict[str, Any]) -> NodeStatus:
        """
        Execute energy sustaining flight - maintain optimal energy state.

        Used for patrol and approach phases.
        """
        desired_altitude, desired_speed, throttle = self.calculate_optimal_flight_profile(context)

        current_altitude = self.aircraft.position.alt
        current_speed = self.aircraft.speed

        # Calculate altitude adjustment
        altitude_diff = desired_altitude - current_altitude
        if abs(altitude_diff) > 500:  # Significant altitude difference
            climb_rate = np.clip(altitude_diff / 100, -self.max_climb_rate, self.max_climb_rate)
            pitch_angle = np.clip(climb_rate * 0.1, -10.0, 10.0)
        else:
            pitch_angle = 0.0

        # Speed management through throttle
        speed_diff = desired_speed - current_speed
        throttle_adjustment = speed_diff * 0.002  # Fine throttle control
        throttle = np.clip(throttle + throttle_adjustment, 0.0, 1.0)

        context["desired_pitch"] = pitch_angle
        context["desired_throttle"] = throttle

        return NodeStatus.SUCCESS

    def calculate_turn_performance(self, load_factor: float = 1.0) -> dict[str, float]:
        """
        Calculate current turn performance metrics.

        Args:
            load_factor: G-force loading (1.0 = level flight)

        Returns:
            dict with turn radius, turn rate, and stall speed
        """
        velocity = max(self.aircraft.speed, 50)  # Avoid division by zero
        gravity = 9.81

        # Turn radius calculation: R = V² / (g * √(n²-1))
        # Where n = load factor (G-force)
        if load_factor > 1.0:
            turn_radius = (velocity**2) / (gravity * np.sqrt(load_factor**2 - 1))
            turn_rate = velocity / turn_radius  # radians per second
            turn_rate_deg_per_sec = np.degrees(turn_rate)
        else:
            turn_radius = float("inf")
            turn_rate_deg_per_sec = 0.0

        # Stall speed increases with load factor: V_stall = V_stall_1g * √n
        base_stall_speed = getattr(self.aircraft, "stall0_mps", 60)
        stall_speed = base_stall_speed * np.sqrt(load_factor)

        return {
            "turn_radius_m": turn_radius,
            "turn_rate_deg_per_sec": turn_rate_deg_per_sec,
            "stall_speed_mps": stall_speed,
            "max_sustainable_g": min(9.0, velocity / base_stall_speed),  # Rough estimate
        }

    def get_energy_state_summary(self) -> dict[str, Any]:
        """Get summary of current energy state."""
        current_energy = self.calculate_specific_energy()
        energy_rate = self.calculate_energy_rate()

        return {
            "specific_energy": current_energy,
            "energy_rate": energy_rate,
            "altitude_m": self.aircraft.position.alt,
            "speed_mps": self.aircraft.speed,
            "optimal_altitude_m": self.preferred_altitude_m,
            "optimal_speed_mps": self.preferred_speed_mps,
            "energy_state": self._classify_energy_state(current_energy),
        }

    def _classify_energy_state(self, current_energy: float) -> str:
        """Classify current energy state."""
        optimal_energy = (self.preferred_speed_mps**2) / (2 * 9.81) + self.preferred_altitude_m

        energy_ratio = current_energy / optimal_energy

        if energy_ratio > 1.2:
            return "high"
        elif energy_ratio > 0.8:
            return "optimal"
        elif energy_ratio > 0.6:
            return "low"
        else:
            return "critical"
