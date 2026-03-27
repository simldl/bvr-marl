"""
Enhanced tactical controller that leverages the new asymmetric P_s limits and energy+lift-vector action space.
Provides advanced energy management and physics-aware control for superior combat performance.
"""

import math
from typing import Any, Optional

import numpy as np

from air_to_air_rl.aircrafts.core.nez import NoEscapeZoneCalculator
from air_to_air_rl.aircrafts.core.target_prio import TrackPrioritySystem
from air_to_air_rl.automation.core.auto_helper import AutoHelper, AutoHelperConfig
from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus
from air_to_air_rl.automation.scripted_control.behavior_tree.tactical_tree import (
    TacticalBehaviorTree,
)
from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics
from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager
from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import GeometryCalculator
from air_to_air_rl.automation.scripted_control.tactics.range_calculator import RangeCalculator
from air_to_air_rl.rl.environment.spaces.action_space import EnergyLiftVectorActionProcessor
from air_to_air_rl.simulator.utils.angles import normalize_angle
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class EnhancedEnergyManager(EnergyManager):
    """Enhanced energy manager that uses the new asymmetric P_s limits."""

    def __init__(self, aircraft, action_processor: EnergyLiftVectorActionProcessor | None = None):
        super().__init__(aircraft)
        self.action_processor = action_processor

        # Enhanced energy management parameters
        self.aggressive_ps_factor = 0.8  # How aggressively to use P_s range
        self.defensive_ps_factor = 0.6  # Conservative P_s usage when defensive

        # Load factor management
        self.max_defensive_g = 7.0  # Max G for defensive maneuvers
        self.cruise_g = 1.2  # Slight positive G for efficiency
        self.energy_climb_g = 0.8  # Reduced G during energy climbs

    def get_ps_limits(self, simulator) -> tuple[float, float, float]:
        """Get current asymmetric P_s limits from the action processor."""
        if not self.action_processor or not hasattr(self.action_processor, "agent_states"):
            # Fallback to estimated limits
            return self._estimate_ps_limits()

        # Get agent state if available
        agent_id = getattr(self.aircraft, "id", 1)
        if agent_id in self.action_processor.agent_states:
            scalars = self.action_processor.get_envelope_scalars(agent_id)
            ps_min = scalars.get("Ps_min", -50.0)
            ps_max = scalars.get("Ps_max", 100.0)
            ps_range = ps_max - ps_min
            return ps_min, ps_max, ps_range

        return self._estimate_ps_limits()

    def _estimate_ps_limits(self) -> tuple[float, float, float]:
        """Estimate P_s limits when action processor not available."""
        # Rough estimates based on typical fighter aircraft
        current_speed = max(self.aircraft.speed, 50.0)

        # Simple drag estimate
        drag_estimate = 0.5 * 1.225 * (current_speed**2) * 25.0 * 0.02  # Basic drag

        # Thrust estimates (very rough)
        weight = 15000 * 9.81  # Rough aircraft weight
        max_thrust = weight * 1.2  # T/W ~ 1.2
        min_thrust = 0.0  # Idle

        ps_max = 0.9 * (max_thrust - drag_estimate) * current_speed / weight
        ps_min = 0.9 * (min_thrust - drag_estimate) * current_speed / weight
        ps_range = ps_max - ps_min

        return ps_min, ps_max, ps_range

    def calculate_optimal_energy_command(
        self, context: dict[str, Any]
    ) -> tuple[float, float, float]:
        """
        Calculate optimal P_s, load factor, and bank angle commands for current situation.

        Returns:
            tuple of (ps_command, load_factor, bank_angle_deg)
        """
        ps_min, ps_max, ps_range = self.get_ps_limits(context.get("simulator"))

        # Assess tactical situation
        tactical_state = context.get("tactical_state", "search")
        target = context.get("target")
        threats = context.get("threats", [])
        energy_advantage = context.get("tactical_advantage", 0.0)

        # Check boundary avoidance first (highest priority)
        boundary_heading = context.get("boundary_avoidance_heading")
        if boundary_heading is not None:
            # Emergency boundary avoidance overrides tactical maneuvering
            current_heading = self.aircraft.yaw_deg
            heading_diff = normalize_angle(boundary_heading - current_heading)
            return (
                0.0,
                2.0,
                np.clip(heading_diff * 0.8, -60.0, 60.0),
            )  # Moderate turn to avoid boundary

        # Default commands
        ps_command = 0.0  # Maintain energy
        load_factor = self.cruise_g
        bank_angle = 0.0

        if tactical_state == "defensive":
            # Defensive: Use negative P_s for deceleration, high G for maneuvers
            ps_command = ps_min * self.defensive_ps_factor  # Controlled deceleration
            load_factor = min(self.max_defensive_g, 6.0)

            # Bank toward escape route or perpendicular to threats
            if threats:
                primary_threat = min(threats, key=lambda t: t.distance_m)
                self._calculate_bearing_to_target(primary_threat)
                # Bank perpendicular to threat (notch maneuver)
                bank_angle = 90.0 if np.random.random() > 0.5 else -90.0

        elif tactical_state == "offensive":
            # Offensive: Aggressive energy management
            if energy_advantage > 0.2:
                # We have energy advantage - maintain or trade for position
                ps_command = ps_max * 0.3  # Moderate acceleration
                load_factor = 2.5  # Moderate G for positioning
            else:
                # Need energy - climb or accelerate
                ps_command = ps_max * self.aggressive_ps_factor
                load_factor = self.energy_climb_g

        elif tactical_state == "approach":
            # Approach: Energy setup for BVR engagement
            if target:
                target_range = context.get("target_range_m", float("inf"))
                context.get("closure_rate_mps", 0)
                current_alt = self.aircraft.position.alt
                current_speed = self.aircraft.speed

                # Optimal BVR setup: 12km altitude, ~400 m/s
                optimal_alt = 12000.0
                optimal_speed = 400.0
                nez_out = context.get("nez_out_range_m", 60000)

                # Energy management based on range to NEZ entry
                if target_range > nez_out * 1.5:  # Far from NEZ - build optimal setup
                    alt_deficit = optimal_alt - current_alt
                    speed_deficit = optimal_speed - current_speed

                    if alt_deficit > 1000:  # Need altitude
                        ps_command = ps_max * 0.7  # Climb with energy
                        load_factor = 0.8  # Unload for climb
                    elif speed_deficit > 50:  # Need speed
                        ps_command = ps_max * 0.8  # Accelerate
                        load_factor = 1.2  # Slight positive G
                    else:  # Maintain optimal state
                        ps_command = ps_max * 0.3
                        load_factor = 1.5

                elif target_range > nez_out:  # Approaching NEZ - maintain energy
                    ps_command = ps_max * 0.5  # Sustain energy
                    load_factor = 1.8  # Prepare for maneuvering

                elif target_range < 5000:  # Close range - prepare for merge
                    ps_command = ps_max * 0.4  # Maintain energy
                    load_factor = 3.0  # Ready for maneuvering
                else:  # In NEZ - aggressive maneuvering
                    ps_command = ps_max * 0.6
                    load_factor = 2.5

                # Bank toward target intercept
                if target:
                    target_bearing = self._calculate_bearing_to_target(target)
                    current_bearing = self.aircraft.yaw_deg
                    bearing_diff = normalize_angle(target_bearing - current_bearing)
                    bank_angle = np.clip(bearing_diff * 0.5, -45.0, 45.0)

        elif tactical_state == "escape":
            # Escape: Maximum energy gain
            ps_command = ps_max * 0.9  # Near maximum acceleration
            load_factor = 1.0  # Level flight for efficiency
            bank_angle = 0.0  # Straight line escape initially

        elif tactical_state == "search":
            # Search: Efficient cruise
            current_energy = self.calculate_specific_energy()
            optimal_energy = (self.preferred_speed_mps**2) / (2 * 9.81) + self.preferred_altitude_m

            energy_deficit = optimal_energy - current_energy
            if abs(energy_deficit) > 1000:  # Significant energy difference
                ps_fraction = np.clip(energy_deficit / 10000, -0.5, 0.5)
                ps_command = ps_fraction * ps_range
            else:
                ps_command = 0.0  # Maintain current energy

            load_factor = self.cruise_g

        return ps_command, load_factor, bank_angle

    def _calculate_bearing_to_target(self, target) -> float:
        """Calculate bearing to target."""
        return geodetic_bearing_deg(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            target.position.lat,
            target.position.lon,
        )


class EnhancedTacticalController:
    """
    Enhanced tactical controller using asymmetric P_s limits and energy+lift-vector control.

    Provides advanced physics-aware flight control with superior energy management.
    """

    def __init__(
        self,
        aircraft,
        simulator,
        config: AutoHelperConfig | None = None,
        use_energy_space: bool = True,
    ):
        self.aircraft = aircraft
        self.simulator = simulator
        self.use_energy_space = use_energy_space

        # Initialize enhanced action processor
        if use_energy_space:
            from air_to_air_rl.rl.environment.spaces.action_space import (
                EnergyLiftVectorActionProcessor,
            )

            self.action_processor = EnergyLiftVectorActionProcessor(simulator)
            # Initialize agent state
            self.action_processor._init_agent_state(aircraft.id, aircraft)
        else:
            from air_to_air_rl.rl.environment.spaces.action_space import ActionProcessor

            self.action_processor = ActionProcessor(simulator, use_speed_control=True)
            self.action_processor._init_agent_state(aircraft.id, aircraft)

        # Initialize semi-automated helper
        self.auto_helper = AutoHelper(aircraft, config)

        # Initialize enhanced tactical modules
        self.bvr_tactics = BVRTactics(aircraft)
        self.energy_manager = EnhancedEnergyManager(
            aircraft, self.action_processor if use_energy_space else None
        )
        self.geometry_calc = GeometryCalculator(aircraft)
        self.range_calc = RangeCalculator(aircraft)

        # Initialize proper fighter systems
        self.target_priority = TrackPrioritySystem(aircraft)
        self.nez_calculator = NoEscapeZoneCalculator(aircraft)

        # Initialize behavior tree
        self.behavior_tree = TacticalBehaviorTree(aircraft, self)

        # State tracking
        self.current_tactical_state = "search"
        self.target = None
        self.last_decision_time = 0.0
        self.control_mode = "energy_space" if use_energy_space else "speed_control"

        # Enhanced control state
        self.desired_ps = 0.0
        self.desired_load_factor = 1.0
        self.desired_bank_angle = 0.0

    def get_action(self, simulator, dt: float) -> np.ndarray:
        """
        Get enhanced tactical action leveraging asymmetric P_s limits.

        Returns:
            10-element action array for the enhanced action processor
        """
        # Build comprehensive tactical context
        context = self._build_enhanced_tactical_context(simulator, dt)

        # Execute behavior tree for high-level decisions
        tree_status = self.behavior_tree.tick(context)

        # Debug: Check fire_missile immediately after behavior tree
        fire_missile_after_bt = context.get("fire_missile", False)
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        if self.target and context.get("target_range_m", 0) < 100000:
            print(
                f"Aircraft {aircraft_id}: POST_BT - fire_missile: {fire_missile_after_bt}, tree_status: {tree_status}"
            )

        # Get semi-automated actions (countermeasures, target selection)
        auto_actions = self.auto_helper.update(simulator, dt)

        # Get envelope scalars for physics-aware control
        envelope_scalars = self.action_processor.get_envelope_scalars(self.aircraft.id)

        if self.use_energy_space:
            return self._get_energy_space_action(context, auto_actions, envelope_scalars)
        else:
            return self._get_legacy_action(context, auto_actions)

    def _get_energy_space_action(
        self, context: dict[str, Any], auto_actions: dict, envelope_scalars: dict
    ) -> np.ndarray:
        """Generate action using energy+lift-vector space."""
        # Calculate optimal energy commands
        ps_command, load_factor, bank_angle = self.energy_manager.calculate_optimal_energy_command(
            context
        )

        # Get P_s limits for normalization
        ps_min = envelope_scalars.get("Ps_min", -50.0)
        ps_max = envelope_scalars.get("Ps_max", 100.0)

        # Normalize P_s command to [0,1] action space
        if ps_max > ps_min:
            a_ps = (ps_command - ps_min) / (ps_max - ps_min)
        else:
            a_ps = 0.5
        a_ps = np.clip(a_ps, 0.0, 1.0)

        # Normalize load factor to [0,1] action space
        n_max = envelope_scalars.get("n_max", 9.0)
        n_min = -2.0  # From action processor
        a_n = (load_factor - n_min) / (n_max - n_min)
        a_n = np.clip(a_n, 0.0, 1.0)

        # Normalize bank angle to [0,1] action space
        phi_max = envelope_scalars.get("phi_max_deg", 85.0)
        a_phi = (bank_angle + phi_max) / (2.0 * phi_max)
        a_phi = np.clip(a_phi, 0.0, 1.0)

        # Debug: Check fire_missile flag
        fire_missile_flag = context.get("fire_missile", False)
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        if (
            self.target and context.get("target_range_m", 0) < 100000
        ):  # Only log when in reasonable range
            print(f"Aircraft {aircraft_id}: ACTION_GEN - fire_missile context: {fire_missile_flag}")

        # Build action array for energy+lift-vector space
        action = np.zeros(10)
        action[0] = a_ps  # P_s command
        action[1] = a_n  # load factor command
        action[2] = a_phi  # bank angle command
        action[3] = auto_actions["target_selection"]  # target selection (semi-auto)
        action[4] = 1.0 if fire_missile_flag else 0.0  # missile fire
        action[5] = 1.0 if context.get("fire_gun", False) else 0.0  # gun fire
        action[6] = 1.0 if auto_actions["countermeasures"]["flares"] else 0.0  # flares
        action[7] = 1.0 if auto_actions["countermeasures"]["chaff"] else 0.0  # chaff
        action[8] = 1.0 if auto_actions["countermeasures"]["ecm"] else 0.0  # ecm
        action[9] = 1.0 if auto_actions["countermeasures"]["decoys"] else 0.0  # decoys

        # Store for debugging
        self.desired_ps = ps_command
        self.desired_load_factor = load_factor
        self.desired_bank_angle = bank_angle

        return action

    def _get_legacy_action(self, context: dict[str, Any], auto_actions: dict) -> np.ndarray:
        """Generate action using legacy throttle/pitch/yaw space."""
        # Extract desired controls from context (set by behavior tree)
        throttle = context.get("desired_throttle", 0.8)
        yaw = context.get("desired_yaw", self.aircraft.yaw_deg)
        pitch = context.get("desired_pitch", 0.0)

        # Build legacy action array
        action = np.zeros(10)
        action[0] = np.clip(throttle, 0.0, 1.0)  # throttle/speed
        action[1] = self._yaw_to_action_space(yaw)  # yaw change
        action[2] = self._pitch_to_action_space(pitch)  # pitch change
        action[3] = auto_actions["target_selection"]  # target selection
        action[4] = 1.0 if context.get("fire_missile", False) else 0.0  # missile fire
        action[5] = 1.0 if context.get("fire_gun", False) else 0.0  # gun fire
        action[6] = 1.0 if auto_actions["countermeasures"]["flares"] else 0.0  # flares
        action[7] = 1.0 if auto_actions["countermeasures"]["chaff"] else 0.0  # chaff
        action[8] = 1.0 if auto_actions["countermeasures"]["ecm"] else 0.0  # ecm
        action[9] = 1.0 if auto_actions["countermeasures"]["decoys"] else 0.0  # decoys

        return action

    def _build_enhanced_tactical_context(self, simulator, dt: float) -> dict[str, Any]:
        """Build enhanced tactical context with energy and envelope information."""
        # Get base context
        context = self._build_base_tactical_context(simulator, dt)

        # Add enhanced energy information
        envelope_scalars = self.action_processor.get_envelope_scalars(self.aircraft.id)
        context.update(
            {
                "envelope_scalars": envelope_scalars,
                "simulator": simulator,  # Needed for P_s limit calculation
                "s_factor": envelope_scalars.get("s_factor", 1.0),
                "c_factor": envelope_scalars.get("c_factor", 1.0),
            }
        )

        # Add P_s capability information
        if "Ps_min" in envelope_scalars and "Ps_max" in envelope_scalars:
            ps_min = envelope_scalars["Ps_min"]
            ps_max = envelope_scalars["Ps_max"]
            context.update(
                {
                    "ps_capability": {
                        "min_ms": ps_min,
                        "max_ms": ps_max,
                        "range_ms": ps_max - ps_min,
                        "can_decelerate": ps_min < 0,
                        "max_accel_ms2": ps_max / max(self.aircraft.speed, 50.0),
                    }
                }
            )

        return context

    def _check_boundary_avoidance(self, context: dict[str, Any]) -> float | None:
        """
        Check if aircraft is near map boundary and return avoidance heading.

        Returns:
            Heading to avoid boundary, or None if no avoidance needed
        """
        # Get actual map boundaries from aircraft's map_limits
        map_limits = getattr(self.aircraft, "map_limits", None)
        if not map_limits:
            return None

        current_lat = self.aircraft.position.lat
        current_lon = self.aircraft.position.lon

        # Calculate buffer zone (10% of map size minimum, but at least 0.05 degrees)
        map_width_deg = map_limits.right_lon - map_limits.left_lon
        map_height_deg = map_limits.top_lat - map_limits.bottom_lat
        buffer_lon = max(0.05, map_width_deg * 0.15)  # 15% buffer
        buffer_lat = max(0.05, map_height_deg * 0.15)  # 15% buffer

        # Calculate boundary thresholds
        left_threshold = map_limits.left_lon + buffer_lon
        right_threshold = map_limits.right_lon - buffer_lon
        bottom_threshold = map_limits.bottom_lat + buffer_lat
        top_threshold = map_limits.top_lat - buffer_lat

        # Check boundary proximity and calculate avoidance heading
        avoid_heading = None

        if current_lon < left_threshold:  # Too far west
            avoid_heading = 90.0  # Turn east
        elif current_lon > right_threshold:  # Too far east
            avoid_heading = 270.0  # Turn west

        if current_lat < bottom_threshold:  # Too far south
            if avoid_heading is not None:
                # Combine with existing avoidance
                avoid_heading = (avoid_heading + 0.0) / 2  # Bias toward north
            else:
                avoid_heading = 0.0  # Turn north
        elif current_lat > top_threshold:  # Too far north
            if avoid_heading is not None:
                # Combine with existing avoidance
                avoid_heading = (avoid_heading + 180.0) / 2  # Bias toward south
            else:
                avoid_heading = 180.0  # Turn south

        if avoid_heading is not None:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            print(
                f"Aircraft {aircraft_id}: BOUNDARY_AVOIDANCE - Current: ({current_lat:.3f}, {current_lon:.3f}), Turning to: {avoid_heading:.1f}°"
            )

        return avoid_heading

    def get_prioritized_targets(self, simulator) -> list:
        """
        Get targets prioritized using the proper TrackPrioritySystem.

        Returns list of targets ordered by priority (highest first).
        """
        # Get all potential targets from the threat assessment
        threats = self.auto_helper.threat_assessment.assess_threats(simulator)

        if not threats:
            return []

        # Convert threats to track states format expected by TrackPrioritySystem
        track_states = []
        for threat in threats:
            if hasattr(threat, "source") and hasattr(threat.source, "position"):
                source = threat.source
                # Create state vector [x, y, z, vx, vy, vz] relative to own aircraft
                np.array([0, 0, 0])  # Own aircraft at origin

                # Calculate relative position (simplified)
                range_km = geodetic_distance_km(
                    self.aircraft.position.lat,
                    self.aircraft.position.lon,
                    self.aircraft.position.alt,
                    source.position.lat,
                    source.position.lon,
                    source.position.alt,
                )
                bearing_deg = geodetic_bearing_deg(
                    self.aircraft.position.lat,
                    self.aircraft.position.lon,
                    source.position.lat,
                    source.position.lon,
                )

                # Convert to local coordinates (simplified)
                rel_x = range_km * 1000 * np.sin(np.radians(bearing_deg))
                rel_y = range_km * 1000 * np.cos(np.radians(bearing_deg))
                rel_z = -(source.position.alt - self.aircraft.position.alt)

                # Velocity components (simplified)
                target_speed = getattr(source, "speed", 250.0)
                target_heading = getattr(source, "yaw_deg", 0.0)
                vx = target_speed * np.sin(np.radians(target_heading))
                vy = target_speed * np.cos(np.radians(target_heading))
                vz = 0.0  # Simplified

                state_vec = np.array([rel_x, rel_y, rel_z, vx, vy, vz])
                cov = np.eye(6)  # Identity covariance matrix
                track_states.append((state_vec, cov))

        if not track_states:
            return []

        try:
            # Use TrackPrioritySystem to prioritize targets
            prioritized_tracks = self.target_priority.prioritize(track_states)

            # Convert back to threat objects, ordered by priority (lowest score = highest priority)
            prioritized_targets = []
            for i, (state_vec, score) in enumerate(prioritized_tracks):
                if i < len(threats):
                    prioritized_targets.append(threats[i].source)

            return prioritized_targets

        except Exception:
            # Fallback to original threat ordering if priority system fails
            return [threat.source for threat in threats if hasattr(threat, "source")]

    def _build_base_tactical_context(self, simulator, dt: float) -> dict[str, Any]:
        """Build base tactical context (similar to original controller)."""
        # Get threat assessment
        threats = self.auto_helper.threat_assessment.assess_threats(simulator)

        # Get current target
        target = self.auto_helper.target_manager.current_target
        self.target = target

        # Calculate geometry and ranges
        geometry_info = self.geometry_calc.get_geometry_summary(target) if target else {}
        range_info = self.range_calc.get_range_summary(target, threats) if target else {}
        energy_info = self.energy_manager.get_energy_state_summary()

        # Build base context
        context = {
            "dt": dt,
            "sim_time": getattr(simulator, "utc_time", 0.0),
            "aircraft": self.aircraft,
            "target": target,
            "threats": threats,
            "geometry": geometry_info,
            "ranges": range_info,
            "energy": energy_info,
            "tactical_state": self.current_tactical_state,
        }

        # Check for boundary avoidance (highest priority)
        boundary_heading = self._check_boundary_avoidance(context)

        # Add derived tactical information
        if target:
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

            # Get comprehensive range information for BVR
            nez_ranges = self.range_calc.calculate_nez_ranges(target)
            f_pole_m = self.range_calc.calculate_f_pole_range(target)
            dor_m = self.range_calc.calculate_desired_out_range(threats)

            # Calculate closure rate and gimbal status
            closure_rate = geometry_info.get("engagement_envelope", {}).get("closure_rate_mps", 0)
            gimbal_ok = geometry_info.get("engagement_envelope", {}).get("in_radar_gimbal", True)

            # Enhanced launch range logic - More aggressive
            # Allow firing at 90% of aerodynamic limit even if outside NEZ
            max_fire_range = max(
                nez_ranges.get("nez_out_m", 0), nez_ranges.get("r_aero_m", 100000) * 0.9
            )
            in_launch_range = (
                target_range <= max_fire_range
                and (
                    nez_ranges.get("nez_valid", False) or target_range <= 80000
                )  # Allow up to 80km always
                and gimbal_ok
                and closure_rate > 10
            )  # Reduced closure requirement to 10 m/s

            # Calculate target bearing for lock acquisition
            target_bearing = geodetic_bearing_deg(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                target.position.lat,
                target.position.lon,
            )

            context.update(
                {
                    "target_range_m": target_range,
                    "target_bearing_deg": target_bearing,
                    "nez_in_range_m": nez_ranges.get("nez_in_m", 0),
                    "nez_out_range_m": nez_ranges.get("nez_out_m", 0),
                    "r_aero_m": nez_ranges.get("r_aero_m", 100000),  # DLZ aerodynamic limit
                    "r_tr_m": nez_ranges.get("r_tr_m", 50000),  # DLZ turning target range
                    "r_pi_m": nez_ranges.get("r_pi_m", 70000),  # DLZ non-maneuvering range
                    "f_pole_m": f_pole_m,
                    "desired_out_range_m": dor_m,
                    "in_launch_range": in_launch_range,
                    "closure_rate_mps": closure_rate,
                    "gimbal_ok": gimbal_ok,
                    "target_aspect_deg": geometry_info.get("engagement_envelope", {}).get(
                        "aspect_angle_deg", 0
                    ),
                    "angle_off_tail_deg": geometry_info.get("engagement_envelope", {}).get(
                        "angle_off_tail_deg", 180
                    ),
                    "boundary_avoidance_heading": boundary_heading,
                    "missile_max_range_m": self.range_calc.missile_params["max_range_m"],
                }
            )

        # Assess tactical advantage
        context["tactical_advantage"] = self._assess_enhanced_tactical_advantage(context)

        return context

    def _assess_enhanced_tactical_advantage(self, context: dict[str, Any]) -> float:
        """Enhanced tactical advantage assessment using energy information."""
        if not context.get("target"):
            return 0.0

        advantage = 0.0

        # Energy advantage (enhanced)
        energy_state = context["energy"]["energy_state"]
        if energy_state == "high":
            advantage += 0.4  # Increased weight for energy
        elif energy_state == "optimal":
            advantage += 0.2
        elif energy_state == "low":
            advantage -= 0.3
        elif energy_state == "critical":
            advantage -= 0.7

        # P_s capability advantage
        ps_info = context.get("ps_capability", {})
        if ps_info.get("can_decelerate", False):
            advantage += 0.2  # Having deceleration capability is valuable
        if ps_info.get("max_accel_ms2", 0) > 5.0:
            advantage += 0.1  # Good acceleration capability

        # Envelope position advantage
        s_factor = context.get("s_factor", 1.0)
        c_factor = context.get("c_factor", 1.0)
        envelope_health = (s_factor + c_factor) / 2.0
        advantage += (envelope_health - 0.5) * 0.3  # Penalty for being near envelope limits

        # Range and threat assessment (similar to original)
        range_situation = context.get("ranges", {}).get("tactical_situation", "neutral")
        if range_situation == "offensive_optimal":
            advantage += 0.3
        elif range_situation == "offensive_viable":
            advantage += 0.1
        elif range_situation == "defensive_immediate":
            advantage -= 0.5

        return np.clip(advantage, -1.0, 1.0)

    def _is_in_launch_range(self, target, range_info: dict) -> bool:
        """Check if target is in missile launch range."""
        envelope = range_info.get("launch_envelope", {})
        return envelope.get("recommended_launch", False)

    def _yaw_to_action_space(self, desired_yaw: float) -> float:
        """Convert desired yaw to action space format (legacy compatibility)."""
        current_yaw = self.aircraft.yaw_deg
        yaw_diff = normalize_angle(desired_yaw - current_yaw)
        normalized_diff = (yaw_diff + 180) / 360
        return np.clip(normalized_diff, 0.0, 1.0)

    def _pitch_to_action_space(self, desired_pitch: float) -> float:
        """Convert desired pitch to action space format (legacy compatibility)."""
        current_pitch = getattr(self.aircraft, "pitch_deg", 0.0)
        pitch_diff = desired_pitch - current_pitch
        normalized_diff = (pitch_diff + 90) / 180
        return np.clip(normalized_diff, 0.0, 1.0)

    def get_enhanced_status(self) -> dict[str, Any]:
        """Get comprehensive enhanced controller status."""
        base_status = {
            "tactical_state": self.current_tactical_state,
            "current_target": getattr(self.target, "id", None) if self.target else None,
            "control_mode": self.control_mode,
            "energy_state": self.energy_manager.get_energy_state_summary(),
        }

        if self.use_energy_space:
            envelope_scalars = self.action_processor.get_envelope_scalars(self.aircraft.id)
            base_status.update(
                {
                    "energy_commands": {
                        "desired_ps_ms": self.desired_ps,
                        "desired_load_factor": self.desired_load_factor,
                        "desired_bank_angle_deg": self.desired_bank_angle,
                    },
                    "envelope_status": {
                        "s_factor": envelope_scalars.get("s_factor", 1.0),
                        "c_factor": envelope_scalars.get("c_factor", 1.0),
                        "ps_min": envelope_scalars.get("Ps_min", 0.0),
                        "ps_max": envelope_scalars.get("Ps_max", 0.0),
                    },
                    "action_processor_debug": self.action_processor.get_debug_info(
                        self.aircraft.id
                    ),
                }
            )

        return base_status

    # Enhanced behavior tree action implementations
    def execute_enhanced_defensive_maneuvers(self, context: dict[str, Any]) -> NodeStatus:
        """Execute enhanced defensive maneuvers using asymmetric P_s limits."""
        threats = context.get("threats", [])
        ps_capability = context.get("ps_capability", {})

        # Find most critical threat
        missile_threats = [
            t for t in threats if t.threat_type == "missile" and t.threat_level == "critical"
        ]

        if missile_threats:
            closest_threat = min(missile_threats, key=lambda t: t.distance_m)

            # Use energy space for enhanced defensive maneuvering
            if self.use_energy_space and ps_capability.get("can_decelerate", False):
                # Advanced defensive: combine deceleration with high-g maneuver
                context["desired_ps"] = ps_capability["min_ms"] * 0.8  # Strong deceleration
                context["desired_load_factor"] = min(
                    7.0, context.get("envelope_scalars", {}).get("n_max", 9.0)
                )
                context["desired_bank_angle"] = 90.0  # Hard turn
                return NodeStatus.SUCCESS
            else:
                # Fallback to traditional defensive maneuvers
                return self.bvr_tactics.execute_notch_maneuver(
                    {"primary_missile_threat": closest_threat}
                )

        return NodeStatus.FAILURE

    def execute_enhanced_approach(self, context: dict[str, Any]) -> NodeStatus:
        """Execute enhanced approach using energy optimization."""
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        ps_capability = context.get("ps_capability", {})
        target_range = context.get("target_range_m", float("inf"))

        if self.use_energy_space:
            # Energy-optimized approach
            if target_range > 20000:  # Long range - build energy
                ps_command = ps_capability.get("max_ms", 100.0) * 0.7
                load_factor = 1.5
            elif target_range < 10000:  # Close range - prepare for merge
                ps_command = ps_capability.get("max_ms", 100.0) * 0.4
                load_factor = 3.0
            else:  # Medium range
                ps_command = ps_capability.get("max_ms", 100.0) * 0.5
                load_factor = 2.0

            # Calculate intercept angle
            target_bearing = self.energy_manager._calculate_bearing_to_target(target)
            current_bearing = self.aircraft.yaw_deg
            bearing_diff = normalize_angle(target_bearing - current_bearing)
            bank_angle = np.clip(bearing_diff * 0.5, -45.0, 45.0)

            context["desired_ps"] = ps_command
            context["desired_load_factor"] = load_factor
            context["desired_bank_angle"] = bank_angle

            return NodeStatus.SUCCESS
        else:
            # Legacy approach
            return self.bvr_tactics.execute_approach(context)

    def _execute_coordinated_grid_search(self, context: dict[str, Any]) -> float:
        """
        Execute coordinated grid search pattern based on team composition.
        Used when no targets are within radar range to search for new contacts.

        Returns:
            Heading for this aircraft's grid search pattern
        """
        aircraft_id = getattr(self.aircraft, "id", 1)
        aircraft_group = getattr(self.aircraft, "group", "unknown")

        # Get team information from context or environment
        # For now, assume we have 2 aircraft per team (4 total in 2v2)
        team_size = 2

        # Get current simulation time for pattern cycling
        sim_time = context.get("sim_time", 0.0)
        if not sim_time:
            # Fallback to step-based timing
            sim_step = context.get("sim_step", 0)
            sim_time = sim_step * 0.1  # Assume 0.1s timestep

        # Calculate grid search patterns based on aircraft ID within team
        if aircraft_group == "agent":
            # Agent team (aircraft IDs 1, 2)
            aircraft_team_id = aircraft_id if aircraft_id <= 2 else aircraft_id - 2
        else:
            # Opponent team (aircraft IDs 3, 4 typically map to 5, 6, 7, 8)
            aircraft_team_id = (aircraft_id - 4) if aircraft_id > 4 else aircraft_id

        # Ensure we have a valid team ID
        aircraft_team_id = max(1, min(aircraft_team_id, team_size))

        # Create search pattern - sweep back and forth in assigned sectors
        pattern_duration = 120.0  # 2 minutes per pattern cycle
        cycle_progress = (sim_time % pattern_duration) / pattern_duration

        if team_size == 2:
            if aircraft_team_id == 1:
                # Aircraft 1: Search north-south on left side
                if cycle_progress < 0.5:
                    heading = 0.0  # North
                else:
                    heading = 180.0  # South
            else:
                # Aircraft 2: Search north-south on right side
                if cycle_progress < 0.5:
                    heading = 180.0  # South (opposite phase)
                else:
                    heading = 0.0  # North
        else:
            # For larger teams, create more complex grid patterns
            base_headings = [0.0, 90.0, 180.0, 270.0]  # N, E, S, W
            heading = base_headings[aircraft_team_id % len(base_headings)]

        return heading
