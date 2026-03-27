"""
Full tactical controller for air-to-air combat.
Integrates behavior trees, tactical modules, and semi-automated systems
to provide complete flight and combat control.
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.automation.core.auto_helper import AutoHelper, AutoHelperConfig
from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import NodeStatus
from air_to_air_rl.automation.scripted_control.behavior_tree.tactical_tree import (
    TacticalBehaviorTree,
)
from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics
from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager
from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import GeometryCalculator
from air_to_air_rl.automation.scripted_control.tactics.range_calculator import RangeCalculator
from air_to_air_rl.simulator.utils.angles import normalize_angle
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class FullTacticalController:
    """
    Complete tactical controller integrating all combat systems.

    Provides full automation of:
    - Flight control (throttle, yaw, pitch)
    - Missile firing decisions
    - Target selection and countermeasures (via AutoHelper)
    - Advanced tactical maneuvers
    """

    def __init__(self, aircraft, simulator=None, config: AutoHelperConfig | None = None):
        self.aircraft = aircraft
        self.simulator = simulator

        # Initialize action processor for envelope queries
        if simulator is not None:
            from air_to_air_rl.rl.environment.spaces.action_space import (
                EnergyLiftVectorActionProcessor,
            )

            self.action_processor = EnergyLiftVectorActionProcessor(simulator)
            self.action_processor._init_agent_state(aircraft.id, aircraft)
        else:
            self.action_processor = None

        # Initialize semi-automated helper for countermeasures and target selection
        self.auto_helper = AutoHelper(aircraft, config)

        # Initialize tactical modules
        self.bvr_tactics = BVRTactics(aircraft)
        self.energy_manager = EnergyManager(aircraft)
        self.geometry_calc = GeometryCalculator(aircraft)
        self.range_calc = RangeCalculator(aircraft)

        # Initialize behavior tree
        self.behavior_tree = TacticalBehaviorTree(aircraft, self)

        # State tracking
        self.current_tactical_state = "search"
        self.target = None
        self.last_decision_time = 0.0
        self.flight_controls = {
            "throttle": 0.8,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "missile_fire": False,
        }

    def get_action(self, simulator, dt: float) -> np.ndarray:
        """
        Get complete tactical action for the aircraft.

        Returns:
            Complete 10-element action array for the energy+lift-vector action processor
        """
        # Build tactical context
        context = self._build_tactical_context(simulator, dt)

        # Execute behavior tree
        tree_status = self.behavior_tree.tick(context)

        # Log current behavior tree state
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        target = context.get("target")
        target_range = (
            context.get("target_range_m", 0) / 1000 if context.get("target_range_m") else 0
        )
        fire_missile = context.get("fire_missile", False)
        missile_status = "FIRING" if fire_missile else "NO FIRE"
        print(
            f"Aircraft {aircraft_id}: {'HAS TARGET' if target else 'NO TARGET'} "
            f"(range: {target_range:.1f}km) - Tree: {tree_status} - Missile: {missile_status}"
        )

        # Get semi-automated actions (countermeasures, target selection)
        auto_actions = self.auto_helper.update(simulator, dt)

        # Extract flight controls from context (set by behavior tree actions)
        throttle = context.get("desired_throttle", self.flight_controls["throttle"])
        yaw = context.get("desired_yaw", self.aircraft.yaw_deg)
        pitch = context.get("desired_pitch", self.flight_controls["pitch_deg"])
        missile_fire = context.get("fire_missile", False)

        # Convert legacy controls to energy-space [Ps, n, phi]
        ps_action, n_action, phi_action = self._legacy_to_energy_space(throttle, yaw, pitch)

        # Convert to action space format (10-element for energy+lift-vector)
        action = np.zeros(10)

        # Energy+lift-vector controls (converted from legacy)
        action[0] = ps_action  # P_s command
        action[1] = n_action  # load factor command
        action[2] = phi_action  # bank angle command
        action[3] = auto_actions["target_selection"]  # target selection
        action[4] = 1.0 if missile_fire else 0.0  # missile fire
        action[5] = 0.0  # gun fire (not used by legacy controller)

        # Log missile firing attempts
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        if missile_fire:
            remaining_missiles = getattr(self.aircraft, "remaining_missiles", "unknown")
            print(
                f"Aircraft {aircraft_id}: MISSILE_FIRE_ACTION = 1.0, Remaining: {remaining_missiles}"
            )

        # Log target selection for debugging
        if missile_fire:
            print(f"Aircraft {aircraft_id}: TARGET_SELECTION_ACTION = {action[3]:.3f}")

        action[6] = 1.0 if auto_actions["countermeasures"]["flares"] else 0.0  # flares
        action[7] = 1.0 if auto_actions["countermeasures"]["chaff"] else 0.0  # chaff
        action[8] = 1.0 if auto_actions["countermeasures"]["ecm"] else 0.0  # ecm
        action[9] = 1.0 if auto_actions["countermeasures"]["decoys"] else 0.0  # decoys

        # Update state
        self._update_flight_controls(throttle, yaw, pitch, missile_fire)

        return action

    def _build_tactical_context(self, simulator, dt: float) -> dict[str, Any]:
        """Build comprehensive tactical context for decision making."""

        # Get threat assessment from auto helper
        threats = self.auto_helper.threat_assessment.assess_threats(simulator)

        # Get current target
        target = self.auto_helper.target_manager.current_target
        self.target = target

        # Calculate geometry and ranges
        geometry_info = self.geometry_calc.get_geometry_summary(target) if target else {}
        range_info = self.range_calc.get_range_summary(target, threats) if target else {}
        energy_info = self.energy_manager.get_energy_state_summary()

        # Build context
        context = {
            "dt": dt,
            "sim_time": getattr(simulator, "elapsed_time", 0.0),
            "aircraft": self.aircraft,
            "target": target,
            "threats": threats,
            "geometry": geometry_info,
            "ranges": range_info,
            "energy": energy_info,
            "tactical_state": self.current_tactical_state,
        }

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

            context.update(
                {
                    "target_range_m": target_range,
                    "in_launch_range": self._is_in_launch_range(target, range_info),
                    "closure_rate_mps": geometry_info.get("engagement_envelope", {}).get(
                        "closure_rate_mps", 0
                    ),
                    "target_aspect_deg": geometry_info.get("engagement_envelope", {}).get(
                        "aspect_angle_deg", 0
                    ),
                    "angle_off_tail_deg": geometry_info.get("engagement_envelope", {}).get(
                        "angle_off_tail_deg", 180
                    ),
                    "nez_out_range_m": range_info.get("launch_envelope", {})
                    .get("nez_ranges", {})
                    .get("nez_out_m", 0),
                }
            )

        # Add missile availability
        context["short_range_missiles"] = self._count_short_range_missiles()
        context["long_range_missiles"] = getattr(self.aircraft, "remaining_missiles", 0)

        # Assess tactical advantage
        context["tactical_advantage"] = self._assess_tactical_advantage(context)

        return context

    def _is_in_launch_range(self, target, range_info: dict) -> bool:
        """Check if target is in missile launch range."""
        envelope = range_info.get("launch_envelope", {})
        target_range = envelope.get("target_range_m", float("inf"))

        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # If no valid target range, compute it directly
        if not np.isfinite(target_range) and target:
            from air_to_air_rl.simulator.utils.geodesics import geodetic_distance_km

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

        # Use realistic long-range missile capabilities (120km radar seeker)
        # Fire if within reasonable BVR range based on actual missile specs
        aggressive_range = (
            3000 <= target_range <= 100000
        )  # Realistic launch window for 120km+ missile
        envelope_launch = envelope.get("recommended_launch", False)
        in_range = aggressive_range or envelope_launch

        print(
            f"Aircraft {aircraft_id}: LAUNCH_RANGE_CALC - Range: {target_range / 1000:.1f}km, "
            f"AggressiveOK: {aggressive_range}, EnvelopeOK: {envelope_launch}, InRange: {in_range}"
        )

        return in_range

    def _count_short_range_missiles(self) -> int:
        """Count available short-range missiles."""
        # This would need to be adapted based on your missile system
        return 2  # Placeholder

    def _assess_tactical_advantage(self, context: dict[str, Any]) -> float:
        """
        Assess overall tactical advantage.

        Returns:
            Positive = we have advantage, Negative = enemy has advantage
        """
        if not context.get("target"):
            return 0.0

        advantage = 0.0

        # Energy advantage
        energy_state = context["energy"]["energy_state"]
        if energy_state == "high":
            advantage += 0.3
        elif energy_state == "optimal":
            advantage += 0.1
        elif energy_state == "low":
            advantage -= 0.2
        elif energy_state == "critical":
            advantage -= 0.5

        # Range advantage
        range_situation = context.get("ranges", {}).get("tactical_situation", "neutral")
        if range_situation == "offensive_optimal":
            advantage += 0.4
        elif range_situation == "offensive_viable":
            advantage += 0.2
        elif range_situation == "defensive_immediate":
            advantage -= 0.6

        # Threat situation
        num_threats = len(context.get("threats", []))
        high_threats = sum(1 for t in context.get("threats", []) if t.threat_level == "high")

        if high_threats == 0 and num_threats <= 1:
            advantage += 0.2  # Clear tactical situation
        elif high_threats >= 2:
            advantage -= 0.4  # Multiple high threats

        return np.clip(advantage, -1.0, 1.0)

    def _legacy_to_energy_space(
        self, throttle: float, desired_yaw: float, desired_pitch: float
    ) -> tuple:
        """
        Convert legacy throttle/yaw/pitch commands to energy-space [Ps, n, phi].
        Uses live envelope scalars for accurate normalization.

        Returns:
            Tuple of (ps_action, n_action, phi_action) in [0,1] normalized space
        """
        # Get current aircraft state
        max(self.aircraft.speed, 50.0)
        current_yaw = self.aircraft.yaw_deg
        current_pitch = getattr(self.aircraft, "pitch_deg", 0.0)

        # Get live envelope scalars if action processor available
        if self.action_processor is not None:
            try:
                envelope_scalars = self.action_processor.get_envelope_scalars(self.aircraft.id)
                ps_min = envelope_scalars.get("Ps_min", -50.0)
                ps_max = envelope_scalars.get("Ps_max", 100.0)
                n_max = envelope_scalars.get("n_max", 9.0)
                phi_max_deg = envelope_scalars.get("phi_max_deg", 85.0)
            except Exception:
                # Fallback to estimates
                ps_min, ps_max = -50.0, 100.0
                n_max = 9.0
                phi_max_deg = 85.0
        else:
            # Use estimates when no action processor
            ps_min, ps_max = -50.0, 100.0
            n_max = 9.0
            phi_max_deg = 85.0

        # Calculate heading error for bank angle
        yaw_diff = normalize_angle(desired_yaw - current_yaw)

        # Map yaw error to bank angle command (simple proportional)
        # Large heading errors → large bank angles
        desired_bank_deg = np.clip(yaw_diff * 0.8, -phi_max_deg, phi_max_deg)

        # Map to action space [0,1] where phi ranges from [-phi_max, +phi_max]
        phi_action = (desired_bank_deg + phi_max_deg) / (2.0 * phi_max_deg)
        phi_action = np.clip(phi_action, 0.0, 1.0)

        # Map pitch to load factor
        # Positive pitch (climb) → lower n (unload)
        # Negative pitch (dive) → higher n
        # Level flight → cruise n (~1.2-2.0)
        pitch_diff = desired_pitch - current_pitch

        if abs(yaw_diff) > 30:  # In turn
            # Use higher load factor for tighter turns
            desired_n = np.clip(2.5 + abs(yaw_diff) / 30.0, 2.0, min(6.0, n_max))
        elif pitch_diff < -10:  # Diving
            desired_n = 2.0  # Moderate G in dive
        elif pitch_diff > 10:  # Climbing
            desired_n = 0.8  # Unload for climb
        else:  # Level or small pitch changes
            desired_n = 1.5  # Cruise load factor

        # Map to action space [0,1] where n ranges from [n_min=-2, n_max]
        n_min = -2.0
        n_action = (desired_n - n_min) / (n_max - n_min)
        n_action = np.clip(n_action, 0.0, 1.0)

        # Map throttle to Ps command
        # High throttle → positive Ps (accelerate/climb)
        # Low throttle → negative Ps (decelerate/descend)
        if throttle > 0.7:  # High throttle - accelerate
            desired_ps = ps_max * ((throttle - 0.7) / 0.3) * 0.7
        elif throttle < 0.4:  # Low throttle - decelerate
            desired_ps = ps_min * ((0.4 - throttle) / 0.4) * 0.6
        else:  # Mid throttle - maintain energy
            desired_ps = 0.0

        # Map to action space [0,1] using live Ps range
        if (ps_max - ps_min) > 1e-3:
            ps_action = (desired_ps - ps_min) / (ps_max - ps_min)
        else:
            ps_action = 0.5  # Neutral if no range
        ps_action = np.clip(ps_action, 0.0, 1.0)

        return ps_action, n_action, phi_action

    def _yaw_to_action_space(self, desired_yaw: float) -> float:
        """Convert desired yaw to action space format (legacy, not used in energy-space)."""
        current_yaw = self.aircraft.yaw_deg
        yaw_diff = normalize_angle(desired_yaw - current_yaw)

        # Convert to 0-1 range where 0.5 = no change
        # Clamp to reasonable turn rates (±180 degrees)
        normalized_diff = (yaw_diff + 180) / 360
        return np.clip(normalized_diff, 0.0, 1.0)

    def _pitch_to_action_space(self, desired_pitch: float) -> float:
        """Convert desired pitch to action space format."""
        current_pitch = getattr(self.aircraft, "pitch_deg", 0.0)
        pitch_diff = desired_pitch - current_pitch

        # Convert to 0-1 range where 0.5 = no change
        # Clamp to reasonable pitch rates (±90 degrees)
        normalized_diff = (pitch_diff + 90) / 180
        return np.clip(normalized_diff, 0.0, 1.0)

    def _update_flight_controls(
        self, throttle: float, yaw: float, pitch: float, missile_fire: bool
    ):
        """Update internal flight control state."""
        self.flight_controls = {
            "throttle": throttle,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "missile_fire": missile_fire,
        }

    # Behavior tree action implementations
    def execute_defensive_maneuvers(self, context: dict[str, Any]) -> NodeStatus:
        """Execute defensive maneuvers against threats."""
        threats = context.get("threats", [])

        # Find most immediate missile threat
        missile_threats = [
            t for t in threats if t.threat_type == "missile" and t.threat_level == "critical"
        ]

        if missile_threats:
            closest_threat = min(missile_threats, key=lambda t: t.distance_m)

            # Execute notch maneuver
            result = self.bvr_tactics.execute_notch_maneuver(
                {"primary_missile_threat": closest_threat}
            )

            if result == NodeStatus.SUCCESS:
                # Apply defensive energy management
                self.energy_manager.execute_energy_dive(context)
                return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

    def execute_crank_maneuver(self, context: dict[str, Any]) -> NodeStatus:
        """Execute crank maneuver."""
        return self.bvr_tactics.execute_crank_maneuver(context)

    def execute_launch_and_leave(self, context: dict[str, Any]) -> NodeStatus:
        """Execute launch and leave tactics."""
        return self.bvr_tactics.execute_launch_and_leave(context)

    def execute_launch_and_decide(self, context: dict[str, Any]) -> NodeStatus:
        """Execute launch and decide tactics."""
        return self.bvr_tactics.execute_launch_and_decide(context)

    def execute_approach(self, context: dict[str, Any]) -> NodeStatus:
        """Execute approach to target."""
        # Use BVR approach with energy management
        bvr_result = self.bvr_tactics.execute_approach(context)

        if bvr_result == NodeStatus.SUCCESS:
            # Apply energy sustaining flight
            energy_result = self.energy_manager.execute_energy_sustain(context)
            return energy_result

        return bvr_result

    def execute_gun_attack(self, context: dict[str, Any]) -> NodeStatus:
        """Execute gun attack."""
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Calculate lead pursuit angle
        lead_angle = self.geometry_calc.calculate_lead_pursuit_angle(target)

        # Calculate approach vector
        target_bearing = geodetic_bearing_deg(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            target.position.lat,
            target.position.lon,
        )

        attack_bearing = normalize_angle(target_bearing + lead_angle)

        context["desired_yaw"] = attack_bearing
        context["desired_pitch"] = 0.0
        context["desired_throttle"] = 1.0  # Full throttle for gun attack

        return NodeStatus.SUCCESS

    def fire_short_range_missile(self, context: dict[str, Any]) -> NodeStatus:
        """Fire short-range missile."""
        # Check multiple sources for missile availability
        missiles_remaining = getattr(self.aircraft, "remaining_missiles", 0)
        weapons = getattr(self.aircraft, "weapons", None)  # weapons, not weapon_system
        max_missiles = getattr(self.aircraft, "max_missiles", 0)

        if weapons:
            missiles_remaining = max(missiles_remaining, getattr(weapons, "remaining_missiles", 0))

        # If no missiles found but max_missiles is set, assume full load
        if missiles_remaining == 0 and max_missiles > 0:
            missiles_remaining = max_missiles

        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(
            f"Aircraft {aircraft_id}: MISSILE_DEBUG - aircraft.remaining_missiles={getattr(self.aircraft, 'remaining_missiles', 'None')}, "
            f"weapons={weapons is not None}, max_missiles={max_missiles}, final_count={missiles_remaining}"
        )

        if missiles_remaining > 0:
            context["fire_missile"] = True
            print(
                f"Aircraft {aircraft_id}: SHORT_RANGE_MISSILE_FIRE - Missiles available: {missiles_remaining}"
            )
            return NodeStatus.SUCCESS
        else:
            print(
                f"Aircraft {aircraft_id}: SHORT_RANGE_MISSILE_FAIL - No missiles available (checked: remaining_missiles={getattr(self.aircraft, 'remaining_missiles', 0)}, weapons={weapons})"
            )
            return NodeStatus.FAILURE

    def execute_bfm(self, context: dict[str, Any]) -> NodeStatus:
        """Execute basic fighter maneuvers."""
        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Simple BFM: try to get behind target
        target_heading = getattr(target, "yaw_deg", 0)
        intercept_bearing = normalize_angle(target_heading + 180)  # Go to target's 6 o'clock

        context["desired_yaw"] = intercept_bearing
        context["desired_pitch"] = 0.0
        context["desired_throttle"] = 0.9

        return NodeStatus.SUCCESS

    def execute_search_patrol(self, context: dict[str, Any]) -> NodeStatus:
        """Execute search and patrol behavior with boundary avoidance and grid search."""
        # Simple patrol: maintain altitude and speed
        self.energy_manager.execute_energy_sustain(context)

        # Check for boundary avoidance first (highest priority)
        boundary_heading = self._check_boundary_avoidance(context)
        if boundary_heading is not None:
            context["desired_yaw"] = boundary_heading
            return NodeStatus.SUCCESS

        # Execute coordinated grid search
        search_heading = self._execute_coordinated_grid_search(context)
        context["desired_yaw"] = search_heading

        return NodeStatus.SUCCESS

    def _check_boundary_avoidance(self, context: dict[str, Any]) -> float | None:
        """
        Check if aircraft is near map boundary and return avoidance heading.

        Returns:
            Heading to avoid boundary, or None if no avoidance needed
        """
        # Get actual map boundaries from aircraft's map_limits
        map_limits = getattr(self.aircraft, "map_limits", None)
        if not map_limits:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            print(f"Aircraft {aircraft_id}: WARNING - No map_limits found!")
            return None

        # Get aircraft position in degrees
        pos_lat = self.aircraft.position.lat
        pos_lon = self.aircraft.position.lon

        # Define buffer zone in degrees (proportional to map size)
        map_width_deg = map_limits.right_lon - map_limits.left_lon
        map_height_deg = map_limits.top_lat - map_limits.bottom_lat

        # Use 20% of map size as buffer zone
        buffer_lon = map_width_deg * 0.2
        buffer_lat = map_height_deg * 0.2

        # Calculate boundary thresholds
        left_threshold = map_limits.left_lon + buffer_lon
        right_threshold = map_limits.right_lon - buffer_lon
        bottom_threshold = map_limits.bottom_lat + buffer_lat
        top_threshold = map_limits.top_lat - buffer_lat

        # Check which boundaries we're approaching
        near_left = pos_lon < left_threshold
        near_right = pos_lon > right_threshold
        near_bottom = pos_lat < bottom_threshold
        near_top = pos_lat > top_threshold

        # Calculate avoidance heading based on boundary proximity
        avoid_heading = None

        # Handle corner cases first (diagonal avoidance)
        if near_right and near_top:
            avoid_heading = 225.0  # SW
        elif near_right and near_bottom:
            avoid_heading = 315.0  # NW
        elif near_left and near_top:
            avoid_heading = 135.0  # SE
        elif near_left and near_bottom:
            avoid_heading = 45.0  # NE
        # Handle edge cases (straight avoidance)
        elif near_right:
            avoid_heading = 270.0  # Turn west
        elif near_left:
            avoid_heading = 90.0  # Turn east
        elif near_top:
            avoid_heading = 180.0  # Turn south
        elif near_bottom:
            avoid_heading = 0.0  # Turn north

        # Log boundary avoidance activation
        if avoid_heading is not None:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            boundaries = []
            if near_left:
                boundaries.append("LEFT")
            if near_right:
                boundaries.append("RIGHT")
            if near_top:
                boundaries.append("TOP")
            if near_bottom:
                boundaries.append("BOTTOM")
            print(
                f"Aircraft {aircraft_id}: BOUNDARY AVOIDANCE ACTIVE - Near {'/'.join(boundaries)} - "
                f"Position: ({pos_lon:.3f}, {pos_lat:.3f}) - Avoiding to {avoid_heading:.0f}°"
            )

        return avoid_heading

    def _execute_coordinated_grid_search(self, context: dict[str, Any]) -> float:
        """
        Execute coordinated grid search pattern based on team composition.

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

        print(
            f"Aircraft {aircraft_id}: GRID SEARCH - Heading: {heading:.0f}° (Progress: {cycle_progress:.2f})"
        )

        return heading

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive controller status."""
        return {
            "tactical_state": self.current_tactical_state,
            "current_target": getattr(self.target, "id", None) if self.target else None,
            "behavior_tree_active": True,
            "current_maneuver": self.bvr_tactics.get_current_maneuver(),
            "flight_controls": self.flight_controls.copy(),
            "energy_state": self.energy_manager.get_energy_state_summary(),
            "automation_status": self.auto_helper.get_status(),
        }
