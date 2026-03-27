"""
Tactical behavior tree for air-to-air combat decision making.
Implements the main decision logic using behavior tree patterns.
Uses ObservationHelper for all geometry calculations (single source of truth).
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.aircrafts.systems.observation_helper import ObservationHelper
from air_to_air_rl.automation.scripted_control.behavior_tree.nodes import (
    ActionNode,
    BehaviorNode,
    ConditionNode,
    NodeStatus,
    ParallelNode,
    SelectorNode,
    SequenceNode,
)


class TacticalBehaviorTree:
    """Main tactical behavior tree for air-to-air combat AI."""

    def __init__(self, aircraft, tactical_controller):
        self.aircraft = aircraft
        self.tactical_controller = tactical_controller
        self.root = self._build_tree()
        self.context = {}

        # ObservationHelper for consistent geometry/DLZ/NEZ (same as RL observations)
        self.obs_helper = ObservationHelper(aircraft)

        # Missile tracking to limit shots per target
        self.missiles_fired_at_targets = {}  # target_id -> [missile_ids]
        self.max_missiles_per_target = 2  # Maximum missiles to fire at same target
        self.missile_cooldown_time = 10.0  # Seconds to wait before re-engaging same target

    def _build_tree(self) -> BehaviorNode:
        """Build the main tactical behavior tree."""

        # Root selector - choose between different tactical modes
        root = SelectorNode("TacticalRoot")

        # Emergency/Defensive sequence (highest priority)
        emergency_seq = SequenceNode("EmergencyDefense")
        emergency_seq.add_child(ConditionNode("IncomingMissile", self._incoming_missile_threat))
        emergency_seq.add_child(ActionNode("DefensiveManeuvers", self._execute_defensive_maneuvers))
        root.add_child(emergency_seq)

        # BVR Combat sequence
        bvr_seq = SequenceNode("BVRCombat")
        bvr_seq.add_child(ConditionNode("HasBVRTarget", self._has_bvr_target))
        bvr_seq.add_child(self._build_bvr_subtree())
        root.add_child(bvr_seq)

        # WVR Combat sequence
        wvr_seq = SequenceNode("WVRCombat")
        wvr_seq.add_child(ConditionNode("HasWVRTarget", self._has_wvr_target))
        wvr_seq.add_child(self._build_wvr_subtree())
        root.add_child(wvr_seq)

        # Search and patrol (default behavior)
        search_seq = SequenceNode("SearchPatrol")
        search_seq.add_child(ActionNode("PatrolSearch", self._execute_search_patrol))
        root.add_child(search_seq)

        return root

    def _build_bvr_subtree(self) -> BehaviorNode:
        """Build advanced BVR (Beyond Visual Range) combat subtree with sophisticated tactics."""

        bvr_selector = SelectorNode("AdvancedBVRTactics")

        # === DEFENSIVE PRIORITIES (Highest Priority) ===

        # Defensive Beam/Notch sequence (immediate threat response)
        beam_notch_seq = SequenceNode("DefensiveBeamNotch")
        beam_notch_seq.add_child(ConditionNode("UnderRadarThreat", self._under_radar_threat))
        beam_notch_seq.add_child(ActionNode("ExecuteBeamManeuver", self._execute_beam_maneuver))
        bvr_selector.add_child(beam_notch_seq)

        # F-Pole Defense sequence (missile going active soon)
        f_pole_seq = SequenceNode("FPoleDefense")
        f_pole_seq.add_child(ConditionNode("NearFPole", self._near_f_pole))
        f_pole_seq.add_child(ActionNode("ExecuteFPoleDefense", self._execute_f_pole_defense))
        bvr_selector.add_child(f_pole_seq)

        # === TARGET ACQUISITION ===

        # Priority Target Selection sequence (advanced target prioritization)
        target_select_seq = SequenceNode("PriorityTargetSelection")
        target_select_seq.add_child(
            ConditionNode("NeedsTargetSelection", self._needs_target_selection)
        )
        target_select_seq.add_child(
            ActionNode("ExecutePriorityTargeting", self._execute_priority_targeting)
        )
        bvr_selector.add_child(target_select_seq)

        # Target Acquisition sequence (enhanced with geometry awareness)
        target_acq_seq = SequenceNode("GeometryAwareTargetAcquisition")
        target_acq_seq.add_child(ConditionNode("NeedsTargetLock", self._needs_target_lock))
        target_acq_seq.add_child(
            ActionNode(
                "ExecuteAdvancedTargetAcquisition", self._execute_advanced_target_acquisition
            )
        )
        bvr_selector.add_child(target_acq_seq)

        # === ADVANCED POSITIONING ===

        # Crank sequence (energy-optimized cranking)
        advanced_crank_seq = SequenceNode("EnergyOptimizedCrank")
        advanced_crank_seq.add_child(
            ConditionNode("ShouldAdvancedCrank", self._should_advanced_crank)
        )
        advanced_crank_seq.add_child(
            ActionNode("ExecuteEnergyOptimizedCrank", self._execute_energy_optimized_crank)
        )
        bvr_selector.add_child(advanced_crank_seq)

        # Multi-Aspect Attack sequence (attacking from optimal angles)
        multi_aspect_seq = SequenceNode("MultiAspectAttack")
        multi_aspect_seq.add_child(
            ConditionNode("OptimalAttackGeometry", self._optimal_attack_geometry)
        )
        multi_aspect_seq.add_child(
            ActionNode("ExecuteMultiAspectAttack", self._execute_multi_aspect_attack)
        )
        bvr_selector.add_child(multi_aspect_seq)

        # === LAUNCH TACTICS ===

        # Advanced Launch and Leave (with energy state consideration)
        adv_launch_leave_seq = SequenceNode("AdvancedLaunchLeave")
        adv_launch_leave_seq.add_child(
            ConditionNode("InAdvancedLaunchRange", self._in_advanced_launch_range)
        )
        adv_launch_leave_seq.add_child(
            ConditionNode("ShouldAdvancedLaunchLeave", self._should_advanced_launch_leave)
        )
        adv_launch_leave_seq.add_child(
            ActionNode("ExecuteAdvancedLaunchLeave", self._execute_advanced_launch_leave)
        )
        bvr_selector.add_child(adv_launch_leave_seq)

        # Drag Launch sequence (extended range shots with energy advantage)
        drag_launch_seq = SequenceNode("DragLaunch")
        drag_launch_seq.add_child(ConditionNode("CanDragLaunch", self._can_drag_launch))
        drag_launch_seq.add_child(ActionNode("ExecuteDragLaunch", self._execute_drag_launch))
        bvr_selector.add_child(drag_launch_seq)

        # Launch and Decide sequence (enhanced with post-launch tactics)
        adv_launch_decide_seq = SequenceNode("AdvancedLaunchDecide")
        adv_launch_decide_seq.add_child(
            ConditionNode("InAdvancedLaunchRange", self._in_advanced_launch_range)
        )
        adv_launch_decide_seq.add_child(
            ConditionNode("ShouldAdvancedLaunchDecide", self._should_advanced_launch_decide)
        )
        adv_launch_decide_seq.add_child(
            ActionNode("ExecuteAdvancedLaunchDecide", self._execute_advanced_launch_decide)
        )
        bvr_selector.add_child(adv_launch_decide_seq)

        # === ENERGY MANAGEMENT ===

        # Energy Position sequence (gain altitude/energy advantage)
        energy_pos_seq = SequenceNode("EnergyPositioning")
        energy_pos_seq.add_child(
            ConditionNode("NeedsEnergyAdvantage", self._needs_energy_advantage)
        )
        energy_pos_seq.add_child(
            ActionNode("ExecuteEnergyPositioning", self._execute_energy_positioning)
        )
        bvr_selector.add_child(energy_pos_seq)

        # Approach sequence (enhanced with energy considerations)
        adv_approach_seq = SequenceNode("EnergyAwareApproach")
        adv_approach_seq.add_child(ConditionNode("NotInRange", self._not_in_launch_range))
        adv_approach_seq.add_child(
            ActionNode("ExecuteEnergyAwareApproach", self._execute_energy_aware_approach)
        )
        bvr_selector.add_child(adv_approach_seq)

        return bvr_selector

    def _build_wvr_subtree(self) -> BehaviorNode:
        """Build WVR (Within Visual Range) combat subtree."""

        wvr_selector = SelectorNode("WVRTactics")

        # Gun engagement
        gun_seq = SequenceNode("GunEngagement")
        gun_seq.add_child(ConditionNode("InGunRange", self._in_gun_range))
        gun_seq.add_child(ConditionNode("GoodGunAspect", self._good_gun_aspect))
        gun_seq.add_child(ActionNode("ExecuteGunAttack", self._execute_gun_attack))
        wvr_selector.add_child(gun_seq)

        # Short range missile
        short_missile_seq = SequenceNode("ShortRangeMissile")
        short_missile_seq.add_child(
            ConditionNode("HasShortRangeMissiles", self._has_short_range_missiles)
        )
        short_missile_seq.add_child(ConditionNode("GoodIRLock", self._good_ir_lock))
        short_missile_seq.add_child(
            ActionNode("FireShortRangeMissile", self._fire_short_range_missile)
        )
        wvr_selector.add_child(short_missile_seq)

        # Maneuvering for advantage
        maneuver_seq = SequenceNode("ManeuverForAdvantage")
        maneuver_seq.add_child(ActionNode("ExecuteBFM", self._execute_bfm))
        wvr_selector.add_child(maneuver_seq)

        return wvr_selector

    def tick(self, context: dict[str, Any]) -> NodeStatus:
        """Execute one iteration of the behavior tree."""
        self.context = context

        # Update missile tracking at each tick
        self._update_missile_tracking(context)

        return self.root.tick(context)

    # Condition Functions
    def _incoming_missile_threat(self, context: dict[str, Any]) -> bool:
        """Check if there's an incoming missile threat requiring immediate defensive action."""
        threats = context.get("threats", [])
        for threat in threats:
            if (
                threat.threat_type == "missile"
                and threat.threat_level in ["critical", "high"]
                and threat.distance_m < 15000
            ):  # 15km immediate threat (realistic for high-speed missiles)
                return True
        return False

    def _has_bvr_target(self, context: dict[str, Any]) -> bool:
        """Check if we have a BVR target."""
        target = context.get("target")
        if not target:
            return False
        target_range = context.get("target_range_m", float("inf"))
        has_bvr = target_range > 10000  # 10km+ is BVR (realistic long-range engagement)

        aircraft_id = getattr(self.aircraft, "id", "unknown")
        if has_bvr:
            print(
                f"Aircraft {aircraft_id}: HAS_BVR_TARGET - Range: {target_range / 1000:.1f}km (>3km = BVR)"
            )

        return has_bvr

    def _has_wvr_target(self, context: dict[str, Any]) -> bool:
        """Check if we have a WVR target."""
        target = context.get("target")
        if not target:
            return False
        target_range = context.get("target_range_m", float("inf"))
        return target_range <= 10000  # 10km or less is WVR

    def _should_crank(self, context: dict[str, Any]) -> bool:
        """Determine if we should execute a crank maneuver using percentage-based ranges."""
        target_range = context.get("target_range_m", float("inf"))
        closure_rate = context.get("closure_rate_mps", 0)
        aspect_angle = context.get("target_aspect_deg", 0)
        max_range = context.get("missile_max_range_m", 100000)

        # Use percentage-based crank range (30% to 80% of missile max range)
        crank_min = max_range * 0.30  # 30km for 100km missile
        crank_max = max_range * 0.80  # 80km for 100km missile

        # Crank if we're closing too fast on a target in medium range
        should_crank = (
            crank_min < target_range < crank_max  # Percentage-based range
            and closure_rate > 200  # Closing > 200 m/s
            and abs(aspect_angle) > 30
        )  # Not head-on

        if should_crank:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            print(
                f"Aircraft {aircraft_id}: SHOULD_CRANK - Range: {target_range / 1000:.1f}km, Band: {crank_min / 1000:.1f}-{crank_max / 1000:.1f}km, Closure: {closure_rate:.0f}m/s"
            )

        return should_crank

    def _in_launch_range(self, context: dict[str, Any]) -> bool:
        """
        Check if target is in missile launch range.
        Uses WeaponSystem.check_fire_feasibility() as single source of truth.
        Adds tactical criteria (closure rate, range envelope) on top of basic gates.
        """
        target = context.get("target")
        if not target:
            return False

        # Get fire feasibility from WeaponSystem (lock, FOV, inventory, gimbal checks)
        if not hasattr(self.aircraft, "weapons"):
            return False

        feasibility = self.aircraft.weapons.check_fire_feasibility(target)

        # If WeaponSystem says no, respect it
        if not feasibility["can_fire"]:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            print(
                f"Aircraft {aircraft_id}: LAUNCH_VETO - {feasibility.get('veto_reason', 'unknown')}"
            )
            return False

        # WeaponSystem says OK - now add tactical criteria
        target_range = context.get("target_range_m", float("inf"))
        nez_out_range = context.get("nez_out_range_m", 0)
        closure_rate = context.get("closure_rate_mps", 0)
        r_aero = context.get("r_aero_m", 100000)

        # Tactical range envelope (DLZ-based)
        radar_max_range = (
            getattr(self.aircraft.radar, "max_range_m", 100000)
            if hasattr(self.aircraft, "radar")
            else 100000
        )
        max_fire_range = min(max(nez_out_range, r_aero * 0.9), radar_max_range * 0.95)

        # Tactical criteria (stricter than WeaponSystem basic gates)
        range_ok = target_range <= max_fire_range
        closure_ok = closure_rate > 10  # Minimum closure requirement (tactical doctrine)

        # Check missile tracking limits (seeker FOV, gimbal)
        can_fire_at_target = self._can_fire_at_target(target, context)

        # Final decision: WeaponSystem OK + tactical criteria
        in_range = range_ok and closure_ok and can_fire_at_target

        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(
            f"Aircraft {aircraft_id}: LAUNCH_CHECK - Range: {target_range / 1000:.1f}km, NEZ: {nez_out_range / 1000:.1f}km, "
            f"MaxFire: {max_fire_range / 1000:.1f}km, Closure: {closure_rate:.0f}m/s, "
            f"WpnSys: OK, Tactical: {'OK' if in_range else 'FAIL'}"
        )

        return in_range

    def _should_launch_leave(self, context: dict[str, Any]) -> bool:
        """Determine if we should use launch-and-leave tactics based on F-pole and DOR."""
        target_range = context.get("target_range_m", float("inf"))
        f_pole = context.get("f_pole_m", 0)
        dor = context.get("desired_out_range_m", 20000)
        num_enemies = len(context.get("threats", []))

        # Use Launch & Leave if F-pole is small OR we're at/inside DOR OR multiple threats
        should_skate = (
            f_pole < 10000  # F-pole < 10km (close to pitbull)
            or target_range <= dor  # At or inside desired out range
            or num_enemies > 1
        )  # Multiple threats

        if should_skate:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            print(
                f"Aircraft {aircraft_id}: LAUNCH_LEAVE - Range: {target_range / 1000:.1f}km, F-pole: {f_pole / 1000:.1f}km, "
                f"DOR: {dor / 1000:.1f}km, Threats: {num_enemies}"
            )

        return should_skate

    def _should_launch_decide(self, context: dict[str, Any]) -> bool:
        """Determine if we should use launch-and-decide tactics."""
        should_banzai = not self._should_launch_leave(context)

        if should_banzai:
            aircraft_id = getattr(self.aircraft, "id", "unknown")
            target_range = context.get("target_range_m", 0)
            f_pole = context.get("f_pole_m", 0)
            print(
                f"Aircraft {aircraft_id}: LAUNCH_DECIDE - Range: {target_range / 1000:.1f}km, F-pole: {f_pole / 1000:.1f}km (staying aggressive)"
            )

        return should_banzai

    def _not_in_launch_range(self, context: dict[str, Any]) -> bool:
        """Check if target is not yet in launch range."""
        return not self._in_launch_range(context)

    def _in_gun_range(self, context: dict[str, Any]) -> bool:
        """Check if target is in gun range."""
        target_range = context.get("target_range_m", float("inf"))
        return target_range < 1000  # 1km gun range

    def _good_gun_aspect(self, context: dict[str, Any]) -> bool:
        """Check if we have good aspect for gun engagement."""
        angle_off_tail = context.get("angle_off_tail_deg", 180)
        return angle_off_tail < 45  # Within 45 degrees of target's tail

    def _has_short_range_missiles(self, context: dict[str, Any]) -> bool:
        """Check if we have short-range missiles available."""
        return context.get("short_range_missiles", 0) > 0

    def _good_ir_lock(self, context: dict[str, Any]) -> bool:
        """Check if we have good IR lock conditions."""
        target_aspect = abs(context.get("target_aspect_deg", 0))
        return target_aspect > 90  # Rear aspect for IR missiles

    # Action Functions
    def _execute_defensive_maneuvers(self, context: dict[str, Any]) -> NodeStatus:
        """Execute defensive maneuvers against incoming missiles."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        threats = context.get("threats", [])
        missile_threats = [t for t in threats if t.threat_type == "missile"]
        print(
            f"Aircraft {aircraft_id}: DEFENSIVE MANEUVERS - {len(missile_threats)} missile threats"
        )
        # Implement notch, beam, and evasive maneuvers
        return self.tactical_controller.execute_defensive_maneuvers(context)

    def _execute_crank(self, context: dict[str, Any]) -> NodeStatus:
        """Execute crank maneuver."""
        return self.tactical_controller.execute_crank_maneuver(context)

    def _execute_launch_leave(self, context: dict[str, Any]) -> NodeStatus:
        """Execute launch-and-leave tactics with target pointing."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: EXECUTING LAUNCH-AND-LEAVE TACTICS")

        # Ensure aircraft is pointing towards target before launch
        self._ensure_target_pointing(context)

        # Execute launch
        result = self.tactical_controller.execute_launch_and_leave(context)

        # Record missile launch if successful
        if result == NodeStatus.SUCCESS and context.get("fire_missile", False):
            self._record_missile_launch_from_context(context)

        return result

    def _execute_launch_decide(self, context: dict[str, Any]) -> NodeStatus:
        """Execute launch-and-decide tactics with target pointing."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: EXECUTING LAUNCH-AND-DECIDE TACTICS")

        # Ensure aircraft is pointing towards target before launch
        self._ensure_target_pointing(context)

        # Execute launch
        result = self.tactical_controller.execute_launch_and_decide(context)

        # Record missile launch if successful
        if result == NodeStatus.SUCCESS and context.get("fire_missile", False):
            self._record_missile_launch_from_context(context)

        return result

    def _execute_approach(self, context: dict[str, Any]) -> NodeStatus:
        """Execute approach to target."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        target_range = context.get("target_range_m", 0) / 1000
        print(f"Aircraft {aircraft_id}: APPROACHING TARGET - Range: {target_range:.1f}km")

        # Check boundary limits first (highest priority)
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading
            print(f"  Boundary avoidance during approach: heading {boundary_heading:.0f}°")

        return self.tactical_controller.execute_approach(context)

    def _execute_gun_attack(self, context: dict[str, Any]) -> NodeStatus:
        """Execute gun attack."""
        return self.tactical_controller.execute_gun_attack(context)

    def _fire_short_range_missile(self, context: dict[str, Any]) -> NodeStatus:
        """Fire short-range missile with target pointing."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        target_range = context.get("target_range_m", 0) / 1000
        print(f"Aircraft {aircraft_id}: FIRING SHORT-RANGE MISSILE - Range: {target_range:.1f}km")

        # Ensure aircraft is pointing towards target before launch
        self._ensure_target_pointing(context)

        # Execute launch
        result = self.tactical_controller.fire_short_range_missile(context)

        # Record missile launch if successful
        if result == NodeStatus.SUCCESS and context.get("fire_missile", False):
            self._record_missile_launch_from_context(context)

        return result

    def _execute_bfm(self, context: dict[str, Any]) -> NodeStatus:
        """Execute basic fighter maneuvers."""
        return self.tactical_controller.execute_bfm(context)

    def _execute_search_patrol(self, context: dict[str, Any]) -> NodeStatus:
        """Execute search and patrol behavior."""
        return self.tactical_controller.execute_search_patrol(context)

    # Target Acquisition and Lock Management Methods
    def _needs_target_lock(self, context: dict[str, Any]) -> bool:
        """Check if we need to acquire a target lock (highest priority)."""
        target = context.get("target")
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # Check if we have any targets within radar range first
        if not target:
            if self._has_targets_in_radar_range(context):
                # Have targets in radar range but none assigned - need target acquisition
                print(
                    f"Aircraft {aircraft_id}: NEEDS_TARGET_LOCK - Targets in radar range but none assigned"
                )
                return True
            else:
                # No targets within radar range - will trigger grid search
                print(f"Aircraft {aircraft_id}: NEEDS_TARGET_LOCK - No targets within radar range")
                return True

        # Check target range - if beyond radar range, switch to search
        target_range = context.get("target_range_m", float("inf"))
        radar_range = (
            getattr(self.aircraft.radar, "max_range_m", 100000)
            if hasattr(self.aircraft, "radar")
            else 100000
        )

        if target_range > radar_range:
            print(
                f"Aircraft {aircraft_id}: NEEDS_TARGET_LOCK - Target beyond radar range ({target_range / 1000:.1f}km > {radar_range / 1000:.1f}km)"
            )
            return True

        # Check if we have radar lock on current target
        if hasattr(self.aircraft, "sensor") and self.aircraft.sensor:
            has_lock = self.aircraft.sensor.has_radar_lock(target)
            if not has_lock:
                print(
                    f"Aircraft {aircraft_id}: NEEDS_TARGET_LOCK - Target at {target_range / 1000:.1f}km but no radar lock"
                )
                return True

        # Check if target is within gimbal/FOV limits for lock maintenance
        gimbal_ok = context.get("gimbal_ok", True)
        if not gimbal_ok:
            print(f"Aircraft {aircraft_id}: NEEDS_TARGET_LOCK - Target outside gimbal limits")
            return True

        return False

    def _has_targets_in_radar_range(self, context: dict[str, Any]) -> bool:
        """Check if there are any targets within radar detection range."""
        if not (hasattr(self.aircraft, "sensor") and self.aircraft.sensor):
            return False

        # Check sensor tracks or detected targets
        sensor_tracks = getattr(self.aircraft.sensor, "sensor_tracks", [])
        if sensor_tracks:
            return True

        # Fallback: check for any enemy units within radar range
        # This would need access to simulator - for now assume context provides this info
        threats = context.get("threats", [])
        radar_range = (
            getattr(self.aircraft.radar, "max_range_m", 100000)
            if hasattr(self.aircraft, "radar")
            else 100000
        )

        for threat in threats:
            if hasattr(threat, "distance_m") and threat.distance_m <= radar_range:
                return True

        return False

    def _has_lock_needs_maintenance(self, context: dict[str, Any]) -> bool:
        """Check if we have a lock but it needs maintenance/optimization."""
        target = context.get("target")
        if not target:
            return False

        # Must have a lock to need maintenance
        if not (
            hasattr(self.aircraft, "sensor")
            and self.aircraft.sensor
            and self.aircraft.sensor.has_radar_lock(target)
        ):
            return False

        target_range = context.get("target_range_m", float("inf"))
        closure_rate = context.get("closure_rate_mps", 0)
        gimbal_ok = context.get("gimbal_ok", True)

        # Need maintenance if:
        # 1. Target is getting close to gimbal limits
        # 2. Closure rate is very high (might lose lock soon)
        # 3. Target is at edge of engagement envelope

        aircraft_id = getattr(self.aircraft, "id", "unknown")

        if not gimbal_ok:
            print(f"Aircraft {aircraft_id}: LOCK_MAINTENANCE - Gimbal limits reached")
            return True

        if closure_rate > 400:  # Very high closure rate
            print(
                f"Aircraft {aircraft_id}: LOCK_MAINTENANCE - High closure rate {closure_rate:.0f}m/s"
            )
            return True

        if target_range < 5000:  # Very close - might lose lock
            print(
                f"Aircraft {aircraft_id}: LOCK_MAINTENANCE - Very close range {target_range / 1000:.1f}km"
            )
            return True

        return False

    def _execute_target_acquisition(self, context: dict[str, Any]) -> NodeStatus:
        """Execute target acquisition maneuvers to get radar lock or coordinated search."""
        target = context.get("target")
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # Check boundary limits first (highest priority)
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["desired_yaw"] = boundary_heading
            context["desired_pitch"] = 0.0  # Level flight for boundary avoidance
            context["desired_throttle"] = 0.8
            print(
                f"Aircraft {aircraft_id}: TARGET_ACQUISITION - Boundary avoidance: heading {boundary_heading:.0f}°"
            )
            return NodeStatus.RUNNING

        if not target or not self._has_targets_in_radar_range(context):
            # No target or no targets in radar range - execute coordinated grid search
            print(f"Aircraft {aircraft_id}: TARGET_ACQUISITION - Executing coordinated grid search")

            # Use the coordinated grid search from tactical controller
            if hasattr(self.tactical_controller, "_execute_coordinated_grid_search"):
                search_heading = self.tactical_controller._execute_coordinated_grid_search(context)
                context["desired_yaw"] = search_heading
                context["desired_pitch"] = 0.0  # Level flight for maximum radar coverage
                context["desired_throttle"] = 0.75  # Moderate speed for search
                print(f"  Grid search heading: {search_heading:.1f}°")
            else:
                # Fallback to simple search pattern if coordinated search unavailable
                context["desired_yaw"] = self.aircraft.yaw_deg + 30.0  # Gentle search turn
                context["desired_pitch"] = 0.0
                context["desired_throttle"] = 0.7
                print("  Simple search pattern")

            return NodeStatus.RUNNING

        # Have target in radar range but no lock - point at target for lock acquisition
        target_bearing = context.get("target_bearing_deg", 0)
        target_range = context.get("target_range_m", 0)

        print(
            f"Aircraft {aircraft_id}: TARGET_ACQUISITION - Acquiring lock on target at {target_range / 1000:.1f}km"
        )

        # Point directly at target for optimal lock acquisition
        context["desired_yaw"] = target_bearing
        context["desired_pitch"] = 0.0  # Level flight for stable lock
        context["desired_throttle"] = 0.8  # Steady approach

        return NodeStatus.RUNNING

    def _execute_lock_maintenance(self, context: dict[str, Any]) -> NodeStatus:
        """Execute maneuvers to maintain radar lock while optimizing engagement geometry."""
        target = context.get("target")
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        if not target:
            return NodeStatus.FAILURE

        target_bearing = context.get("target_bearing_deg", 0)
        target_range = context.get("target_range_m", 0)
        closure_rate = context.get("closure_rate_mps", 0)

        print(
            f"Aircraft {aircraft_id}: LOCK_MAINTENANCE - Maintaining lock at {target_range / 1000:.1f}km, "
            f"closure: {closure_rate:.0f}m/s"
        )

        # Check boundary limits first (highest priority)
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["desired_yaw"] = boundary_heading
            context["desired_pitch"] = 0.0  # Level flight for boundary avoidance
            context["desired_throttle"] = 0.8
            print(f"  Boundary avoidance: heading {boundary_heading:.0f}°")
            return NodeStatus.RUNNING

        # Use proper BVR engagement tactics instead of simple lock maintenance
        # Delegate to the tactical controller for proper BVR maneuvers
        if hasattr(self.tactical_controller, "execute_enhanced_approach"):
            return self.tactical_controller.execute_enhanced_approach(context)
        elif hasattr(self.tactical_controller, "execute_approach"):
            return self.tactical_controller.execute_approach(context)
        else:
            # Fallback: maintain lock with level flight (no upward climb)
            context["desired_yaw"] = target_bearing
            context["desired_pitch"] = 0.0  # Level flight, not climbing
            context["desired_throttle"] = 0.75
            print("  Fallback lock maintenance")
            return NodeStatus.RUNNING

    def _check_boundary_limits(self, context: dict[str, Any]) -> float | None:
        """Check if aircraft is approaching map boundaries and return avoidance heading."""
        try:
            # Get map limits from simulator or context
            simulator = context.get("simulator")
            if simulator and hasattr(simulator, "map_limits"):
                map_limits = simulator.map_limits
            elif hasattr(self.aircraft, "map_limits"):
                map_limits = self.aircraft.map_limits
            else:
                # No map limits available
                return None

            # Current position
            pos = self.aircraft.position

            # Safety buffer (15% of map size)
            lat_range = map_limits.top_lat - map_limits.bottom_lat
            lon_range = map_limits.right_lon - map_limits.left_lon
            lat_buffer = lat_range * 0.15
            lon_buffer = lon_range * 0.15

            # Check boundaries
            safe_top = map_limits.top_lat - lat_buffer
            safe_bottom = map_limits.bottom_lat + lat_buffer
            safe_right = map_limits.right_lon - lon_buffer
            safe_left = map_limits.left_lon + lon_buffer

            # Determine avoidance direction
            if pos.lat > safe_top:  # Too far north
                return 180.0  # Head south
            elif pos.lat < safe_bottom:  # Too far south
                return 0.0  # Head north
            elif pos.lon > safe_right:  # Too far east
                return 270.0  # Head west
            elif pos.lon < safe_left:  # Too far west
                return 90.0  # Head east

            return None  # Within safe boundaries

        except Exception:
            # If boundary checking fails, don't interfere
            return None

    def _ensure_target_pointing(self, context: dict[str, Any]) -> None:
        """Ensure aircraft is pointing towards target before missile launch."""
        target = context.get("target")
        if not target:
            return

        target_bearing = context.get("target_bearing_deg")
        if target_bearing is None:
            # Calculate target bearing using ObservationHelper (consistent with observations)
            try:
                geom = self.obs_helper.get_geometry_kinematics(target)
                if geom["valid"]:
                    target_bearing = geom["bearing_deg"]
                    context["target_bearing_deg"] = target_bearing
                else:
                    return
            except Exception:
                return

        current_heading = self.aircraft.yaw_deg
        heading_error = target_bearing - current_heading

        # Normalize to [-180, 180]
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360

        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # Check if we're reasonably pointed at target
        if abs(heading_error) > 15.0:  # More than 15 degrees off
            print(
                f"Aircraft {aircraft_id}: Adjusting heading by {heading_error:.1f}° to point at target before launch"
            )

            # Set desired heading towards target
            context["desired_yaw"] = target_bearing
            context["desired_pitch"] = 0.0  # Level flight for stable launch
            context["desired_throttle"] = 0.75  # Maintain speed

            # Add a small delay to allow pointing adjustment
            context["pre_launch_pointing"] = True
        else:
            print(
                f"Aircraft {aircraft_id}: Good target pointing (error: {heading_error:.1f}°), ready to launch"
            )
            context["pre_launch_pointing"] = False

    def _can_fire_at_target(self, target, context: dict[str, Any]) -> bool:
        """Check if we can fire at this target based on active missiles and limits."""
        if not target:
            return False

        target_id = getattr(target, "id", str(target))
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        # Update missile tracking - remove missiles that are no longer active
        self._update_missile_tracking(context)

        # Check how many missiles we have fired at this target that are still active
        active_missiles_at_target = len(self.missiles_fired_at_targets.get(target_id, []))

        if active_missiles_at_target >= self.max_missiles_per_target:
            print(
                f"Aircraft {aircraft_id}: Cannot fire at target {target_id} - already have {active_missiles_at_target} missiles in flight (max: {self.max_missiles_per_target})"
            )
            return False

        print(
            f"Aircraft {aircraft_id}: Can fire at target {target_id} - currently {active_missiles_at_target}/{self.max_missiles_per_target} missiles in flight"
        )
        return True

    def _update_missile_tracking(self, context: dict[str, Any]) -> None:
        """Update missile tracking by removing missiles that are no longer active."""
        simulator = context.get("simulator")
        if not simulator or not hasattr(simulator, "active_units"):
            return

        # Get all active missile IDs
        active_missile_ids = set()
        for unit_id, unit in simulator.active_units.items():
            if hasattr(unit, "name") and "missile" in unit.name.lower():
                # Check if this missile belongs to our aircraft
                if hasattr(unit, "origin_aircraft_id") and unit.origin_aircraft_id == getattr(
                    self.aircraft, "id"
                ):
                    active_missile_ids.add(unit_id)
                elif (
                    hasattr(unit, "parent")
                    and hasattr(unit.parent, "id")
                    and unit.parent.id == getattr(self.aircraft, "id")
                ):
                    active_missile_ids.add(unit_id)

        # Clean up tracking dictionary - remove missiles that are no longer active
        for target_id in list(self.missiles_fired_at_targets.keys()):
            missile_list = self.missiles_fired_at_targets[target_id]
            # Keep only missiles that are still active
            active_missiles = [mid for mid in missile_list if mid in active_missile_ids]
            if active_missiles:
                self.missiles_fired_at_targets[target_id] = active_missiles
            else:
                # No active missiles for this target - remove entry
                del self.missiles_fired_at_targets[target_id]

    def _record_missile_launch(self, target, missile_id: str) -> None:
        """Record that we launched a missile at a target."""
        if not target:
            return

        target_id = getattr(target, "id", str(target))
        aircraft_id = getattr(self.aircraft, "id", "unknown")

        if target_id not in self.missiles_fired_at_targets:
            self.missiles_fired_at_targets[target_id] = []

        self.missiles_fired_at_targets[target_id].append(missile_id)
        print(
            f"Aircraft {aircraft_id}: Recorded missile {missile_id} launch at target {target_id} - now {len(self.missiles_fired_at_targets[target_id])} missiles tracking this target"
        )

    def _record_missile_launch_from_context(self, context: dict[str, Any]) -> None:
        """Record missile launch based on context information."""
        target = context.get("target")
        if not target:
            return

        # Generate a unique missile ID based on aircraft, target, and time
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        target_id = getattr(target, "id", str(target))
        import time

        timestamp = int(time.time() * 1000)  # Millisecond timestamp
        missile_id = f"missile_{aircraft_id}_to_{target_id}_{timestamp}"

        self._record_missile_launch(target, missile_id)

    # === ADVANCED BVR CONDITIONS ===

    def _under_radar_threat(self, context: dict[str, Any]) -> bool:
        """Check if under immediate radar threat requiring beam/notch maneuver."""
        threats = context.get("threats", [])
        for threat in threats:
            if (
                threat.threat_level == "high"
                and hasattr(threat, "distance_m")
                and threat.distance_m < 40000  # Within 40km
                and hasattr(threat, "closure_rate")
                and threat.closure_rate > 200
            ):  # Approaching fast
                return True
        return False

    def _near_f_pole(self, context: dict[str, Any]) -> bool:
        """Check if near F-Pole (missile going active) requiring defensive action."""
        threats = context.get("threats", [])
        for threat in threats:
            if (
                threat.threat_type == "missile"
                and hasattr(threat, "distance_m")
                and threat.distance_m < 20000
            ):  # Within 20km missile range
                return True
        return False

    def _needs_target_selection(self, context: dict[str, Any]) -> bool:
        """Check if we need to select optimal target from multiple options."""
        available_targets = context.get("available_targets", [])
        current_target = context.get("target")

        # Need target selection if we have multiple targets and no current target
        # or if current target has max missiles already
        if len(available_targets) > 1:
            if not current_target:
                return True
            # Check if current target has max missiles
            if not self._can_fire_at_target(current_target, context):
                return True
        return False

    def _should_advanced_crank(self, context: dict[str, Any]) -> bool:
        """Check if we should perform energy-optimized cranking."""
        target = context.get("target")
        if not target:
            return False

        target_range = context.get("target_range_m", float("inf"))
        closure_rate = context.get("closure_rate_mps", 0)

        # Advanced crank if in optimal range with high closure
        return 30000 <= target_range <= 80000 and closure_rate > 250

    def _optimal_attack_geometry(self, context: dict[str, Any]) -> bool:
        """Check if we have optimal attack geometry (aspect angle, energy state)."""
        target = context.get("target")
        if not target:
            return False

        aspect_angle = context.get("aspect_angle", 0)
        energy_advantage = context.get("energy_advantage", 0)
        target_range = context.get("target_range_m", float("inf"))

        # Optimal geometry: good aspect angle, energy advantage, and range
        return (
            45 < abs(aspect_angle) < 135  # Beam or rear aspect
            and energy_advantage > 0  # Energy advantage
            and 20000 <= target_range <= 60000
        )  # Optimal engagement range

    def _in_advanced_launch_range(self, context: dict[str, Any]) -> bool:
        """Advanced launch range check including energy state and geometry."""
        # Use existing launch range check but add energy considerations
        basic_in_range = self._in_launch_range(context)
        if not basic_in_range:
            return False

        # Additional advanced criteria
        energy_advantage = context.get("energy_advantage", 0)
        aspect_angle = context.get("aspect_angle", 0)

        # Prefer launching with energy advantage or good aspect
        return energy_advantage >= 0 or abs(aspect_angle) > 30

    def _should_advanced_launch_leave(self, context: dict[str, Any]) -> bool:
        """Decide if we should use advanced launch-and-leave tactics."""
        threats = context.get("threats", [])
        energy_advantage = context.get("energy_advantage", 0)

        # Launch and leave if outnumbered or at energy disadvantage
        threat_count = len([t for t in threats if t.threat_level == "high"])
        return (
            threat_count >= 2 or energy_advantage < -500
        )  # Multiple threats or energy disadvantage

    def _can_drag_launch(self, context: dict[str, Any]) -> bool:
        """Check if we can perform drag launch (extended range with energy advantage)."""
        target = context.get("target")
        if not target:
            return False

        target_range = context.get("target_range_m", float("inf"))
        energy_advantage = context.get("energy_advantage", 0)
        altitude = self.aircraft.position.alt

        # Can drag launch if at long range with significant energy advantage
        return (
            target_range > 70000  # Long range
            and energy_advantage > 1000  # Good energy advantage
            and altitude > 10000
        )  # High altitude

    def _should_advanced_launch_decide(self, context: dict[str, Any]) -> bool:
        """Decide if we should use advanced launch-and-decide tactics."""
        energy_advantage = context.get("energy_advantage", 0)
        target_range = context.get("target_range_m", float("inf"))

        # Launch and decide if we have energy advantage and are in good range
        return energy_advantage > 0 and 40000 <= target_range <= 70000

    def _needs_energy_advantage(self, context: dict[str, Any]) -> bool:
        """Check if we need to gain energy advantage."""
        energy_advantage = context.get("energy_advantage", 0)
        altitude = self.aircraft.position.alt
        speed = self.aircraft.speed

        # Need energy if at disadvantage or low altitude/speed
        return energy_advantage < -200 or altitude < 8000 or speed < 250

    # === ADVANCED BVR ACTIONS ===

    def _execute_beam_maneuver(self, context: dict[str, Any]) -> NodeStatus:
        """Execute defensive beam maneuver against radar threat."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: EXECUTING DEFENSIVE BEAM MANEUVER")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # Use BVR tactics beam maneuver
        if hasattr(self.tactical_controller, "execute_beam_maneuver"):
            return self.tactical_controller.execute_beam_maneuver(context)
        else:
            # Fallback beam maneuver - fly perpendicular to primary threat
            threats = context.get("threats", [])
            if threats:
                primary_threat = threats[0]  # Highest priority threat
                if hasattr(primary_threat, "bearing"):
                    beam_heading = (primary_threat.bearing + 90) % 360
                    context["desired_yaw"] = beam_heading
                    context["desired_pitch"] = 0.0
                    context["desired_throttle"] = 0.9
                    print(f"  Beam heading: {beam_heading:.0f}°")
                    return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

    def _execute_f_pole_defense(self, context: dict[str, Any]) -> NodeStatus:
        """Execute F-Pole defensive maneuvers (missile going active)."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: EXECUTING F-POLE DEFENSE")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # F-Pole defense: aggressive defensive maneuver
        context["desired_pitch"] = -10.0  # Dive for speed
        context["desired_throttle"] = 1.0  # Full power

        # Turn perpendicular to missile threat
        threats = context.get("threats", [])
        for threat in threats:
            if threat.threat_type == "missile":
                threat_bearing = getattr(threat, "bearing", 0)
                # Notch 90 degrees from missile
                notch_heading = (threat_bearing + 90) % 360
                context["desired_yaw"] = notch_heading
                print(f"  F-Pole notch heading: {notch_heading:.0f}°")
                return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

    def _execute_priority_targeting(self, context: dict[str, Any]) -> NodeStatus:
        """Execute advanced target prioritization and selection."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        available_targets = context.get("available_targets", [])

        if not available_targets:
            return NodeStatus.FAILURE

        # Use TrackPrioritySystem for advanced target selection
        if hasattr(self.tactical_controller, "track_priority"):
            best_target = self.tactical_controller.track_priority.select_best_target(
                available_targets
            )
            if best_target:
                context["target"] = best_target
                target_id = getattr(best_target, "id", "unknown")
                print(f"Aircraft {aircraft_id}: PRIORITY TARGET SELECTED - Target {target_id}")
                return NodeStatus.SUCCESS

        # Fallback: select target we can fire at
        for target in available_targets:
            if self._can_fire_at_target(target, context):
                context["target"] = target
                target_id = getattr(target, "id", "unknown")
                print(f"Aircraft {aircraft_id}: FALLBACK TARGET SELECTED - Target {target_id}")
                return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

    def _execute_advanced_target_acquisition(self, context: dict[str, Any]) -> NodeStatus:
        """Execute geometry-aware target acquisition."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ADVANCED TARGET ACQUISITION")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading
            return NodeStatus.RUNNING

        # Use enhanced tactical controller if available
        if hasattr(self.tactical_controller, "execute_enhanced_approach"):
            return self.tactical_controller.execute_enhanced_approach(context)
        else:
            # Fallback to basic target acquisition
            return self._execute_target_acquisition(context)

    def _execute_energy_optimized_crank(self, context: dict[str, Any]) -> NodeStatus:
        """Execute energy-optimized cranking maneuver."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ENERGY-OPTIMIZED CRANK")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # Use enhanced energy management for cranking
        if hasattr(self.tactical_controller, "execute_enhanced_approach"):
            context["crank_maneuver"] = True  # Signal for energy-optimized crank
            return self.tactical_controller.execute_enhanced_approach(context)
        else:
            # Fallback to basic crank
            return self._execute_crank(context)

    def _execute_multi_aspect_attack(self, context: dict[str, Any]) -> NodeStatus:
        """Execute multi-aspect attack with optimal geometry."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: MULTI-ASPECT ATTACK")

        target = context.get("target")
        if not target:
            return NodeStatus.FAILURE

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # Position for optimal attack angle (beam or rear aspect)
        aspect_angle = context.get("aspect_angle", 0)
        target_bearing = context.get("target_bearing_deg", 0)

        if abs(aspect_angle) < 45:  # Too close to head-on
            # Move to beam aspect
            offset_angle = 60 if aspect_angle >= 0 else -60
            desired_heading = (target_bearing + offset_angle) % 360
        else:
            # Good aspect - maintain and approach
            desired_heading = target_bearing

        context["desired_yaw"] = desired_heading
        context["desired_pitch"] = 2.0  # Slight climb for energy
        context["desired_throttle"] = 0.85

        print(f"  Multi-aspect approach: {desired_heading:.0f}°, aspect: {aspect_angle:.0f}°")
        return NodeStatus.SUCCESS

    def _execute_advanced_launch_leave(self, context: dict[str, Any]) -> NodeStatus:
        """Execute advanced launch-and-leave with energy optimization."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ADVANCED LAUNCH-AND-LEAVE")

        # Use energy-optimized launch
        context["energy_optimized_launch"] = True
        return self._execute_launch_leave(context)

    def _execute_drag_launch(self, context: dict[str, Any]) -> NodeStatus:
        """Execute drag launch (extended range with energy advantage)."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        target_range = context.get("target_range_m", 0) / 1000
        print(f"Aircraft {aircraft_id}: DRAG LAUNCH at {target_range:.1f}km")

        # Drag launch: maintain energy while launching at extended range
        self._ensure_target_pointing(context)

        context["drag_launch"] = True  # Signal for extended range shot
        result = self.tactical_controller.execute_launch_and_decide(context)

        if result == NodeStatus.SUCCESS and context.get("fire_missile", False):
            self._record_missile_launch_from_context(context)

        return result

    def _execute_advanced_launch_decide(self, context: dict[str, Any]) -> NodeStatus:
        """Execute advanced launch-and-decide with post-launch tactics."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ADVANCED LAUNCH-AND-DECIDE")

        # Use energy-aware launch and decide
        context["advanced_post_launch"] = True
        return self._execute_launch_decide(context)

    def _execute_energy_positioning(self, context: dict[str, Any]) -> NodeStatus:
        """Execute energy positioning (gain altitude/energy advantage)."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ENERGY POSITIONING")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # Climb for energy advantage
        context["desired_pitch"] = 5.0  # Moderate climb
        context["desired_throttle"] = 0.9  # High power

        # Maintain current heading unless boundary override
        if boundary_heading is None:
            context["desired_yaw"] = self.aircraft.yaw_deg

        return NodeStatus.SUCCESS

    def _execute_energy_aware_approach(self, context: dict[str, Any]) -> NodeStatus:
        """Execute energy-aware approach to target."""
        aircraft_id = getattr(self.aircraft, "id", "unknown")
        print(f"Aircraft {aircraft_id}: ENERGY-AWARE APPROACH")

        # Check boundary limits first
        boundary_heading = self._check_boundary_limits(context)
        if boundary_heading is not None:
            context["boundary_avoidance_heading"] = boundary_heading

        # Use enhanced tactical controller for energy-aware approach
        if hasattr(self.tactical_controller, "execute_enhanced_approach"):
            return self.tactical_controller.execute_enhanced_approach(context)
        else:
            return self._execute_approach(context)
