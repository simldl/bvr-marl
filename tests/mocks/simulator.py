#!/usr/bin/env python3
"""
Mock simulator classes for testing.
Provides lightweight mock implementations of simulation components.
"""

import time
from typing import Any, Optional

import numpy as np


class MockSimulator:
    """Mock simulator for testing."""

    def __init__(self):
        self.active_units = {}  # id -> unit mapping
        self.utc_time = 0.0
        self.elapsed_time_s = 0.0
        self.sim_time = 0.0
        self.tick_count = 0
        self.dt = 0.1  # Default time step

        # Event tracking
        self.events = []
        self.added_units = []
        self.removed_units = []

        # Performance tracking
        self.start_time = time.time()
        self.last_update_time = 0.0

    def add_unit(self, unit):
        """Add a unit to the simulation."""
        if not hasattr(unit, "id") or unit.id is None:
            # Generate unique ID based on total units added (not just current active)
            unit.id = f"unit_{len(self.added_units)}"

        self.active_units[unit.id] = unit
        self.added_units.append(unit)

        # Fire event
        self.events.append({"type": "unit_added", "unit": unit, "time": self.utc_time})

    def remove_unit(self, unit_id):
        """Remove a unit from the simulation."""
        if unit_id in self.active_units:
            unit = self.active_units.pop(unit_id)
            self.removed_units.append(unit)

            # Fire event
            self.events.append({"type": "unit_removed", "unit": unit, "time": self.utc_time})

    def get_unit(self, unit_id):
        """Get a unit by ID."""
        return self.active_units.get(unit_id)

    def get_units_by_group(self, group):
        """Get all units in a specific group."""
        return [
            unit
            for unit in self.active_units.values()
            if hasattr(unit, "group") and unit.group == group
        ]

    def get_units_by_type(self, unit_type):
        """Get all units of a specific type."""
        return [
            unit
            for unit in self.active_units.values()
            if hasattr(unit, "type") and unit.type == unit_type
        ]

    def step(self, dt=None):
        """Perform one simulation step."""
        if dt is None:
            dt = self.dt

        self.tick_count += 1
        self.utc_time += dt
        self.elapsed_time_s += dt
        self.sim_time += dt

        # Update all units
        units_to_remove = []
        for unit_id, unit in self.active_units.items():
            try:
                if hasattr(unit, "update"):
                    events = unit.update(dt, self)
                    if events:
                        self.events.extend(events)

                # Check if unit should be removed
                if hasattr(unit, "is_active") and not unit.is_active:
                    units_to_remove.append(unit_id)

            except Exception as e:
                # Log error but continue simulation
                self.events.append(
                    {
                        "type": "update_error",
                        "unit_id": unit_id,
                        "error": str(e),
                        "time": self.utc_time,
                    }
                )

        # Remove inactive units
        for unit_id in units_to_remove:
            self.remove_unit(unit_id)

        self.last_update_time = self.utc_time

    def run(self, duration, dt=None):
        """Run simulation for specified duration."""
        if dt is None:
            dt = self.dt

        end_time = self.utc_time + duration

        while self.utc_time < end_time:
            remaining_time = end_time - self.utc_time
            step_dt = min(dt, remaining_time)
            self.step(step_dt)

    def reset(self):
        """Reset simulation state."""
        self.active_units.clear()
        self.utc_time = 0.0
        self.elapsed_time_s = 0.0
        self.sim_time = 0.0
        self.tick_count = 0
        self.events.clear()
        self.added_units.clear()
        self.removed_units.clear()
        self.start_time = time.time()

    def get_statistics(self):
        """Get simulation statistics."""
        current_time = time.time()
        wall_time = current_time - self.start_time

        return {
            "sim_time": self.sim_time,
            "wall_time": wall_time,
            "time_ratio": self.sim_time / max(wall_time, 1e-6),
            "tick_count": self.tick_count,
            "active_units": len(self.active_units),
            "total_events": len(self.events),
            "units_added": len(self.added_units),
            "units_removed": len(self.removed_units),
        }

    def get_events(self, event_type=None, since_time=None):
        """Get events, optionally filtered by type and time."""
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.get("type") == event_type]

        if since_time is not None:
            filtered_events = [e for e in filtered_events if e.get("time", 0) >= since_time]

        return filtered_events

    def clear_events(self):
        """Clear event history."""
        self.events.clear()


class MockScenario:
    """Mock scenario for testing."""

    def __init__(self, name="Test Scenario"):
        self.name = name
        self.simulator = MockSimulator()
        self.initial_units = []
        self.objectives = []
        self.success_criteria = []

    def setup(self):
        """Setup the scenario."""
        # Add initial units to simulator
        for unit in self.initial_units:
            self.simulator.add_unit(unit)

    def add_unit(self, unit):
        """Add a unit to the scenario."""
        self.initial_units.append(unit)
        if self.simulator:
            self.simulator.add_unit(unit)

    def run(self, duration=60.0, dt=0.1):
        """Run the scenario."""
        self.simulator.run(duration, dt)
        return self.evaluate()

    def evaluate(self):
        """Evaluate scenario success."""
        # Mock evaluation
        results = {
            "success": True,
            "objectives_met": len(self.objectives),
            "final_time": self.simulator.utc_time,
            "units_remaining": len(self.simulator.active_units),
            "events": len(self.simulator.events),
        }

        return results


class MockEnvironment:
    """Mock environment for RL testing."""

    def __init__(self, scenario=None):
        self.scenario = scenario or MockScenario()
        self.state_space = None
        self.action_space = None
        self.current_state = None
        self.done = False
        self.reward = 0.0
        self.info = {}

    def reset(self):
        """Reset the environment."""
        self.scenario.simulator.reset()
        self.scenario.setup()
        self.done = False
        self.reward = 0.0
        self.current_state = self._get_state()
        return self.current_state

    def step(self, action):
        """Perform one environment step."""
        # Apply action to controlled units
        controlled_units = [
            unit
            for unit in self.scenario.simulator.active_units.values()
            if hasattr(unit, "apply_rl_action")
        ]

        for i, unit in enumerate(controlled_units):
            if i < len(action) if isinstance(action, list) else 1:
                unit_action = action[i] if isinstance(action, list) else action
                unit.apply_rl_action(unit_action, self.scenario.simulator)

        # Step simulation
        self.scenario.simulator.step()

        # Calculate reward and check termination
        self.current_state = self._get_state()
        self.reward = self._calculate_reward()
        self.done = self._check_termination()
        self.info = self._get_info()

        return self.current_state, self.reward, self.done, self.info

    def _get_state(self):
        """Get current environment state."""
        # Mock state - positions and velocities of all units
        state = []
        for unit in self.scenario.simulator.active_units.values():
            if hasattr(unit, "position"):
                state.extend([unit.position.lat, unit.position.lon, unit.position.alt])
            if hasattr(unit, "speed"):
                state.append(unit.speed)
            if hasattr(unit, "yaw_deg"):
                state.append(unit.yaw_deg)

        return np.array(state) if state else np.zeros(10)  # Default state size

    def _calculate_reward(self):
        """Calculate reward for current step."""
        # Mock reward calculation
        reward = 0.0

        # Reward for survival
        controlled_units = [
            unit
            for unit in self.scenario.simulator.active_units.values()
            if hasattr(unit, "group") and unit.group == "BLUE"
        ]
        reward += len(controlled_units) * 0.1

        # Penalty for enemies
        enemy_units = [
            unit
            for unit in self.scenario.simulator.active_units.values()
            if hasattr(unit, "group") and unit.group == "RED"
        ]
        reward -= len(enemy_units) * 0.05

        # Reward for successful hits
        hit_events = self.scenario.simulator.get_events("missile_hit")
        reward += len(hit_events) * 10.0

        return reward

    def _check_termination(self):
        """Check if episode should terminate."""
        # Terminate if no controlled units left
        controlled_units = [
            unit
            for unit in self.scenario.simulator.active_units.values()
            if hasattr(unit, "group") and unit.group == "BLUE"
        ]

        if not controlled_units:
            return True

        # Terminate if simulation time exceeded
        if self.scenario.simulator.utc_time > 300.0:  # 5 minutes
            return True

        return False

    def _get_info(self):
        """Get additional info dictionary."""
        return {
            "sim_time": self.scenario.simulator.utc_time,
            "active_units": len(self.scenario.simulator.active_units),
            "events": len(self.scenario.simulator.events),
        }


class MockBenchmark:
    """Mock benchmark for performance testing."""

    def __init__(self, name="Performance Test"):
        self.name = name
        self.metrics = {}
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the benchmark."""
        self.start_time = time.time()

    def stop(self):
        """Stop the benchmark."""
        self.end_time = time.time()
        self.metrics["duration"] = self.end_time - self.start_time

    def add_metric(self, name, value):
        """Add a metric."""
        self.metrics[name] = value

    def get_results(self):
        """Get benchmark results."""
        return {
            "name": self.name,
            "duration": self.metrics.get("duration", 0.0),
            "metrics": self.metrics,
        }


# Convenience functions for creating common test scenarios
def create_basic_engagement_scenario():
    """Create a basic air-to-air engagement scenario."""
    from tests.mocks.aircraft import MockAircraft, MockPosition
    from tests.mocks.missiles import MockMissile

    scenario = MockScenario("Basic Engagement")

    # Blue fighter
    blue_fighter = MockAircraft(
        name="Blue Fighter",
        position=MockPosition(0.0, 0.0, 10000.0),
        yaw_deg=0.0,
        speed=300.0,
        group="BLUE",
    )
    blue_fighter.missile_types = [MockMissile]
    scenario.add_unit(blue_fighter)

    # Red fighter
    red_fighter = MockAircraft(
        name="Red Fighter",
        position=MockPosition(0.1, 0.0, 10000.0),  # 10km north
        yaw_deg=180.0,  # Facing south
        speed=280.0,
        group="RED",
    )
    red_fighter.missile_types = [MockMissile]
    scenario.add_unit(red_fighter)

    return scenario


def create_bvr_scenario():
    """Create a Beyond Visual Range (BVR) scenario."""
    from tests.mocks.aircraft import MockAircraft, MockPosition
    from tests.mocks.missiles import MockAIM120

    scenario = MockScenario("BVR Engagement")

    # Blue fighter with long-range missiles
    blue_fighter = MockAircraft(
        name="Blue BVR",
        position=MockPosition(0.0, 0.0, 12000.0),
        yaw_deg=45.0,
        speed=400.0,
        group="BLUE",
    )
    blue_fighter.missile_types = [MockAIM120]
    blue_fighter.max_missiles = 8
    scenario.add_unit(blue_fighter)

    # Red fighter
    red_fighter = MockAircraft(
        name="Red BVR",
        position=MockPosition(0.5, 0.3, 11000.0),  # 50km away
        yaw_deg=225.0,  # Facing southwest
        speed=380.0,
        group="RED",
    )
    red_fighter.missile_types = [MockAIM120]
    scenario.add_unit(red_fighter)

    return scenario


def create_multi_threat_scenario():
    """Create a multi-threat scenario."""
    from tests.mocks.aircraft import MockAircraft, MockPosition
    from tests.mocks.missiles import MockMissile

    scenario = MockScenario("Multi-Threat")

    # Blue fighter (outnumbered)
    blue_fighter = MockAircraft(
        name="Blue Solo",
        position=MockPosition(0.0, 0.0, 8000.0),
        yaw_deg=0.0,
        speed=350.0,
        group="BLUE",
    )
    blue_fighter.missile_types = [MockMissile]
    blue_fighter.max_missiles = 6
    scenario.add_unit(blue_fighter)

    # Multiple red fighters
    red_positions = [
        MockPosition(0.2, 0.1, 8500.0),  # Northeast
        MockPosition(0.15, -0.1, 7500.0),  # Southeast
        MockPosition(-0.1, 0.05, 8200.0),  # Northwest
    ]

    for i, pos in enumerate(red_positions):
        red_fighter = MockAircraft(
            name=f"Red {i + 1}",
            position=pos,
            yaw_deg=180.0 + i * 30,  # Different headings
            speed=320.0,
            group="RED",
        )
        red_fighter.missile_types = [MockMissile]
        scenario.add_unit(red_fighter)

    return scenario
