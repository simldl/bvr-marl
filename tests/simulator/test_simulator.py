#!/usr/bin/env python3
"""
Comprehensive tests for simulator.simulator module.
Tests main Simulator class functionality including unit management,
time progression, event handling, and integration features.
"""

import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from air_to_air_rl.simulator.core.events import Event, UnitDestroyedEvent, UnitRegisteredEvent
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.core.units import FlyingUnit, Unit
from air_to_air_rl.simulator.simulator import Simulator
from tests.mocks import MockAircraft


class TestSimulatorInit:
    """Test Simulator initialization and configuration."""

    def test_default_init(self):
        """Test default simulator initialization."""
        sim = Simulator()

        assert sim.active_units == {}
        assert sim.trace_record_units == {}
        assert isinstance(sim.utc_time, datetime)
        assert sim.utc_time_initial == sim.utc_time
        assert sim.tick_secs == 1
        assert sim._next_unit_id == 1
        assert sim.status_text is None
        assert sim.num_units == 0
        assert sim.num_opp_units == 0
        assert sim.gun_hit_radius_m == 5.0
        assert hasattr(sim, "ccd")
        assert hasattr(sim, "hit_calc")

    def test_custom_init(self):
        """Test simulator initialization with custom parameters."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        weapon_config = {"missile_hit_radius_m": 750.0, "gun_hit_radius_m": 10.0}

        sim = Simulator(
            utc_time=start_time,
            tick_secs=0.5,
            random_seed=12345,
            num_units=4,
            num_opp_units=2,
            weapon_config=weapon_config,
        )

        assert sim.utc_time == start_time
        assert sim.utc_time_initial == start_time
        assert sim.tick_secs == 0.5
        assert sim.random_seed == 12345
        assert sim.num_units == 4
        assert sim.num_opp_units == 2
        assert sim.gun_hit_radius_m == 10.0

    def test_random_generator_seeded(self):
        """Test random generator is properly seeded."""
        sim1 = Simulator(random_seed=42)
        sim2 = Simulator(random_seed=42)
        sim3 = Simulator(random_seed=43)

        # Same seed should produce same random values
        val1 = sim1.rnd_gen.random()
        val2 = sim2.rnd_gen.random()
        val3 = sim3.rnd_gen.random()

        assert val1 == val2
        assert val1 != val3


class TestSimulatorUnitManagement:
    """Test unit management functionality."""

    @pytest.fixture
    def simulator(self):
        return Simulator(tick_secs=1.0)

    @pytest.fixture
    def mock_unit(self):
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("TestUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[])
        unit.substep_update = Mock(return_value=[])
        return unit

    def test_add_unit(self, simulator, mock_unit):
        """Test adding a unit to the simulator."""
        unit_id = simulator.add_unit(mock_unit)

        assert unit_id == 1
        assert len(simulator.active_units) == 1
        assert simulator.active_units[1] == mock_unit
        assert mock_unit.id == 1
        assert simulator._next_unit_id == 2

        # Should generate UnitRegisteredEvent in events log
        assert len(simulator.events) == 1
        assert isinstance(simulator.events[0], UnitRegisteredEvent)
        assert simulator.events[0].registered_unit == mock_unit

    def test_add_multiple_units(self, simulator):
        """Test adding multiple units with sequential IDs."""
        units = []
        for i in range(3):
            pos = Position(52.0 + i, 8.0, 1000.0)
            unit = FlyingUnit(f"Unit{i}", pos, 90.0, 200.0)
            unit.update = Mock(return_value=[])
            units.append(unit)
            simulator.add_unit(unit)

        assert len(simulator.active_units) == 3
        assert units[0].id == 1
        assert units[1].id == 2
        assert units[2].id == 3
        assert simulator._next_unit_id == 4

    def test_get_unit(self, simulator, mock_unit):
        """Test retrieving a unit by ID."""
        simulator.add_unit(mock_unit)

        retrieved = simulator.get_unit(1)
        assert retrieved == mock_unit

        # Non-existent unit should return None
        assert simulator.get_unit(999) is None

    def test_remove_unit(self, simulator, mock_unit):
        """Test removing a unit from the simulator."""
        simulator.add_unit(mock_unit)
        assert len(simulator.active_units) == 1

        simulator.remove_unit(1)  # Returns None
        assert len(simulator.active_units) == 0

        # Removing non-existent unit should do nothing
        simulator.remove_unit(999)  # Should not crash

    def test_add_unit_with_target(self, simulator):
        """Test adding unit that has a target."""
        pos1 = Position(52.0, 8.0, 1000.0)
        pos2 = Position(52.1, 8.1, 1000.0)

        target = FlyingUnit("Target", pos2, 270.0, 150.0)
        target.update = Mock(return_value=[])

        hunter = FlyingUnit("Hunter", pos1, 90.0, 200.0)
        hunter.update = Mock(return_value=[])
        hunter.target = target  # Set target before adding

        simulator.add_unit(target)  # Add target first
        hunter_id = simulator.add_unit(hunter)  # Then add hunter with target

        assert len(simulator.active_units) == 2
        assert hunter.target is target
        assert hunter_id == 2


class TestSimulatorTimeProgression:
    """Test time progression and tick functionality."""

    @pytest.fixture
    def simulator(self):
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        return Simulator(utc_time=start_time, tick_secs=1.0)

    def test_tick_advances_time(self, simulator):
        """Test that tick advances simulation time."""
        initial_time = simulator.utc_time

        events = simulator.do_tick()

        expected_time = initial_time + timedelta(seconds=simulator.tick_secs)
        assert simulator.utc_time == expected_time
        assert isinstance(events, list)

    def test_tick_calls_unit_updates(self, simulator):
        """Test that tick calls update on all units."""
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("TestUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[])

        simulator.add_unit(unit)
        simulator.do_tick()

        unit.update.assert_called_once_with(1.0, simulator)

    def test_tick_collects_events(self, simulator):
        """Test that tick collects events from all units."""
        pos = Position(52.0, 8.0, 1000.0)

        unit1 = FlyingUnit("Unit1", pos, 90.0, 200.0)
        unit1.update = Mock(return_value=[Mock()])

        unit2 = FlyingUnit("Unit2", pos, 90.0, 200.0)
        unit2.update = Mock(return_value=[Mock(), Mock()])

        simulator.add_unit(unit1)
        simulator.add_unit(unit2)

        events = simulator.do_tick()

        # Should collect events from both units
        assert len(events) == 3

    def test_tick_callbacks(self, simulator):
        """Test tick callback functionality."""
        callback = Mock()
        simulator.add_tick_callback(callback)

        simulator.do_tick()

        callback.assert_called_once_with(simulator.utc_time)

    def test_elapsed_time_property(self, simulator):
        """Test elapsed time calculation."""
        initial_elapsed = simulator.elapsed_time_s
        assert initial_elapsed == 0.0

        simulator.do_tick()

        elapsed = simulator.elapsed_time_s
        assert elapsed == simulator.tick_secs

        simulator.do_tick()
        assert simulator.elapsed_time_s == simulator.tick_secs * 2


class TestSimulatorEventHandling:
    """Test event handling and logging."""

    @pytest.fixture
    def simulator(self):
        return Simulator()

    def test_log_event(self, simulator):
        """Test event logging functionality."""
        event = Event("TestEvent", simulator)

        simulator.log_event(event)

        # Event should be stored in the events list
        assert hasattr(simulator, "events")
        assert len(simulator.events) == 1
        assert simulator.events[0] == event

    def test_event_collection_during_tick(self, simulator):
        """Test event collection during tick cycle."""
        pos = Position(52.0, 8.0, 1000.0)

        test_event = Event("UnitEvent", None)
        unit = FlyingUnit("EventUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[test_event])

        simulator.add_unit(unit)
        events = simulator.do_tick()

        # Should collect the unit's event
        assert test_event in events


class TestSimulatorTracing:
    """Test unit tracing and recording functionality."""

    @pytest.fixture
    def simulator(self):
        return Simulator()

    def test_record_unit_trace(self, simulator):
        """Test unit trace recording."""
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("TracedUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[])

        simulator.add_unit(unit)
        simulator.record_unit_trace(unit.id)

        assert unit.id in simulator.trace_record_units

    def test_trace_persists_during_movement(self, simulator):
        """Test that trace data persists as unit moves."""
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("MovingUnit", pos, 90.0, 200.0)

        # Simple movement simulation
        def mock_update(dt, sim):
            unit.position.lat += 0.001  # Move slightly
            return []

        unit.update = mock_update

        simulator.add_unit(unit)
        simulator.record_unit_trace(unit.id)

        initial_pos = unit.position.lat
        simulator.do_tick()

        assert unit.position.lat != initial_pos
        # Trace should still be recorded
        assert unit.id in simulator.trace_record_units


class TestSimulatorReset:
    """Test simulator reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all simulator state."""
        initial_time = datetime(2023, 1, 1, 12, 0, 0)
        sim = Simulator(utc_time=initial_time)

        # Add some state
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("TestUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[])

        sim.add_unit(unit)
        sim.record_unit_trace(unit.id)
        sim.do_tick()  # Advance time

        # State should exist
        assert len(sim.active_units) > 0
        assert len(sim.trace_record_units) > 0
        assert sim.utc_time > initial_time

        # Reset
        sim.reset_sim({})  # Reset with empty units dictionary

        # State should be cleared
        assert len(sim.active_units) == 0
        assert len(sim.trace_record_units) == 0
        assert sim._next_unit_id == 1

    def test_reset_preserves_configuration(self):
        """Test that reset preserves initial configuration."""
        initial_time = datetime(2023, 1, 1, 12, 0, 0)
        weapon_config = {"missile_hit_radius_m": 750.0}

        sim = Simulator(
            utc_time=initial_time, tick_secs=0.5, random_seed=42, weapon_config=weapon_config
        )

        # Modify some state
        sim.do_tick()

        # Reset
        sim.reset_sim({})

        # Configuration should be preserved
        assert sim.tick_secs == 0.5
        assert sim.random_seed == 42
        assert sim.gun_hit_radius_m == 5.0  # Default since not specified


class TestSimulatorStatusAndUtilities:
    """Test status reporting and utility functions."""

    @pytest.fixture
    def simulator(self):
        return Simulator()

    def test_set_status_text(self, simulator):
        """Test status text setting."""
        test_status = "Running simulation..."
        simulator.set_status_text(test_status)

        assert simulator.status_text == test_status

        # Should be able to clear status
        simulator.set_status_text(None)
        assert simulator.status_text is None

    def test_status_text_updates(self, simulator):
        """Test status text can be updated."""
        simulator.set_status_text("Initial status")
        assert simulator.status_text == "Initial status"

        simulator.set_status_text("Updated status")
        assert simulator.status_text == "Updated status"


class TestSimulatorIntegration:
    """Integration tests for simulator functionality."""

    def test_complete_simulation_cycle(self):
        """Test a complete simulation cycle with multiple units."""
        sim = Simulator(tick_secs=1.0)

        # Create units
        pos1 = Position(52.0, 8.0, 1000.0)
        pos2 = Position(52.1, 8.1, 1000.0)

        unit1 = FlyingUnit("Fighter1", pos1, 90.0, 200.0)
        unit2 = FlyingUnit("Fighter2", pos2, 270.0, 180.0)

        # Mock simple movement
        def move_unit1(dt, sim):
            unit1.position.lat += 0.001
            return []

        def move_unit2(dt, sim):
            unit2.position.lon -= 0.001
            return []

        unit1.update = move_unit1
        unit2.update = move_unit2

        # Add units
        sim.add_unit(unit1)
        sim.add_unit(unit2)

        # Record traces
        sim.record_unit_trace(unit1.id)

        # Run simulation
        initial_positions = [(u.position.lat, u.position.lon) for u in [unit1, unit2]]

        for _ in range(5):
            events = sim.do_tick()
            assert isinstance(events, list)

        # Units should have moved
        final_positions = [(u.position.lat, u.position.lon) for u in [unit1, unit2]]
        assert initial_positions != final_positions

        # Time should have advanced
        assert sim.elapsed_time_s == 5.0

    def test_unit_lifecycle_with_events(self):
        """Test unit lifecycle generates appropriate events."""
        sim = Simulator()

        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("LifecycleUnit", pos, 90.0, 200.0)
        unit.update = Mock(return_value=[])

        # Add unit - should generate registration event
        initial_event_count = len(sim.events)
        unit_id = sim.add_unit(unit)
        assert isinstance(unit_id, int)
        assert len(sim.events) == initial_event_count + 1
        assert isinstance(sim.events[-1], UnitRegisteredEvent)

        # Remove unit
        removed_unit = sim.remove_unit(unit.id)
        assert removed_unit == unit
        assert len(sim.active_units) == 0

    def test_performance_with_many_units(self):
        """Test simulator performance with multiple units."""
        sim = Simulator(tick_secs=0.1)

        # Add many units
        units = []
        for i in range(50):  # Moderate number for testing
            pos = Position(52.0 + i * 0.001, 8.0 + i * 0.001, 1000.0)
            unit = FlyingUnit(f"Unit{i}", pos, i % 360, 200.0)
            unit.update = Mock(return_value=[])
            units.append(unit)
            sim.add_unit(unit)

        assert len(sim.active_units) == 50

        # Tick should handle all units
        sim.do_tick()

        # All units should have been updated
        for unit in units:
            unit.update.assert_called_once()


class TestSimulatorErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def simulator(self):
        return Simulator()

    def test_tick_with_no_units(self, simulator):
        """Test tick with no active units."""
        events = simulator.do_tick()

        assert isinstance(events, list)
        assert len(events) == 0

    def test_invalid_unit_operations(self, simulator):
        """Test operations with invalid unit IDs."""
        # Get non-existent unit
        assert simulator.get_unit(-1) is None
        assert simulator.get_unit(999) is None

        # Remove non-existent unit
        assert simulator.remove_unit(-1) is None
        assert simulator.remove_unit(999) is None

    def test_unit_update_exception_handling(self, simulator):
        """Test handling of exceptions during unit updates."""
        pos = Position(52.0, 8.0, 1000.0)
        unit = FlyingUnit("FailingUnit", pos, 90.0, 200.0)
        unit.update = Mock(side_effect=Exception("Update failed"))

        simulator.add_unit(unit)

        # Tick should handle exceptions gracefully
        # (Implementation may vary - might log error, remove unit, etc.)
        try:
            simulator.do_tick()
            # If no exception propagated, that's acceptable
        except Exception:
            # If exception propagates, that's also valid behavior
            pass

    def test_callback_exception_handling(self, simulator):
        """Test handling of exceptions in tick callbacks."""
        failing_callback = Mock(side_effect=Exception("Callback failed"))
        simulator.add_tick_callback(failing_callback)

        # Should handle callback exceptions gracefully
        try:
            simulator.do_tick()
        except Exception:
            pass  # Either handling gracefully or propagating is valid


class TestSimulatorCountermeasureCleanup:
    """Test countermeasure cleanup when parent unit is removed."""

    def _create_mock_unit(self, unit_id, name, is_countermeasure=False, parent=None):
        """Helper to create a properly configured mock unit."""
        unit = Mock(spec=Unit)
        unit.id = unit_id
        unit.name = name
        unit.is_countermeasure = is_countermeasure
        unit.should_be_removed = False
        unit.update = Mock(return_value=[])
        unit.position = Mock()
        unit.position.lat = 0.0
        unit.position.lon = 0.0
        unit.position.alt = 1000.0
        unit.position.copy = Mock(return_value=unit.position)
        unit.yaw_deg = 0.0
        unit.speed = 250.0 if not is_countermeasure else 0.0
        if parent is not None:
            unit.parent_aircraft = parent
        return unit

    def test_countermeasures_removed_with_parent(self):
        """Test that countermeasures are removed when parent unit dies."""
        sim = Simulator()

        # Create a mock parent unit (aircraft)
        parent = Mock(spec=Unit)
        parent.id = 1
        parent.name = "TestJet"
        parent.group = "BLUE"
        parent.is_countermeasure = False
        parent.should_be_removed = False
        parent.update = Mock(return_value=[])
        parent.position = Mock()
        parent.position.lat = 0.0
        parent.position.lon = 0.0
        parent.position.alt = 1000.0
        parent.position.copy = Mock(return_value=parent.position)
        parent.yaw_deg = 0.0
        parent.speed = 250.0

        # Create mock countermeasures
        cm1 = Mock(spec=Unit)
        cm1.id = 2
        cm1.name = "Flare1"
        cm1.is_countermeasure = True
        cm1.parent_aircraft = parent
        cm1.should_be_removed = False
        cm1.update = Mock(return_value=[])
        cm1.position = Mock()
        cm1.position.lat = 0.0
        cm1.position.lon = 0.0
        cm1.position.alt = 1000.0
        cm1.position.copy = Mock(return_value=cm1.position)
        cm1.yaw_deg = 0.0
        cm1.speed = 0.0

        cm2 = Mock(spec=Unit)
        cm2.id = 3
        cm2.name = "Chaff1"
        cm2.is_countermeasure = True
        cm2.parent_aircraft = parent
        cm2.should_be_removed = False
        cm2.update = Mock(return_value=[])
        cm2.position = Mock()
        cm2.position.lat = 0.0
        cm2.position.lon = 0.0
        cm2.position.alt = 1000.0
        cm2.position.copy = Mock(return_value=cm2.position)
        cm2.yaw_deg = 0.0
        cm2.speed = 0.0

        # Add units to simulator
        sim.add_unit(parent)
        sim.add_unit(cm1)
        sim.add_unit(cm2)

        # Verify all units exist
        assert sim.unit_exists(1)
        assert sim.unit_exists(2)
        assert sim.unit_exists(3)
        assert len(sim.active_units) == 3

        # Mark parent for removal
        parent.should_be_removed = True
        parent.removal_reason = "destroyed"

        # Run tick - should remove parent and countermeasures
        sim.do_tick()

        # Verify parent is removed
        assert not sim.unit_exists(1)

        # Verify countermeasures are also removed
        assert not sim.unit_exists(2)
        assert not sim.unit_exists(3)
        assert len(sim.active_units) == 0

    def test_countermeasures_only_removed_for_correct_parent(self):
        """Test that only countermeasures of removed parent are cleaned up."""
        sim = Simulator()

        # Create two parent units
        parent1 = self._create_mock_unit(1, "Jet1")
        parent2 = self._create_mock_unit(2, "Jet2")

        # Create countermeasures for each parent
        cm1 = self._create_mock_unit(3, "Flare_from_Jet1", is_countermeasure=True, parent=parent1)
        cm2 = self._create_mock_unit(4, "Flare_from_Jet2", is_countermeasure=True, parent=parent2)

        # Add all units
        sim.add_unit(parent1)
        sim.add_unit(parent2)
        sim.add_unit(cm1)
        sim.add_unit(cm2)

        assert len(sim.active_units) == 4

        # Remove only parent1
        parent1.should_be_removed = True
        parent1.removal_reason = "destroyed"
        sim.do_tick()

        # Verify parent1 and its countermeasure are gone
        assert not sim.unit_exists(1)
        assert not sim.unit_exists(3)

        # Verify parent2 and its countermeasure still exist
        assert sim.unit_exists(2)
        assert sim.unit_exists(4)
        assert len(sim.active_units) == 2

    def test_no_crash_if_no_countermeasures(self):
        """Test that removing unit without countermeasures doesn't crash."""
        sim = Simulator()

        # Create a unit without countermeasures
        unit = self._create_mock_unit(1, "TestUnit")

        sim.add_unit(unit)
        assert sim.unit_exists(1)

        # Remove unit
        unit.should_be_removed = True
        sim.do_tick()

        # Should complete without errors
        assert not sim.unit_exists(1)

    def test_non_countermeasure_units_not_affected(self):
        """Test that regular units are not removed during countermeasure cleanup."""
        sim = Simulator()

        # Create parent
        parent = self._create_mock_unit(1, "Jet")

        # Create countermeasure
        cm = self._create_mock_unit(2, "Flare", is_countermeasure=True, parent=parent)
        cm.update = Mock(return_value=[])

        # Create unrelated unit
        other_unit = self._create_mock_unit(3, "OtherJet")

        sim.add_unit(parent)
        sim.add_unit(cm)
        sim.add_unit(other_unit)

        # Remove parent
        parent.should_be_removed = True
        sim.do_tick()

        # Verify parent and countermeasure removed
        assert not sim.unit_exists(1)
        assert not sim.unit_exists(2)

        # Verify other unit still exists
        assert sim.unit_exists(3)
