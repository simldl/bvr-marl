#!/usr/bin/env python3
"""
Comprehensive tests for simulator.core.events module.
Tests all event classes including initialization, properties, and behavior.
"""

import pytest
from simulator.core.events import (
    Event, UnitRegisteredEvent, UnitDestroyedEvent, MissileEngagementEvent,
    CountermeasureEvent, UnitTraceEvent, CustomSimEvent
)


class MockUnit:
    """Mock unit for testing events."""
    def __init__(self, unit_id="mock_unit"):
        self.id = unit_id
        self.name = f"MockUnit_{unit_id}"


class TestEventBase:
    """Test base Event class functionality."""
    
    def test_event_init(self):
        """Test Event class initialization."""
        origin = MockUnit("origin1")
        event = Event("TestEvent", origin)
        
        assert event.name == "TestEvent"
        assert event.origin == origin
    
    def test_event_repr(self):
        """Test Event string representation."""
        origin = MockUnit("test_id")
        event = Event("MyEvent", origin)
        
        repr_str = repr(event)
        assert "Event" in repr_str
        assert "name=MyEvent" in repr_str
        assert "origin=test_id" in repr_str
    
    def test_event_repr_no_id(self):
        """Test Event repr when origin has no id attribute."""
        origin = object()  # No id attribute
        event = Event("NoIdEvent", origin)
        
        repr_str = repr(event)
        assert "Event" in repr_str
        assert "name=NoIdEvent" in repr_str
        assert "origin=None" in repr_str
    
    def test_event_with_none_origin(self):
        """Test Event with None origin."""
        event = Event("NullOriginEvent", None)
        
        assert event.name == "NullOriginEvent"
        assert event.origin is None
        
        repr_str = repr(event)
        assert "origin=None" in repr_str
    
    def test_event_name_types(self):
        """Test Event with different name types."""
        origin = MockUnit()
        
        # String name
        event1 = Event("StringName", origin)
        assert event1.name == "StringName"
        
        # Empty string
        event2 = Event("", origin)
        assert event2.name == ""
    
    def test_event_inheritance(self):
        """Test that Event is properly inheritable."""
        origin = MockUnit()
        
        class CustomEvent(Event):
            def __init__(self, origin, custom_data):
                super().__init__("Custom", origin)
                self.custom_data = custom_data
        
        custom = CustomEvent(origin, "test_data")
        assert isinstance(custom, Event)
        assert custom.name == "Custom"
        assert custom.custom_data == "test_data"


class TestUnitRegisteredEvent:
    """Test UnitRegisteredEvent functionality."""
    
    def test_unit_registered_event_init(self):
        """Test UnitRegisteredEvent initialization."""
        origin = MockUnit("simulator")
        registered_unit = MockUnit("new_unit")
        
        event = UnitRegisteredEvent(origin, registered_unit)
        
        assert event.name == "UnitRegistered"
        assert event.origin == origin
        assert event.registered_unit == registered_unit
    
    def test_unit_registered_event_inheritance(self):
        """Test UnitRegisteredEvent inherits from Event."""
        origin = MockUnit("sim")
        unit = MockUnit("unit")
        
        event = UnitRegisteredEvent(origin, unit)
        
        assert isinstance(event, Event)
        assert isinstance(event, UnitRegisteredEvent)
    
    def test_unit_registered_event_repr(self):
        """Test UnitRegisteredEvent string representation."""
        origin = MockUnit("sim_id")
        unit = MockUnit("unit_id")
        
        event = UnitRegisteredEvent(origin, unit)
        repr_str = repr(event)
        
        assert "UnitRegisteredEvent" in repr_str
        assert "name=UnitRegistered" in repr_str
        assert "origin=sim_id" in repr_str
    
    def test_unit_registered_event_with_none_unit(self):
        """Test UnitRegisteredEvent with None registered unit."""
        origin = MockUnit("sim")
        
        event = UnitRegisteredEvent(origin, None)
        
        assert event.registered_unit is None
        assert event.name == "UnitRegistered"
    
    def test_unit_registered_event_properties_immutable(self):
        """Test that event properties can be accessed after creation."""
        origin = MockUnit("sim")
        unit = MockUnit("unit")
        
        event = UnitRegisteredEvent(origin, unit)
        
        # Should be able to access properties
        assert event.origin.id == "sim"
        assert event.registered_unit.id == "unit"
        
        # Should be able to modify (no immutability enforced at this level)
        event.registered_unit = MockUnit("new_unit")
        assert event.registered_unit.id == "new_unit"


class TestUnitDestroyedEvent:
    """Test UnitDestroyedEvent functionality."""
    
    def test_unit_destroyed_event_init(self):
        """Test UnitDestroyedEvent initialization."""
        origin = MockUnit("simulator")
        killer = MockUnit("killer_unit")
        victim = MockUnit("victim_unit")
        
        event = UnitDestroyedEvent(origin, killer, victim)
        
        assert event.name == "UnitDestroyed"
        assert event.origin == origin
        assert event.unit_killer == killer
        assert event.unit_destroyed == victim
    
    def test_unit_destroyed_event_inheritance(self):
        """Test UnitDestroyedEvent inherits from Event."""
        event = UnitDestroyedEvent("sim", MockUnit("k"), MockUnit("v"))
        
        assert isinstance(event, Event)
        assert isinstance(event, UnitDestroyedEvent)
    
    def test_unit_destroyed_event_with_none_values(self):
        """Test UnitDestroyedEvent with None values."""
        event = UnitDestroyedEvent("sim", None, None)
        
        assert event.unit_killer is None
        assert event.unit_destroyed is None
        assert event.name == "UnitDestroyed"
    
    def test_unit_destroyed_event_self_destruction(self):
        """Test UnitDestroyedEvent where unit destroys itself."""
        origin = MockUnit("sim")
        unit = MockUnit("self_destruct")
        
        event = UnitDestroyedEvent(origin, unit, unit)
        
        assert event.unit_killer == event.unit_destroyed
        assert event.unit_killer == unit
    
    def test_unit_destroyed_event_different_origin_types(self):
        """Test UnitDestroyedEvent with different origin types."""
        # String origin
        event1 = UnitDestroyedEvent("simulator_string", MockUnit("k"), MockUnit("v"))
        assert event1.origin == "simulator_string"
        
        # Object origin
        sim_obj = MockUnit("sim_obj")
        event2 = UnitDestroyedEvent(sim_obj, MockUnit("k"), MockUnit("v"))
        assert event2.origin == sim_obj


class TestMissileEngagementEvent:
    """Test MissileEngagementEvent functionality."""
    
    def test_missile_engagement_event_init(self):
        """Test MissileEngagementEvent initialization."""
        origin = MockUnit("aircraft")
        missile = MockUnit("missile")
        target = MockUnit("target")
        
        event = MissileEngagementEvent(origin, missile, target)
        
        assert event.name == "MissileEngagement"
        assert event.origin == origin
        assert event.missile == missile
        assert event.target == target
    
    def test_missile_engagement_event_inheritance(self):
        """Test MissileEngagementEvent inherits from Event."""
        event = MissileEngagementEvent(MockUnit("ac"), MockUnit("m"), MockUnit("t"))
        
        assert isinstance(event, Event)
        assert isinstance(event, MissileEngagementEvent)
    
    def test_missile_engagement_event_with_none_missile(self):
        """Test MissileEngagementEvent with None missile."""
        origin = MockUnit("aircraft")
        target = MockUnit("target")
        
        event = MissileEngagementEvent(origin, None, target)
        
        assert event.missile is None
        assert event.target == target
    
    def test_missile_engagement_event_with_none_target(self):
        """Test MissileEngagementEvent with None target."""
        origin = MockUnit("aircraft")
        missile = MockUnit("missile")
        
        event = MissileEngagementEvent(origin, missile, None)
        
        assert event.missile == missile
        assert event.target is None
    
    def test_missile_engagement_event_same_origin_missile(self):
        """Test MissileEngagementEvent where origin is missile."""
        missile = MockUnit("missile")
        target = MockUnit("target")
        
        event = MissileEngagementEvent(missile, missile, target)
        
        assert event.origin == event.missile
        assert event.missile == missile


class TestCountermeasureEvent:
    """Test CountermeasureEvent functionality."""
    
    def test_countermeasure_event_init(self):
        """Test CountermeasureEvent initialization."""
        origin = MockUnit("aircraft")
        affected = MockUnit("affected_unit")
        
        event = CountermeasureEvent(origin, "flare", affected)
        
        assert event.name == "Countermeasure"
        assert event.origin == origin
        assert event.countermeasure_type == "flare"
        assert event.unit_affected == affected
    
    def test_countermeasure_event_inheritance(self):
        """Test CountermeasureEvent inherits from Event."""
        event = CountermeasureEvent(MockUnit("ac"), "chaff", MockUnit("tgt"))
        
        assert isinstance(event, Event)
        assert isinstance(event, CountermeasureEvent)
    
    def test_countermeasure_event_different_types(self):
        """Test CountermeasureEvent with different countermeasure types."""
        origin = MockUnit("aircraft")
        affected = MockUnit("target")
        
        types = ["flare", "chaff", "ecm", "jammer", "decoy"]
        
        for cm_type in types:
            event = CountermeasureEvent(origin, cm_type, affected)
            assert event.countermeasure_type == cm_type
            assert event.name == "Countermeasure"
    
    def test_countermeasure_event_empty_type(self):
        """Test CountermeasureEvent with empty countermeasure type."""
        event = CountermeasureEvent(MockUnit("ac"), "", MockUnit("tgt"))
        
        assert event.countermeasure_type == ""
        assert event.name == "Countermeasure"
    
    def test_countermeasure_event_self_affected(self):
        """Test CountermeasureEvent where origin affects itself."""
        unit = MockUnit("self_jamming")
        
        event = CountermeasureEvent(unit, "self_ecm", unit)
        
        assert event.origin == event.unit_affected
        assert event.countermeasure_type == "self_ecm"


class TestUnitTraceEvent:
    """Test UnitTraceEvent functionality."""
    
    def test_unit_trace_event_init(self):
        """Test UnitTraceEvent initialization."""
        origin = MockUnit("simulator")
        unit = MockUnit("traced_unit")
        state_dict = {"position": (52.0, 8.0, 1000.0), "speed": 200.0}
        
        event = UnitTraceEvent(origin, unit, state_dict)
        
        assert event.name == "UnitTrace"
        assert event.origin == origin
        assert event.unit == unit
        assert event.state_dict == state_dict
    
    def test_unit_trace_event_inheritance(self):
        """Test UnitTraceEvent inherits from Event."""
        event = UnitTraceEvent(MockUnit("sim"), MockUnit("unit"), {})
        
        assert isinstance(event, Event)
        assert isinstance(event, UnitTraceEvent)
    
    def test_unit_trace_event_empty_state(self):
        """Test UnitTraceEvent with empty state dictionary."""
        origin = MockUnit("sim")
        unit = MockUnit("unit")
        
        event = UnitTraceEvent(origin, unit, {})
        
        assert event.state_dict == {}
        assert len(event.state_dict) == 0
    
    def test_unit_trace_event_complex_state(self):
        """Test UnitTraceEvent with complex state dictionary."""
        origin = MockUnit("sim")
        unit = MockUnit("complex_unit")
        
        complex_state = {
            "position": {"lat": 52.0, "lon": 8.0, "alt": 1000.0},
            "velocity": {"vx": 100.0, "vy": 50.0, "vz": 10.0},
            "attitude": {"yaw": 90.0, "pitch": 5.0, "roll": 2.0},
            "fuel": 0.75,
            "weapons": ["AIM-120", "AIM-9"],
            "radar_status": {"active": True, "mode": "search"}
        }
        
        event = UnitTraceEvent(origin, unit, complex_state)
        
        assert event.state_dict == complex_state
        assert event.state_dict["position"]["lat"] == 52.0
        assert event.state_dict["weapons"] == ["AIM-120", "AIM-9"]
    
    def test_unit_trace_event_state_mutability(self):
        """Test that state dictionary can be modified after creation."""
        origin = MockUnit("sim")
        unit = MockUnit("unit")
        initial_state = {"speed": 200.0}
        
        event = UnitTraceEvent(origin, unit, initial_state)
        
        # Modify state through event
        event.state_dict["speed"] = 250.0
        event.state_dict["fuel"] = 0.8
        
        assert event.state_dict["speed"] == 250.0
        assert event.state_dict["fuel"] == 0.8
        assert "fuel" in event.state_dict


class TestCustomSimEvent:
    """Test CustomSimEvent functionality."""
    
    def test_custom_sim_event_init(self):
        """Test CustomSimEvent initialization."""
        origin = MockUnit("custom_origin")
        payload = {"data": "test", "value": 42}
        
        event = CustomSimEvent("MyCustomEvent", origin, payload)
        
        assert event.name == "MyCustomEvent"
        assert event.origin == origin
        assert event.payload == payload
    
    def test_custom_sim_event_inheritance(self):
        """Test CustomSimEvent inherits from Event."""
        event = CustomSimEvent("Custom", MockUnit("origin"))
        
        assert isinstance(event, Event)
        assert isinstance(event, CustomSimEvent)
    
    def test_custom_sim_event_no_payload(self):
        """Test CustomSimEvent with no payload."""
        origin = MockUnit("origin")
        
        event = CustomSimEvent("NoPayload", origin)
        
        assert event.payload == {}
        assert isinstance(event.payload, dict)
    
    def test_custom_sim_event_none_payload(self):
        """Test CustomSimEvent with None payload."""
        origin = MockUnit("origin")
        
        event = CustomSimEvent("NullPayload", origin, None)
        
        assert event.payload == {}
        assert isinstance(event.payload, dict)
    
    def test_custom_sim_event_various_payloads(self):
        """Test CustomSimEvent with various payload types."""
        origin = MockUnit("origin")
        
        # Dictionary payload
        dict_payload = {"key": "value", "number": 123}
        event1 = CustomSimEvent("DictEvent", origin, dict_payload)
        assert event1.payload == dict_payload
        
        # Empty dictionary
        event2 = CustomSimEvent("EmptyEvent", origin, {})
        assert event2.payload == {}
        
        # Complex nested payload
        complex_payload = {
            "metadata": {"timestamp": 123456789, "version": "1.0"},
            "data": [1, 2, 3],
            "flags": {"debug": True, "verbose": False}
        }
        event3 = CustomSimEvent("ComplexEvent", origin, complex_payload)
        assert event3.payload == complex_payload
    
    def test_custom_sim_event_payload_mutability(self):
        """Test that payload can be modified after creation."""
        origin = MockUnit("origin")
        initial_payload = {"initial": True}
        
        event = CustomSimEvent("MutableEvent", origin, initial_payload)
        
        # Modify payload
        event.payload["initial"] = False
        event.payload["added"] = "new_value"
        
        assert event.payload["initial"] is False
        assert event.payload["added"] == "new_value"
    
    def test_custom_sim_event_custom_names(self):
        """Test CustomSimEvent with various custom names."""
        origin = MockUnit("origin")
        
        names = [
            "UserAction",
            "SystemAlert", 
            "DataUpdate",
            "NetworkEvent",
            "ConfigChange",
            ""  # Empty name
        ]
        
        for name in names:
            event = CustomSimEvent(name, origin)
            assert event.name == name


class TestEventIntegration:
    """Integration tests for event system."""
    
    def test_event_chain(self):
        """Test chain of related events."""
        # Simulate a scenario where one event leads to another
        aircraft = MockUnit("fighter_1")
        missile = MockUnit("aim_120")
        target = MockUnit("enemy_1")
        simulator = MockUnit("sim")
        
        # Unit registration
        reg_event = UnitRegisteredEvent(simulator, aircraft)
        assert reg_event.registered_unit == aircraft
        
        # Missile engagement
        engagement_event = MissileEngagementEvent(aircraft, missile, target)
        assert engagement_event.missile == missile
        assert engagement_event.target == target
        
        # Unit destroyed
        destroy_event = UnitDestroyedEvent(simulator, missile, target)
        assert destroy_event.unit_killer == missile
        assert destroy_event.unit_destroyed == target
    
    def test_event_collection(self):
        """Test collecting multiple events."""
        origin = MockUnit("sim")
        
        events = [
            UnitRegisteredEvent(origin, MockUnit("unit1")),
            UnitRegisteredEvent(origin, MockUnit("unit2")),
            MissileEngagementEvent(MockUnit("ac"), MockUnit("m"), MockUnit("t")),
            CountermeasureEvent(MockUnit("ac"), "flare", MockUnit("missile")),
            CustomSimEvent("BattleStart", origin, {"time": 0})
        ]
        
        assert len(events) == 5
        assert all(isinstance(e, Event) for e in events)
        
        # Group by event type
        reg_events = [e for e in events if isinstance(e, UnitRegisteredEvent)]
        assert len(reg_events) == 2
    
    def test_event_filtering_by_type(self):
        """Test filtering events by type."""
        origin = MockUnit("sim")
        
        events = [
            UnitRegisteredEvent(origin, MockUnit("unit1")),
            UnitDestroyedEvent(origin, MockUnit("k"), MockUnit("v")),
            MissileEngagementEvent(MockUnit("ac"), MockUnit("m"), MockUnit("t")),
            UnitRegisteredEvent(origin, MockUnit("unit2")),
            CountermeasureEvent(MockUnit("ac"), "ecm", MockUnit("tgt"))
        ]
        
        # Filter by type
        destroyed_events = [e for e in events if isinstance(e, UnitDestroyedEvent)]
        assert len(destroyed_events) == 1
        
        registered_events = [e for e in events if isinstance(e, UnitRegisteredEvent)]
        assert len(registered_events) == 2
        
        countermeasure_events = [e for e in events if isinstance(e, CountermeasureEvent)]
        assert len(countermeasure_events) == 1


class TestEventCompatibility:
    """Test compatibility with existing event tests."""
    
    def test_event_base_repr_compatibility(self):
        """Test compatibility with existing Event repr test."""
        unit = MockUnit("u42")
        event = Event("MyEvent", unit)
        
        repr_str = repr(event)
        assert repr_str == "<Event name=MyEvent origin=u42>"
    
    def test_unit_registered_event_compatibility(self):
        """Test compatibility with existing UnitRegisteredEvent test."""
        parent = MockUnit("p")
        child = MockUnit("c")
        
        event = UnitRegisteredEvent(parent, child)
        
        assert event.origin == parent
        assert event.registered_unit == child
        assert event.name == "UnitRegistered"
    
    def test_unit_destroyed_event_compatibility(self):
        """Test compatibility with existing UnitDestroyedEvent test."""
        killer = MockUnit("killer")
        victim = MockUnit("victim")
        
        event = UnitDestroyedEvent("sim", killer, victim)
        
        assert event.unit_killer == killer
        assert event.unit_destroyed == victim
        assert event.name == "UnitDestroyed"
    
    def test_custom_sim_event_compatibility(self):
        """Test compatibility with existing CustomSimEvent test."""
        origin = MockUnit("a")
        
        # With payload
        event1 = CustomSimEvent("MyCustom", origin, {"foo": 1})
        assert event1.payload["foo"] == 1
        
        # Without payload
        event2 = CustomSimEvent("X", origin)
        assert event2.payload == {}