import pytest
import numpy as np
from simulator.simulator import Simulator
from simulator.utils.map_limits import MapLimits
from simulator.core.helpers import Position
from simulator.core.events import UnitRemovedEvent
from aircrafts.aircraft import Aircraft
from aircrafts.types.debug_plane import DebugPlane


@pytest.fixture
def map_limits():
    return MapLimits(left_lon=-1.0, bottom_lat=-1.0, right_lon=1.0, top_lat=1.0, min_alt=0, max_alt=20000)


@pytest.fixture
def simulator():
    return Simulator(tick_secs=0.1)


def test_aircraft_removed_on_lat_boundary_violation(simulator, map_limits):
    """Test that aircraft is removed when it violates latitude boundaries."""
    # Create aircraft near the top boundary, flying north
    position = Position(lat=0.9, lon=0.0, alt=10000.0)
    aircraft = DebugPlane(
        position=position,
        yaw_deg=0.0,  # North
        speed_mps=200.0,
        group="BLUE",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0
    )

    aircraft_id = simulator.add_unit(aircraft)

    # Run simulation until aircraft should cross boundary
    # At 200 m/s and 0.1s per tick, it moves 20m per tick
    # 0.1 degrees latitude ≈ 11.1 km, so it should take ~555 ticks to cross
    # Let's run for fewer ticks but force a boundary violation
    events_all = []
    for _ in range(20):
        events = simulator.do_tick()
        events_all.extend(events)
        if aircraft_id not in simulator.active_units:
            break

    # Force boundary violation by setting position beyond boundary
    if aircraft_id in simulator.active_units:
        aircraft.position.lat = map_limits.top_lat + 0.01  # Exceed boundary
        # Boundary violation triggers a 5-tick countdown before removal
        # Run 6 ticks to ensure removal happens (1 to detect + 5 countdown)
        for _ in range(6):
            events = simulator.do_tick()
            events_all.extend(events)
            if aircraft_id not in simulator.active_units:
                break

    # Check that aircraft was removed
    assert aircraft_id not in simulator.active_units

    # Check that a UnitRemovedEvent was generated
    removal_events = [e for e in events_all if isinstance(e, UnitRemovedEvent)]
    assert len(removal_events) == 1
    assert removal_events[0].removed_unit.id == aircraft_id
    assert removal_events[0].reason == "boundary_violation"


def test_aircraft_removed_on_lon_boundary_violation(simulator, map_limits):
    """Test that aircraft is removed when it violates longitude boundaries."""
    # Create aircraft near the right boundary, flying east
    position = Position(lat=0.0, lon=0.9, alt=10000.0)
    aircraft = DebugPlane(
        position=position,
        yaw_deg=90.0,  # East
        speed_mps=200.0,
        group="BLUE",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0
    )

    aircraft_id = simulator.add_unit(aircraft)

    # Force boundary violation
    aircraft.position.lon = map_limits.right_lon + 0.01  # Exceed boundary

    # Boundary violation triggers a 5-tick countdown before removal
    # Run 6 ticks to ensure removal happens (1 to detect + 5 countdown)
    events_all = []
    for _ in range(6):
        events = simulator.do_tick()
        events_all.extend(events)
        if aircraft_id not in simulator.active_units:
            break

    # Check that aircraft was removed
    assert aircraft_id not in simulator.active_units

    # Check that a UnitRemovedEvent was generated
    removal_events = [e for e in events_all if isinstance(e, UnitRemovedEvent)]
    assert len(removal_events) == 1
    assert removal_events[0].removed_unit.id == aircraft_id
    assert removal_events[0].reason == "boundary_violation"


def test_aircraft_not_removed_on_altitude_boundary(simulator, map_limits):
    """Test that aircraft is NOT removed when it hits altitude boundaries (only clamped)."""
    # Create aircraft at max altitude (use aircraft max_alt, not map_limits max_alt)
    aircraft_max_alt = 15000.0
    position = Position(lat=0.0, lon=0.0, alt=aircraft_max_alt + 1000.0)
    aircraft = DebugPlane(
        position=position,
        yaw_deg=0.0,
        speed_mps=200.0,
        group="BLUE",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=aircraft_max_alt
    )

    aircraft_id = simulator.add_unit(aircraft)

    # Run a few ticks
    events_all = []
    for _ in range(5):
        events = simulator.do_tick()
        events_all.extend(events)

    # Check that aircraft is still active (altitude should be clamped, not removed)
    assert aircraft_id in simulator.active_units
    # Altitude should be clamped to aircraft's max altitude, not exactly equal due to physics
    assert aircraft.position.alt <= aircraft_max_alt

    # Check that no removal events were generated
    removal_events = [e for e in events_all if isinstance(e, UnitRemovedEvent)]
    assert len(removal_events) == 0


def test_aircraft_stays_within_boundaries(simulator, map_limits):
    """Test that aircraft within boundaries are not removed."""
    # Create aircraft well within boundaries
    position = Position(lat=0.0, lon=0.0, alt=10000.0)
    aircraft = DebugPlane(
        position=position,
        yaw_deg=45.0,
        speed_mps=100.0,
        group="BLUE",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0
    )

    aircraft_id = simulator.add_unit(aircraft)

    # Run simulation for several ticks
    events_all = []
    for _ in range(10):
        events = simulator.do_tick()
        events_all.extend(events)

    # Check that aircraft is still active
    assert aircraft_id in simulator.active_units

    # Check that no removal events were generated
    removal_events = [e for e in events_all if isinstance(e, UnitRemovedEvent)]
    assert len(removal_events) == 0


def test_boundary_check_method():
    """Test the boundary checking method directly."""
    map_limits = MapLimits(left_lon=-1.0, bottom_lat=-1.0, right_lon=1.0, top_lat=1.0)

    # Create a test aircraft
    position = Position(lat=0.0, lon=0.0, alt=10000.0)
    aircraft = DebugPlane(
        position=position,
        yaw_deg=0.0,
        speed_mps=100.0,
        group="BLUE",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0
    )

    # Test positions within boundaries
    aircraft.position.lat = 0.5
    aircraft.position.lon = 0.5
    assert not aircraft.control._check_boundary_violation(aircraft.position)

    # Test latitude violation
    aircraft.position.lat = 1.1  # Beyond top_lat
    aircraft.position.lon = 0.0
    assert aircraft.control._check_boundary_violation(aircraft.position)

    aircraft.position.lat = -1.1  # Beyond bottom_lat
    aircraft.position.lon = 0.0
    assert aircraft.control._check_boundary_violation(aircraft.position)

    # Test longitude violation
    aircraft.position.lat = 0.0
    aircraft.position.lon = 1.1  # Beyond right_lon
    assert aircraft.control._check_boundary_violation(aircraft.position)

    aircraft.position.lat = 0.0
    aircraft.position.lon = -1.1  # Beyond left_lon
    assert aircraft.control._check_boundary_violation(aircraft.position)

    # Reset to valid position
    aircraft.position.lat = 0.0
    aircraft.position.lon = 0.0
    assert not aircraft.control._check_boundary_violation(aircraft.position)