# Simulator Module - Context

## Purpose
Core simulation engine that orchestrates the entire air combat simulation. Manages time stepping, unit lifecycle, event handling, and coordinate transformations.

## Key Components

### `simulator.py`
- **Simulator class**: Main simulation engine
- **Time management**: Fixed timestep simulation (typically 1s)
- **Unit management**: Tracks active units (aircraft, missiles)
- **Event system**: Handles unit creation, removal, and state changes
- **Boundary enforcement**: Manages map limits and violations

### `core/` Subfolder
Contains fundamental simulation building blocks:
- **`units.py`**: Base classes for all simulation entities
- **`events.py`**: Event system for inter-unit communication
- **`helpers.py`**: Utility classes (Position, Velocity, etc.)

### `utils/` Subfolder
Simulation utilities and mathematical functions:
- **`angles.py`**: Angle normalization and difference calculations
- **`geodesics.py`**: Geographic coordinate transformations
- **`map_limits.py`**: Boundary definition and checking

## Key Features

### Unit Lifecycle Management
```python
# Add units to simulation
unit_id = simulator.add_unit(aircraft)

# Automatic cleanup of destroyed/removed units
simulator.do_tick()  # Processes removals

# Event-driven removal with reasons
UnitRemovedEvent(unit_id, reason="boundary_violation")
```

### Time Management
- **Fixed timestep**: Ensures deterministic simulation
- **Substep support**: For higher precision physics
- **Event scheduling**: Delayed events and triggers

### Boundary System
- **Map limits**: Geographic boundaries with lat/lon/altitude
- **Delayed removal**: 5-step countdown for boundary violations
- **Violation tracking**: Persistent state across simulation steps

### Coordinate Systems
- **WGS84 coordinates**: Latitude, longitude, altitude
- **Local tangent plane**: For physics calculations
- **Geodetic transformations**: Accurate distance/bearing

## Integration Points

### With Aircraft
- Aircraft register with simulator on creation
- Simulator calls `aircraft.update()` each tick
- Boundary violations trigger delayed removal process

### With Physics
- Provides timestep for physics integration
- Manages coordinate frame transformations
- Handles collision detection and response

### With RL Environment
- Exposes `active_units` for observation space
- Processes actions through unit updates
- Generates reward-relevant events

## Recent Changes
- **Enhanced boundary system**: Delayed removal with countdown
- **Event handling**: Improved UnitRemovedEvent processing
- **Performance optimization**: Efficient unit management
- **Substepping**: Calculating substeps for units (target and missile) when they are close
- **Hit Calculation**: Handling of missile impact

## Configuration
- **Tick rate**: Configurable timestep (default 1s)
- **Map bounds**: Flexible boundary definitions
- **Event logging**: Optional event history tracking

## Usage Example
```python
simulator = Simulator(tick_secs=0.1)
aircraft_id = simulator.add_unit(aircraft)
missile_id = simulator.add_unit(missile)

for step in range(1000):
    events = simulator.do_tick()
    # Process events for rewards, logging, etc.
```