# Missiles Module - Context

## Purpose
Implements guided missile systems with realistic aerodynamics, guidance algorithms, and engagement logic. Supports multiple missile types (Fox-1, Fox-2, Fox-3) with different guidance modes.

## Key Components

### `missile.py`
- **Missile class**: Base missile entity with flight dynamics
- **Guidance integration**: Interfaces with guidance algorithms
- **Target tracking**: Maintains lock and engagement state
- **Warhead modeling**: Proximity fuzing and damage assessment

### `core/` Subfolder
Core missile systems and utilities:
- **Base classes**: Common missile functionality
- **Flight dynamics**: Simplified aerodynamics for missile flight
- **Fuzing logic**: Proximity detection and warhead activation

### `guidance/` Subfolder
Missile guidance algorithms and targeting:
- **Proportional navigation**: Classic guidance law implementation
- **Pursuit guidance**: Direct pursuit algorithms
- **Target prediction**: Lead angle calculations
- **Guidance filters**: Noise rejection and smoothing

### Missile Type Subfolders

#### `fox1/` - Semi-Active Radar Homing (SARH)
- **SARH guidance**: Requires continuous illumination from launch platform
- **Radar seeker**: Homes on reflected radar energy
- **Medium range**: Beyond visual range capability
- **Datalink**: Receives updates from launching aircraft

#### `fox2/` - Infrared Homing
- **IR seeker**: Homes on target heat signature
- **Short/medium range**: Visual to beyond visual range
- **Fire-and-forget**: No guidance from launch platform needed
- **Flare susceptibility**: Vulnerable to IR countermeasures

#### `fox3/` - Active Radar Homing (ARH)
- **Active radar**: Self-contained radar seeker
- **Fire-and-forget**: Autonomous after launch
- **Long range**: Extended beyond visual range
- **HOJ capability**: Home-on-jam against ECM

## Key Features

### Guidance Systems
```python
# Proportional navigation guidance
def compute_guidance(self, target_position, target_velocity):
    los_rate = self.calculate_line_of_sight_rate()
    acceleration = self.nav_constant * self.velocity * los_rate
    return acceleration
```

### Multi-Mode Seekers
- **Active radar**: Self-contained target detection
- **Semi-active radar**: Requires external illumination
- **Infrared**: Heat signature tracking
- **Inertial navigation**: Initial guidance phase

### Flight Phases
1. **Boost phase**: High acceleration motor burn
2. **Midcourse**: Inertial or datalink guidance
3. **Terminal**: Active seeker guidance
4. **Intercept**: Proximity fuzing and warhead

### Countermeasure Interaction
- **Chaff vulnerability**: Radar-guided missiles
- **Flare vulnerability**: IR-guided missiles
- **ECM effects**: Radar jamming and deception
- **Maneuver defeat**: Kinematic limitations

## Physics Integration

### Missile-Specific Dynamics
- **High maneuverability**: Sustained high-g capability
- **Thrust vectoring**: Some advanced variants
- **Aerodynamic controls**: Fins and control surfaces
- **Energy management**: Boost-coast flight profiles

### Performance Characteristics
```python
# Missile performance parameters
max_range_km = 100          # Maximum engagement range
max_g = 40                  # Maximum load factor
burn_time_s = 5             # Motor burn duration
coast_time_s = 45           # Unpowered flight time
```

## Guidance Algorithm Details

### Proportional Navigation
- **Principle**: Acceleration proportional to line-of-sight rate
- **Navigation constant**: Typically 3-5 for optimal performance
- **Lead angle**: Predicts target future position
- **Stability**: Well-proven guidance law

### Pursuit Guidance
- **Direct pursuit**: Aims directly at target
- **Pure pursuit**: Simple but inefficient
- **Modified pursuit**: Includes lead angle correction

### Terminal Guidance
- **Seeker cone**: Limited field of view
- **Lock-on range**: Distance at which seeker acquires target
- **Gimbal limits**: Seeker pointing constraints
- **Track gates**: Maintain lock through maneuvers

## Target Engagement Logic

### Launch Envelope
- **NEZ calculations**: No-escape zone boundaries
- **Kinematic constraints**: Range and aspect limitations
- **Energy considerations**: Target and missile energy states
- **Probability of kill**: Engagement success estimates

### Multi-Target Scenarios
- **Target prioritization**: Threat assessment algorithms
- **Missile allocation**: Optimal target assignment
- **Fraticide avoidance**: Prevent friendly fire
- **Coordinated attacks**: Time-on-target coordination

## Integration Points

### With Aircraft
- **Launch commands**: Weapon system integration
- **Datalink**: Continuous guidance updates (Fox-1)
- **Target designation**: Initial target assignment
- **Battle damage assessment**: Hit/miss evaluation

### With Radar Systems
- **Target illumination**: SARH missile support
- **Track data**: Target position and velocity
- **ECM effects**: Jamming and countermeasure modeling
- **Multi-path**: Ground reflection effects

### With Physics
- **Trajectory calculation**: 6-DOF flight simulation
- **Atmospheric effects**: Air density and wind
- **Collision detection**: Warhead proximity fuzing
- **Debris modeling**: Post-intercept fragments

## Recent Enhancements
- **Improved guidance filters**: Better noise rejection
- **Enhanced countermeasure modeling**: Realistic effectiveness
- **Multi-target capability**: Simultaneous engagements
- **Physics integration**: More accurate flight dynamics

## Testing Framework
- **Unit tests**: Individual guidance algorithms
- **Integration tests**: Full missile-target scenarios
- **Monte Carlo**: Statistical performance analysis
- **Hardware-in-loop**: Real seeker integration (future)

## Performance Optimization
- **Guidance updates**: Efficient computation rates
- **Memory management**: Minimize allocation overhead
- **Vectorized operations**: Batch missile processing
- **Early termination**: Miss distance prediction