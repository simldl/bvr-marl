# Physics Module - Context

## Purpose
Implements high-fidelity physics models for aircraft and missile flight dynamics. Provides realistic aerodynamic forces, energy management, atmospheric effects, and afterburner modeling.

## Architecture Overview

```
physics/
├── physics.py           # BasePhysics, atmosphere, constants
├── flying_objects.py    # FlyingPhysics base class
├── aircraft.py          # AircraftPhysics with full 6-DOF
├── missiles.py          # MissilePhysics (simplified)
└── afterburner.py       # Afterburner model (NEW)
```

## Key Components

### `physics.py`
- **BasePhysics**: Foundation class for all physics models
- **ForceModel**: Aerodynamic force calculations (lift, drag, thrust)
- **AtmosphereModel**: Air density, pressure, temperature vs altitude
- **Constants**: Physical constants (gravity, air speed of sound, etc.)

### `flying_objects.py`
- **FlyingPhysics**: Base class for aircraft and missiles
- **Energy management**: Specific energy rate calculations
- **Turn performance**: Instantaneous and sustained turn rates
- **Stall protection**: Energy-aware flight envelope protection
- **Attitude dynamics**: Pitch, yaw, roll rate limiting

### `aircraft.py`
- **AircraftPhysics**: Complete aircraft flight dynamics
- **Drag modeling**: Mach-dependent CD curves with wave drag
- **Lift modeling**: CL_max variations with Mach number
- **Engine modeling**: Thrust vs speed and altitude with afterburner
- **Load factor limits**: Structural and aerodynamic constraints
- **Afterburner integration**: Thrust augmentation and fuel consumption

### `missiles.py`
- **MissilePhysics**: Specialized physics for guided missiles
- **Simplified aerodynamics**: Appropriate for missile flight regimes
- **Thrust profiles**: Boost and sustain phases
- **Maneuverability**: High-g capability modeling

### `afterburner.py` (NEW)
- **Afterburner**: Self-contained thrust augmentation system
- **Spool dynamics**: Smooth on/off transitions (first-order lag)
- **Thrust multiplication**: 1.6× MIL thrust at full afterburner
- **Fuel consumption**: Realistic SFC modeling (MIL vs AB)
- **Parasite drag**: Extra CD0 when afterburner is lit
- **Mach limits**: Minimum Mach requirement for AB engagement

## Physics Model Features

### Aircraft Flight Dynamics
- **4-DOF simulation**: Position (lat, lon), altitude and 1D-velocity
- **Total energy integration**: Specific energy rate (Ps) with altitude ↔ speed trades
- **Load factor integration**: Commanded n from action processor affects drag/Ps
- **Envelope protection**: Stall prevention and recovery
- **Overspeed guards**: Mach-NE and EAS-NE limits for realistic high-speed behavior
- **Realistic performance**: Based on F-16 characteristics
- **Afterburner modeling**: Dynamic thrust augmentation with fuel penalties

### Aerodynamic Modeling

#### Drag Coefficient (Mach-Dependent)
```python
CD = physics.get_base_drag_cd(velocity, altitude)
# Returns CD0 with wave drag and afterburner effects:
# - Subsonic (M < 0.8): CD0 = 0.009
# - Transonic (0.8 < M < 1.05): Wave drag peak (CD0 up to 0.036)
# - Supersonic (M > 1.05): CD0 stabilizes around 0.031
# - AB augmentation: +0.002 when afterburner is lit (nozzle/plume losses)
```

#### Load Factor Limits
```python
# Instantaneous: Aerodynamic + structural
n_max_inst = physics.compute_instantaneous_load_factor(v, alt)

# Sustained: Thrust-limited
n_max_sust = physics.compute_sustained_load_factor(v, alt, throttle)
```

### Engine Performance

#### Thrust Model (MIL and Afterburner)
```python
# Military power (MIL):
T_mil = (m × g) × f_v(v) × (ρ/ρ0)^0.8
# where f_v = 1.0 + 0.7 × (M² - M_full²) for v > v_full

# Afterburner (AB):
T_ab = thrust_mult × T_mil × (ρ/ρ0)^density_exp
# thrust_mult = 1.6 (60% thrust increase)
# density_exp = 0.8 (altitude sensitivity)
```

**Thrust Blend:**
```python
T_actual = throttle × [(1 - spool) × T_mil + spool × T_ab]
# spool ∈ [0, 1]: smooth first-order lag with τ_on = 0.8s, τ_off = 0.4s
```

#### Fuel Consumption
```python
# Specific Fuel Consumption (SFC):
SFC_mil = 1.8e-5 kg/(N·s)  # ~0.064 kg/(N·hr)
SFC_ab  = 3.8e-5 kg/(N·s)  # ~0.137 kg/(N·hr) - 2.1× worse

# Fuel flow:
fuel_flow_kgps = SFC × thrust_N
# Interpolated between MIL and AB based on spool state
```

### Atmospheric Effects
- **Standard atmosphere**: Temperature, pressure, density profiles
- **Air density variations**: Affects all aerodynamic forces
- **Speed of sound**: Temperature-dependent for Mach calculations

## Afterburner System

### Configuration (`AfterburnerParams`)

```python
@dataclass
class AfterburnerParams:
    enabled: bool = True              # Enable/disable afterburner
    thrust_mult: float = 1.6          # AB thrust = 1.6 × MIL thrust
    density_exp: float = 0.8          # Altitude sensitivity exponent
    min_mach: float = 0.70            # Minimum Mach for AB engagement
    tau_on_s: float = 0.8             # Spool-up time constant (seconds)
    tau_off_s: float = 0.4            # Spool-down time constant (seconds)
    delta_cd0: float = 0.002          # Extra parasite drag when AB lit
    sfc_mil_kgps_per_N: float = 1.8e-5   # MIL fuel consumption
    sfc_ab_kgps_per_N: float = 3.8e-5    # AB fuel consumption
```

### API

```python
# Controller interface
afterburner.set_command(on=True)      # Request AB on/off
afterburner.update_spool(dt, v, alt)  # Update spool dynamics (called every tick)

# Physics interface
thrust = afterburner.engine_force(v, alt, throttle)  # Get current thrust
cd0 = afterburner.effective_cd0(base_cd0)            # Drag with AB penalty
fuel_rate = afterburner.fuel_flow_kgps(thrust)       # Fuel consumption

# State queries
spool = afterburner.spool              # Current spool state (0..1)
is_on = afterburner.is_commanded       # Controller command
enabled = afterburner.is_enabled       # System enabled
```

### Spool Dynamics

The afterburner uses first-order lag to model realistic engagement delays:

```python
# Target state:
target = 1.0 if (commanded AND Mach >= min_mach) else 0.0

# First-order filter:
α = 1 - exp(-dt / τ)
spool_new = spool + α × (target - spool)

# Time constants:
τ_on  = 0.8s   # Ignition, fuel flow stabilization
τ_off = 0.4s   # Faster shutdown
```

**Behavior:**
- **Ignition delay**: ~0.8s from command to full thrust
- **Mach check**: AB won't light below M=0.7 (prevents low-speed misuse)
- **Smooth shutdown**: ~0.4s spool-down time
- **Numeric hygiene**: Clamps to exactly 0.0 or 1.0 at extremes

### Integration with AircraftPhysics

```python
class AircraftPhysics(FlyingPhysics):
    def __init__(self, params):
        super().__init__(params)
        self.afterburner = Afterburner(
            air=self.air,
            mass_kg=self.mass_kg,
            g=self.g,
            engine_prof=self.engine_prof,
            speed_full_thrust=self.speed_full_thrust,
            get_speed_of_sound_fn=get_speed_of_sound,
            params=AfterburnerParams(...)
        )

    def get_engine_force(self, v, alt, throttle):
        # Automatically blends MIL/AB based on spool
        return self.afterburner.engine_force(v, alt, throttle)

    def get_base_drag_cd(self, v, alt):
        base = self._compute_mach_cd(v, alt)  # Mach-dependent CD0
        return self.afterburner.effective_cd0(base)  # Add AB drag
```

### Performance Impact

**Thrust Increase:**
- Sea level, M=0.9: **+60%** thrust (1.0 → 1.6 × mg)
- 10,000m, M=1.2: **+50%** thrust (density effects)
- 15,000m, M=1.5: **+40%** thrust (altitude degradation)

**Fuel Consumption:**
- MIL power: 1.8e-5 kg/(N·s) → ~0.8 kg/s at 45 kN
- AB power:  3.8e-5 kg/(N·s) → ~2.7 kg/s at 72 kN
- **Penalty**: 2.1× worse SFC, 3.4× absolute fuel flow

**Drag Penalty:**
- CD0 increase: +0.002 (nozzle open, plume base drag)
- At M=1.2, q=40 kPa, S=30 m²: **+2.4 kN drag**
- Negligible compared to +27 kN thrust gain

## Key Algorithms

### Energy Management
- **Specific Energy**: E = h + V²/(2g)
- **Energy Rate**: Ps = (T-D)×V/W - climb_rate×sin(γ)
- **Trade-offs**: Speed vs altitude optimization

### Turn Performance
- **Instantaneous turns**: Limited by max load factor
- **Sustained turns**: Limited by thrust-to-drag ratio
- **Energy bleed**: Turn rate vs energy loss trade-offs

### Stall Protection
- **Dynamic stall speed**: Varies with load factor and altitude
- **Automatic recovery**: Pitch down + full thrust
- **Energy relief**: Gradual pitch adjustment near stall

## Integration with Control Systems

### Rate Limits
```python
# Physics-based turn rate limits
omega_max = physics.compute_instantaneous_turn_rate(speed, altitude)

# Pitch rate limits
q_max = physics.get_pitch_limit(position, speed)
```

### Envelope Constraints
- **Stall margins**: Prevent departure from controlled flight
- **Ceiling limits**: Reduce performance at high altitude
- **G-load limits**: Structural and physiological constraints
- **Overspeed limits**: Mach-NE and EAS-NE enforcement

### Afterburner Control
```python
# Typically called from aircraft controller/action processor:
aircraft.physics.afterburner.set_command(ab_requested)

# Physics updates spool state every tick:
aircraft.physics.afterburner.update_spool(dt, v, alt)

# Thrust is automatically blended in get_engine_force()
```

## Configuration Examples

### Fighter Aircraft (F-16 Style)
```python
params = AircraftPhysics.Params(
    mass_kg=18000,
    reference_area_m2=30,
    aspect_ratio=7.5,
    oswald_e=0.82,
    max_speed_mps=680,
    n_max=9.0,
    service_ceiling_m=18000,
    mach_ne=2.2,
    eas_ne_mps=410,
    # Afterburner is initialized in __init__
)
```

### Afterburner Tuning
```python
# Conservative (lower performance, better fuel economy)
AfterburnerParams(
    thrust_mult=1.4,        # +40% thrust (vs default 60%)
    min_mach=0.8,           # Restrict to high-speed only
    tau_on_s=1.2,           # Slower spool-up
    delta_cd0=0.003,        # Higher drag penalty
)

# Aggressive (higher performance, worse fuel economy)
AfterburnerParams(
    thrust_mult=1.8,        # +80% thrust
    min_mach=0.6,           # Allow lower-speed AB
    tau_on_s=0.5,           # Faster response
    delta_cd0=0.001,        # Lower drag penalty
)
```

## Recent Improvements

### Physics Model (2024-2025)
- **External load factor integration**: Physics now consumes n_external from action processor
- **Overspeed protection**: Mach-NE and EAS-NE guards for realistic high-speed envelope
- **Energy-correct maneuvers**: Vertical maneuvers (pull-ups/push-overs) now affect drag/Ps
- **Natural speed excursions**: Physics allows TAS to exceed commanded range via energy trades
- **Enhanced stall modeling**: More realistic departure and recovery
- **Improved energy equations**: Better climb/dive performance
- **Mach-dependent limits**: Realistic high-speed constraints
- **Load factor calculations**: Separate instantaneous vs sustained

### Afterburner Integration (October 2025)
- **Full AB system**: Dynamic thrust augmentation with spool dynamics
- **Realistic SFC**: 2.1× fuel penalty for AB operation
- **Drag modeling**: Nozzle/plume losses when AB is lit
- **Mach constraints**: Minimum Mach requirement prevents low-speed misuse
- **Smooth transitions**: First-order lag eliminates discontinuities
- **Integration**: Seamless integration with existing thrust/drag models

## Configuration Parameters

### Aircraft Parameters
- **Aircraft mass**: Affects all performance metrics
- **Wing area**: Determines lift and drag scaling
- **Engine thrust**: Maximum available power (MIL and AB)
- **Aerodynamic coefficients**: Drag and lift curve parameters
- **Overspeed limits**: mach_ne (default 2.2), eas_ne_mps (default 410 m/s)

### Afterburner Parameters
- **thrust_mult**: AB thrust multiplier (default 1.6)
- **density_exp**: Altitude sensitivity (default 0.8)
- **min_mach**: Minimum Mach for engagement (default 0.7)
- **tau_on_s**: Spool-up time constant (default 0.8s)
- **tau_off_s**: Spool-down time constant (default 0.4s)
- **delta_cd0**: Extra parasite drag (default 0.002)
- **sfc_mil/ab**: Fuel consumption rates

## Testing

### Validation Approach
- **Flight test data**: Compared against published performance
- **Energy diagrams**: Ps vs Mach number validation
- **Turn rate curves**: Instantaneous and sustained comparisons
- **Stall behavior**: Realistic departure and recovery characteristics
- **AB performance**: Thrust increase and fuel consumption validation

### Test Scenarios
1. **Acceleration runs**: MIL vs AB thrust comparison
2. **Sustained turns**: Energy bleed with/without AB
3. **Climb performance**: Ps curves at various Mach/altitude
4. **Fuel consumption**: Mission profile endurance tests
5. **Transient response**: AB spool-up/down dynamics

## Performance Considerations

### Computational Efficiency
- **Vectorized operations**: Efficient batch calculations
- **Lookup tables**: Pre-computed atmospheric properties
- **Analytical solutions**: Where possible, avoid numerical integration
- **Stability checks**: Prevent numerical divergence

### Memory Footprint
- **Stateless afterburner**: Only stores spool state (one float)
- **No lookup tables**: Analytical thrust/drag functions
- **Minimal overhead**: ~0.1% performance impact

## Known Limitations

### Aerodynamics
- **Simplified 4-DOF**: No explicit roll/yaw (heading rate only)
- **Planar lift assumption**: No sideslip or bank angle effects
- **No compressibility corrections**: Simple Mach-dependent CD curves
- **Fixed CL_max**: No flap/slat modeling

### Engine Model
- **Simplified altitude effects**: No turbine inlet temperature limits
- **No engine windmilling**: Thrust assumed zero at idle
- **No reheat schedules**: AB is binary (on/off), no partial AB
- **No transient temperature effects**: Instant thermal response

### Afterburner
- **Binary AB**: No partial afterburner (full or off)
- **No reheat scheduling**: Real jets have multiple AB stages
- **Simplified SFC**: No Mach/altitude dependence on SFC
- **No temperature limits**: No EGT constraints

## Future Enhancements

### Aerodynamics
- **6-DOF dynamics**: Full attitude representation
- **Bank angle effects**: Lift vector decomposition
- **Compressibility corrections**: Prandtl-Glauert transformation
- **High-alpha modeling**: Post-stall aerodynamics

### Engine Model
- **Turbine limits**: Inlet temperature and compressor stall
- **Variable nozzle**: Thrust vectoring simulation
- **Partial afterburner**: Multi-stage reheat
- **Transient response**: Engine acceleration/deceleration lag

### Energy Management
- **Optimal energy tactics**: AI-driven Ps management
- **Energy-aware autopilot**: Maintains energy state
- **Predictive stall warning**: Energy-based departure prediction

### Environmental Effects
- **Winds aloft**: Layered wind model
- **Temperature deviations**: Non-standard atmosphere
- **Icing effects**: Performance degradation in clouds

## References

- **Stevens & Lewis**: "Aircraft Control and Simulation" (3rd Ed)
- **McCormick**: "Aerodynamics, Aeronautics, and Flight Mechanics"
- **Mattingly**: "Elements of Propulsion: Gas Turbines and Rockets"
- **Yechout et al.**: "Introduction to Aircraft Flight Mechanics"
- **F-16 Flight Manual**: TO 1F-16C-1 (publicly available excerpts)
- **NASA Technical Reports**: Engine performance data

## Usage Example

```python
# Initialize physics with afterburner
params = AircraftPhysics.Params(mass_kg=18000, reference_area_m2=30)
physics = AircraftPhysics(params)

# In control loop:
# 1. Update afterburner command
physics.afterburner.set_command(ab_requested)

# 2. Update spool dynamics
physics.afterburner.update_spool(dt=0.1, v_mps=300, alt_m=10000)

# 3. Get thrust (automatically blended)
thrust = physics.get_engine_force(v_mps=300, alt_m=10000, throttle=1.0)

# 4. Get drag (with AB penalty)
cd = physics.get_base_drag_cd(v_mps=300, alt_m=10000)

# 5. Monitor state
spool = physics.afterburner.spool
fuel_rate = physics.afterburner.fuel_flow_kgps(thrust)
```

## Notes

- **Afterburner is always initialized** in AircraftPhysics.__init__() with default parameters
- **Spool state** must be updated every physics tick for smooth dynamics
- **Controller responsibility** to call set_command() based on pilot/agent input
- **Fuel consumption** is for logging/telemetry only (no fuel tank simulation yet)
- **AB drag penalty** is automatically included in get_base_drag_cd()
