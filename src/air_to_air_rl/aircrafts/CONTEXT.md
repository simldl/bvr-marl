# Aircrafts Module - Context

## Overview

The aircrafts module implements complete fighter aircraft systems with realistic physics, sensors, weapons, and AI control interfaces. It serves as the bridge between low-level physics simulation and high-level reinforcement learning control, providing a comprehensive platform for BVR (Beyond Visual Range) air combat training.

**Primary Components:**
- Aircraft entity with integrated subsystems (control, sensors, weapons, countermeasures)
- Flight control system with physics integration
- Weapon systems (missiles and gun)
- Sensor fusion (radar, passive radar, missile warning)
- Tactical calculators (NEZ, DLZ, SQI)
- Multiple aircraft type implementations (F-22, F-35, Su-57, Eurofighter)

---

## Directory Structure

```
aircrafts/
├── aircraft.py                          # Main Aircraft class (entity + subsystem integration)
│
├── core/                                # Core utilities and calculations
│   ├── nez.py                          # No Escape Zone & DLZ calculator
│   ├── target_prio.py                  # Track prioritization system
│   └── __init__.py
│
├── control/                            # Flight control and weapon management
│   ├── movement_control.py            # Flight control system (physics integration)
│   ├── weapon_system.py               # Missile and gun firing logic
│   ├── gun_projectile.py              # Gun system and ballistic physics
│   ├── countermeasure.py              # Flares, chaff, ECM, decoys
│   └── __init__.py
│
├── systems/                            # Aircraft subsystems
│   ├── sensor.py                      # Sensor system orchestrator
│   ├── passive_radar.py               # Passive radar warning receiver
│   ├── missile_warner.py              # Missile approach warning
│   ├── observation_helper.py          # RL observation construction
│   ├── metrics_helper.py              # Performance metrics
│   └── __init__.py
│
└── types/                              # Specific aircraft implementations
    ├── f22.py                          # F-22 Raptor (VLO, AESA, AIM-120)
    ├── f35.py                          # F-35 Lightning II
    ├── su57.py                         # Su-57 Felon
    ├── eurofighter.py                  # Eurofighter Typhoon
    ├── debug_plane.py                  # Simplified test aircraft
    └── __init__.py
```

---

## Main Aircraft Class

**File:** `aircraft.py`

The `Aircraft` class is the central entity that inherits from `FlyingUnit` and integrates all subsystems.

### Initialization

```python
class Aircraft(FlyingUnit):
    def __init__(self, name, position, yaw_deg, speed_mps, group, map_limits, config):
        super().__init__(name=name, position=position, yaw_deg=yaw_deg, speed=speed_mps)

        # Core attributes
        self.group = group                    # Team affiliation (blue/red)
        self.type = "Aircraft"
        self.map_limits = map_limits

        # Boundary violation tracking
        self.boundary_violation_active = False
        self.boundary_violation_countdown = 0
        self.boundary_violation_penalty_per_step = -10.0

        # Subsystems
        self.physics = AircraftPhysics(...)   # Flight dynamics
        self.radar = AircraftRadar(...)       # Active radar
        self.sensor = AircraftSensorSystem(self)
        self.control = AircraftControlSystem(self)
        self.weapons = AircraftWeaponSystem(self)
        self.countermeasures = AircraftCountermeasureSystem(self)
        self.wez = NoEscapeZoneCalculator(self)
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `physics` | AircraftPhysics | Flight dynamics model (mass, drag, thrust, envelope) |
| `radar` | AircraftRadar | Active radar system (detection, tracking, lock) |
| `sensor` | AircraftSensorSystem | Sensor orchestrator (radar, passive, missile warning) |
| `control` | AircraftControlSystem | Flight control system (attitude, throttle, movement) |
| `weapons` | AircraftWeaponSystem | Weapon management (missiles, gun) |
| `countermeasures` | AircraftCountermeasureSystem | Defensive systems (flares, chaff, ECM) |
| `wez` | NoEscapeZoneCalculator | Tactical engagement zone calculator |
| `target` | Aircraft | Currently selected target |
| `missiles` | List[Missile] | List of fired missiles still in flight |

### Update Cycle

```python
def update(self, tick_secs: float, sim: Simulator) -> List[Event]:
    # 1. Update flight control (physics tick)
    self.control.update_movement(tick_secs)

    # 2. Update sensor data (radar, passive, missile warning)
    self.sensor.update_sensor_data(sim, tick_secs)

    # 3. Handle boundary violation countdown
    if self.boundary_violation_active:
        self.boundary_violation_countdown -= 1
        if self.boundary_violation_countdown <= 0:
            self.should_be_removed = True

    return []
```

### RL Action Interface

```python
def apply_rl_action(self, action: np.ndarray, simulator: Simulator):
    """Apply 10-dimensional RL action vector."""
    # action[0]: Throttle (0..1)
    self.control.set_throttle(action[0])

    # action[1-2]: Yaw/pitch commands (deltas)
    yaw_change = (action[1] * 2 - 1) * 180°
    pitch_change = (action[2] * 2 - 1) * 90°
    self.control.set_yaw_deg(self.yaw_deg + yaw_change)
    self.control.set_pitch_deg(self.pitch_deg + pitch_change)

    # action[3-5]: Target selection, missile fire, gun fire
    self.target = self.weapons.select_and_engage_target(
        candidates, action[3], action[4], action[5], simulator
    )

    # action[6-9]: Countermeasures (flares, chaff, ECM, decoys)
    if action[6] > 0.5: self.countermeasures.launch_flares()
    if action[7] > 0.5: self.countermeasures.launch_chaff()
    if action[8] > 0.5: self.countermeasures.activate_ecm()
    if action[9] > 0.5: self.countermeasures.deploy_decoys()
```

**Note:** The RL environment may use a different action space (e.g., Energy + Lift-Vector). The `apply_rl_action` method above is a legacy interface; modern training uses `ActionSpaceManager` in the RL module.

### State Representation

```python
def get_state_representation(self):
    """Comprehensive state dict for RL observations and debugging."""
    state = {
        "name": self.name,
        "id": self.id,
        "position": {"lat": ..., "lon": ..., "alt": ...},
        "yaw_deg": self.yaw_deg,
        "pitch_deg": self.control.pitch_deg,
        "roll_deg": self.control.roll_deg,
        "speed": self.speed,
        "missiles_loaded": len(self.missiles),
        "max_missiles": self.max_missiles,
        "flares": self.flares,
        "chaff": self.chaff,
        "ecm": self.ecm,
        "decoys": self.decoys,
    }

    # Add DLZ/NEZ/SQI for current target
    if self.target is not None:
        dlz = self.wez.compute_dlz(self.target)
        state["dlz"] = {...}         # DLZ ranges (r_min, r_tr, r_pi, r_aero, r_nez)
        state["dlz_zone"] = zone     # Current zone (R1, R2, R3, R4)
        state["nez_visible"] = bool  # NEZ display flag
        state["sqi"] = float       # Shot quality [0, 1]

    return state
```

---

## Flight Control System

**File:** `control/movement_control.py`

The `AircraftControlSystem` manages flight control and physics integration.

### Core Functionality

```python
class AircraftControlSystem:
    def __init__(self, parent):
        self.parent = parent
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.yaw_deg = parent.yaw_deg
        self.throttle = 1.0
        self.desired_yaw_deg = parent.yaw_deg
        self.desired_pitch_deg = 0.0
```

### Movement Update (Physics Integration)

```python
def update_movement(self, tick_secs: float):
    """Update aircraft position and attitude via physics."""
    # 1. Call physics to compute new state
    lat, lon, alt, spd, new_yaw, new_pitch, new_roll = self.parent.physics.compute_movement(
        self.parent.position,
        self.yaw_deg,
        self.desired_yaw_deg,
        self.pitch_deg,
        self.desired_pitch_deg,
        self.parent.speed,
        self.throttle,
        tick_secs
    )

    # 2. Update control system state
    self.pitch_deg = new_pitch
    self.yaw_deg = new_yaw
    self.roll_deg = new_roll
    self.parent.speed = np.clip(spd, self.parent.min_speed_mps, self.parent.max_speed_mps)

    # 3. CRITICAL: Sync aircraft attitude with control system
    self.parent.yaw_deg = new_yaw
    self.parent.pitch_deg = new_pitch
    self.parent.roll_deg = new_roll

    # 4. Update position
    self.parent.position.lat = lat
    self.parent.position.lon = lon
    self.parent.position.alt = alt

    # 5. Check for boundary violation
    if self._check_boundary_violation(self.parent.position):
        if not self.parent.boundary_violation_active:
            self.parent.boundary_violation_active = True
            self.parent.boundary_violation_countdown = 5  # 5-step grace period
            self.parent.removal_reason = "boundary_violation"

    # 6. Clamp altitude (lat/lon violations trigger removal, not clamping)
    self.parent.position = self._clamp_position_to_boundary(self.parent.position)
```

**Key Design Decision:** Lines 51-53 are **CRITICAL** for aircraft responsiveness. Previously, the aircraft's attitude (`parent.yaw_deg`, `parent.pitch_deg`, `parent.roll_deg`) was never updated from the control system, causing the aircraft to appear to fly straight even when commanded to turn. This sync ensures the aircraft entity reflects the physics-computed attitude.

### Boundary Management

**Lat/Lon Violations:**
- Detected in `_check_boundary_violation()`
- Triggers delayed removal (5-step countdown)
- Agent receives penalty during grace period
- Agent removed after countdown expires

**Altitude Violations:**
- Clamped to `[min_alt_m, max_alt_m]`
- No removal triggered (altitude is a soft constraint)

---

## Weapon Systems

### Weapon System Manager

**File:** `control/weapon_system.py`

```python
class AircraftWeaponSystem:
    def __init__(self, parent):
        self.parent = parent
        self.missile_types = parent.missile_types
        self.max_missiles = parent.max_missiles
        self.remaining_missiles = parent.max_missiles
        self.gun = GunSystem(parent, ...)  # Gun with ballistic projectiles
```

### Missile Firing Logic

**Comprehensive gating with single source of truth:**

```python
def fire_missile(self, sim, target, missile_cls):
    """
    Fire a missile with comprehensive gating.
    Returns: Missile object if successful, None otherwise.
    """
    # Gate 1: Inventory check
    if self.remaining_missiles <= 0:
        return None

    # Gate 2: Radar lock check
    if not self.parent.sensor.has_radar_lock(target):
        return None

    # Gate 3: Locked tracks verification
    locked_tracks = self.parent.sensor.get_locked_targets()
    if target.id not in locked_tracks:
        return None

    # Gate 4: FOV check (target within radar gimbal limits)
    if not self.is_target_in_fov(target):
        return None

    # All gates passed - create and launch missile
    missile = missile_cls(
        firing_time_s=sim.utc_time,
        target=target,
        source=self.parent,
        map_limits=self.parent.map_limits,
        group=self.parent.group
    )

    # Initialize with tracker state for better guidance
    tracker_state = locked_tracks[target.id]
    missile.initial_tracked_position_enu = tracker_state[:3]
    missile.initial_tracked_velocity_enu = tracker_state[3:6]
    missile.tracker_reference_pos = self.parent.position.copy()
    missile.tracking_lock = True

    # Add to simulation
    sim.add_unit(missile)
    self.parent.missiles.append(missile)
    self.remaining_missiles -= 1

    return missile
```

### Gun System

**File:** `control/gun_projectile.py`

The gun system provides realistic close-range combat capability.

#### GunSystem Class

```python
class GunSystem:
    def __init__(self, parent_aircraft, max_ammo=500,
                 muzzle_velocity_mps=1000.0, max_range_m=2000.0,
                 burst_size=10, burst_duration_s=0.1):
        self.parent = parent_aircraft
        self.max_ammo = max_ammo
        self.current_ammo = max_ammo
        self.muzzle_velocity_mps = muzzle_velocity_mps
        self.max_range_m = max_range_m
        self.burst_size = burst_size
        self.burst_duration_s = burst_duration_s
        self.min_fire_interval_s = 0.05  # Rate limiter
```

**Firing logic:**

```python
def start_firing(self, sim: Simulator, target_position: tuple = None):
    """Start a gun burst."""
    if not self.can_fire(sim.utc_time):
        return []

    # Fire burst of projectiles
    rounds_to_fire = min(self.burst_size, self.current_ammo)
    projectiles = []

    for i in range(rounds_to_fire):
        # Add realistic spread
        spread_x = (random() - 0.5) * 0.02 * 1000
        spread_y = (random() - 0.5) * 0.02 * 1000

        projectile = GunProjectile(
            firing_time=sim.utc_time,
            source_aircraft=self.parent,
            target_position=spread_target,
            source_velocity=self.parent.velocity,
            muzzle_velocity_mps=self.muzzle_velocity_mps,
            max_range_m=self.max_range_m,
            kill_radius_m=5.0
        )
        sim.add_unit(projectile)
        projectiles.append(projectile)

    self.current_ammo -= rounds_to_fire
    return projectiles
```

#### GunProjectile Class

**Ballistic physics with drag and gravity:**

```python
class GunProjectile(FlyingUnit):
    def __init__(self, firing_time, source_aircraft, target_position,
                 source_velocity, muzzle_velocity_mps=1000.0, max_range_m=2000.0,
                 group=None, kill_radius_m=5.0):

        # Initialize at source aircraft position
        super().__init__(
            name=f"Bullet_{source_aircraft.id}_{firing_time.timestamp():.0f}",
            position=source_aircraft.position.copy(),
            yaw_deg=source_aircraft.yaw_deg,
            speed=muzzle_velocity_mps
        )

        # Calculate initial velocity (aircraft velocity + muzzle velocity)
        aircraft_velocity = np.array([source_velocity.vx, vy, vz])
        target_direction = normalize(target_position - self.position)
        muzzle_velocity_vector = target_direction * muzzle_velocity_mps
        initial_velocity = aircraft_velocity + muzzle_velocity_vector

        self.velocity = Velocity(initial_velocity[0], [1], [2])
        self.physics = GunProjectilePhysics(...)
        self.kill_radius_m = kill_radius_m
```

**Update logic:**

```python
def update(self, tick_secs: float, sim: Simulator):
    # Update ballistic physics (drag + gravity)
    self.physics.update_position(self, tick_secs)
    self.traveled_distance_m += distance_step

    # Check expiration conditions
    if self.traveled_distance_m > self.max_range_m:
        sim.remove_unit(self)
        return []

    if self.position.alt < 0:  # Hit ground
        sim.remove_unit(self)
        return []

    # Check for hits against enemy aircraft
    for unit in sim.active_units.values():
        if unit.group != self.group and unit.type == "Aircraft":
            distance = self._calculate_distance_to_unit(unit)

            if distance <= self.kill_radius_m:
                # Hit detected!
                events.append(UnitDestroyedEvent(...))
                sim.remove_unit(self)
                break

    return events
```

**Ballistic physics:**

```python
class GunProjectilePhysics:
    def update_position(self, projectile, dt_s: float):
        # Get velocity vector
        velocity = np.array([projectile.velocity.vx, vy, vz])

        # Drag force (simplified)
        speed = norm(velocity)
        drag_force = -0.5 * drag_coeff * speed^2 * (velocity / speed)
        accel_drag = drag_force / mass

        # Gravity
        accel_gravity = [0, 0, -9.81]

        # Update velocity
        velocity += (accel_drag + accel_gravity) * dt
        projectile.velocity = Velocity(velocity[0], [1], [2])

        # Update position
        projectile.position.lat += velocity[1] * dt / 111000
        projectile.position.lon += velocity[0] * dt / (111000 * cos(lat))
        projectile.position.alt += velocity[2] * dt
```

**Example Configuration (F-22 M61A2 Vulcan):**

```python
gun_config = {
    "max_ammo": 480,              # 480 rounds
    "muzzle_velocity_mps": 1050.0, # 20mm at 1050 m/s
    "max_range_m": 1500.0,        # Effective range 1.5 km
    "burst_size": 50,             # 50 rounds per trigger
    "burst_duration_s": 0.5       # 0.5s burst (6000 rpm → 100 rps)
}
```

---

## Sensor Systems

### Sensor System Orchestrator

**File:** `systems/sensor.py`

```python
class AircraftSensorSystem:
    """Orchestrates all onboard sensor systems."""

    def __init__(self, parent):
        self.parent = parent
        self.radar = parent.radar                    # Active radar
        self.passive_radar = PassiveRadar(...)       # PRWR (radar warning)
        self.nez_calc = NoEscapeZoneCalculator(...)  # Tactical zones
        self.missile_warner = MissileWarner(...)     # Missile approach warning
        self.track_prioritizer = TrackPrioritySystem(...)

        # State containers
        self.sensor_tracks = []
        self.prioritized_tracks = []
        self.active_nez = {}
        self.passive_nez = {}
        self.warnings = []
```

### Update Cycle

```python
def update_sensor_data(self, sim, tick_secs):
    # 1. Update radar (detection + tracking)
    self.sensor_tracks = self.radar.update_for_sensors(
        tick_secs, sim, owner_position=self.parent.position,
        steer_h=5.0, steer_p=3.0
    )

    # 2. Passive radar (detect enemy emissions)
    for unit in sim.active_units.values():
        if unit.group != self.parent.group and hasattr(unit, 'emit_radar'):
            self.passive_radar.receive_emission(unit, self.parent, sim.elapsed_time_s)

    # 3. Compute NEZ zones for all enemies
    enemies = [u for u in sim.active_units.values() if u != self.parent]
    self.active_nez = {u.id: self.nez_calc.active_nez(u) for u in enemies}
    self.passive_nez = {u.id: self.nez_calc.passive_nez(u) for u in enemies}

    # 4. Update missile warnings
    self.missile_warner.check_for_new_missiles(sim.utc_time, sim)
    self.warnings = self.missile_warner.update(sim.utc_time)

    # 5. Prioritize tracks
    raw_tracks = [(state, cov) for tid, state, cov, *_ in self.sensor_tracks]
    self.prioritized_tracks = self.track_prioritizer.prioritize(raw_tracks)
```

### Target Selection

```python
def select_target(self, candidates, selection_value=None):
    """Select target from candidates list."""
    if not self.radar:
        return None

    # Prefer locked targets, sort by range
    locked_candidates = [t for t in candidates if self.has_radar_lock(t)]
    sorted_targets = (
        sorted(locked_candidates, key=lambda t: distance(self.parent, t))
        if locked_candidates
        else sorted(candidates, key=lambda t: distance(self.parent, t))
    )

    # Map selection_value [0, 1] to index
    n = len(sorted_targets)
    if n == 0:
        return None

    selection_value = clip(selection_value, 0.0, 1.0)
    index = int(selection_value * n)
    return sorted_targets[min(index, n-1)]
```

---

## Tactical Calculations

### No Escape Zone (NEZ) & Dynamic Launch Zone (DLZ)

**File:** `core/nez.py`

The `NoEscapeZoneCalculator` computes engagement zones for missile employment.

#### DLZ Dataclass

```python
@dataclass
class DLZ:
    r_min_m: float      # Minimum launch range (missile arming distance)
    r_tr_m: float       # Range at target turn (MAR - Maximum Abort Range)
    r_pi_m: float       # Range at peak intercept (non-maneuvering target)
    r_aero_m: float     # Aerodynamic max range (optimistic)
    r_nez_in_m: float   # NEZ inner boundary (typically r_min)
    r_nez_out_m: float  # NEZ outer boundary (typically r_tr)
```

**DLZ Zones:**
- **R1 (< r_min)**: Too close - missile may not arm
- **R2 (r_min to r_tr)**: NEZ - No Escape Zone (target cannot escape)
- **R3 (r_tr to r_pi)**: Valid launch zone (target can escape with hard turn)
- **R4 (> r_pi)**: Outside effective range (low Pk)

#### DLZ Computation

```python
def compute_dlz(self, target) -> DLZ:
    """Compute Dynamic Launch Zone for current engagement."""
    missile = self._get_best_missile(self.own)
    if not missile:
        # Conservative fallback
        return DLZ(1500, 15000, 30000, 40000, 1500, 15000)

    # Get engagement parameters
    own_speed = self.own.speed
    own_alt = self.own.position.alt
    tgt_speed = target.speed
    tgt_alt = target.position.alt
    rel_angle = self._relative_bearing(self.own, target)

    # Base kinematic range (considers geometry, closure, altitude)
    base_range = self._kinematic_range(
        missile, own_speed, tgt_speed, rel_angle, own_alt, tgt_alt
    )

    # RCS factor (seeker effectiveness)
    tgt_rcs = self._rcs(target)
    base_range *= self._rcs_factor(tgt_rcs, missile.seeker_sensitivity)

    # Compute DLZ edges
    r_min = missile.min_range_m  # e.g., 1500m
    r_tr = r_min + 0.60 * (base_range - r_min)  # 60% for turning target
    r_pi = r_min + 0.88 * (base_range - r_min)  # 88% for non-maneuvering
    r_aero = r_min + 1.04 * (base_range - r_min) # 104% optimistic

    return DLZ(r_min, r_tr, r_pi, r_aero, r_min, r_tr)
```

#### Kinematic Range Calculation

```python
def _kinematic_range(self, missile, own_speed, tgt_speed, rel_angle, own_alt, tgt_alt):
    """Compute kinematic range considering geometry and energy."""
    max_range = missile.max_range_m
    min_range = missile.min_range_m

    # Height bonus (own above target helps)
    height_bonus = ((own_alt - tgt_alt) / 1000.0) * 0.03
    range_with_height = max_range * (1.0 + height_bonus)

    # Closure factor
    closing_speed = own_speed * cos(radians(rel_angle)) - tgt_speed
    closing_factor = 0.20 * clip(closing_speed / 400.0, -1.0, 1.0)

    base = range_with_height * (1.0 + closing_factor)
    return clip(base, min_range, max_range * 1.3)
```

### SQI (Time In Pole Window Indicator)

**Shot quality metric [0, 1]:**

```python
def sqi(self, own, tgt, missile=None, dlz=None) -> float:
    """Compute instantaneous shot quality metric."""
    if dlz is None:
        dlz = self.compute_dlz(tgt)

    # Distance score (1.0 near r_min → 0.0 near r_aero)
    d = self._slant_range_m(own, tgt)
    phi_d = clip(1.0 - (d - dlz.r_min_m) / (dlz.r_aero_m - dlz.r_min_m), 0, 1)

    # Closure rate (normalized)
    rel_angle = self._relative_bearing(own, tgt)
    Vc = own.speed * cos(radians(rel_angle)) - tgt.speed
    Vc_n = clip(Vc / 400.0, -1.0, 1.0)

    # Aspect angle
    cos_aspect = cos(radians(rel_angle))

    # Altitude factor (density ratio)
    rho_ratio = get_density(own.position.alt) / rho0

    # Logistic mapping to [0, 1]
    x = -1.4 + 3.0*phi_d + 1.2*Vc_n + 0.8*cos_aspect + 0.25*(rho_ratio - 1.0)
    return 1.0 / (1.0 + exp(-x))
```

**Interpretation:**
- **SQI > 0.7**: Excellent shot (high Pk)
- **SQI 0.5-0.7**: Good shot
- **SQI 0.3-0.5**: Marginal shot
- **SQI < 0.3**: Poor shot (wasteful)

---

## Countermeasures

**File:** `control/countermeasure.py`

```python
class AircraftCountermeasureSystem:
    def __init__(self, parent):
        self.parent = parent
        self.flares = parent.flares
        self.chaff = parent.chaff
        self.ecm = parent.ecm
        self.decoys = parent.decoys

    def launch_flares(self):
        """Deploy IR countermeasures."""
        if self.flares > 0:
            self.flares -= 1
            self.parent.flare_deployed = True

    def launch_chaff(self):
        """Deploy radar countermeasures."""
        if self.chaff > 0:
            self.chaff -= 1
            self.parent.chaff_deployed = True

    def activate_ecm(self):
        """Activate electronic countermeasures."""
        if self.ecm > 0:
            self.ecm -= 1
            self.parent.ecm_active = True

    def deploy_decoys(self):
        """Deploy towed decoys or expendable jammers."""
        if self.decoys > 0:
            self.decoys -= 1
            self.parent.decoys_deployed = True
```

**Countermeasure Types:**
- **Flares**: IR signature decoys (defeats IR-guided missiles like AIM-9)
- **Chaff**: Radar-reflective strips (defeats radar-guided missiles)
- **ECM**: Active jamming (noise or deception)
- **Decoys**: Towed decoys (TALD, ADM-160) or expendable jammers

---

## Aircraft Type Implementations

### F-22 Raptor

**File:** `types/f22.py`

```python
class F22(Aircraft):
    @dataclass
    class Config:
        # Physics (geometry corrected)
        mass_kg: float = 19700.0               # Empty mass
        reference_area_m2: float = 78.0        # ~840 ft²
        aspect_ratio: float = 2.36             # b²/S = 13.56² / 78.04
        oswald_e: float = 0.82
        max_speed_mps: float = 680.0           # ~Mach 2.0 at altitude
        n_max: float = 9.0
        stall0_mps: float = 75.0

        # Flight envelope
        min_speed_mps: float = 80.0
        max_climb_angle_deg: float = 70.0
        min_alt_m: float = 0.0
        max_alt_m: float = 20000.0

        # Radar (X-band AESA)
        radar_horizontal_fov_deg: float = 120.0
        radar_vertical_fov_deg: float = 60.0
        radar_max_range_m: float = 230_000.0
        radar_frequency_hz: float = 9.5e9
        radar_tx_power_w: float = 25e3
        radar_antenna_gain_db: float = 40.0
        radar_snr_threshold_db: float = 8.0
        radar_beam_rate_hz: float = 10.0
        radar_beam_rate_p_hz: float = 8.0

        # RCS (Very Low Observable)
        rcs: float = 0.0001  # Baseline RCS (highly aspect-dependent)

        # Weapons
        missile_types: Tuple = (AIM120_AMRAAM,)
        max_missiles: int = 8
        flares: int = 2
        chaff: int = 2
        ecm: int = 2
        decoys: int = 2

        # Gun (M61A2 Vulcan)
        gun_config: dict = {
            "max_ammo": 480,
            "muzzle_velocity_mps": 1050.0,
            "max_range_m": 1500.0,
            "burst_size": 50,
            "burst_duration_s": 0.5
        }
```

**RCS Pattern (Aspect-Dependent Stealth):**

```python
self.rcs_pattern = {
    "front_floor": 0.20,   # Nose-on: 0.2 × σ0 (very low)
    "tail_floor": 0.45,    # Tail-on: 0.45 × σ0 (low but higher than nose)
    "n_az": 1.5,           # Sharper beam peak (steeper azimuth dependence)
    "k_top": 0.15,         # Dorsal (top): modest RCS
    "k_bottom": 0.35,      # Ventral (bottom): higher RCS (inlets/bays)
    "n_el": 1.7,           # Elevation shaping exponent
    "sigma_min": 0.05,     # Minimum RCS fraction
    "sigma_max": 5.0,      # Maximum RCS fraction (beam aspect)
}
```

**Other Aircraft Types:**
- **F-35 Lightning II** (`types/f35.py`): Multirole VLO, shorter range radar
- **Su-57 Felon** (`types/su57.py`): Russian 5th gen, higher RCS than F-22
- **Eurofighter Typhoon** (`types/eurofighter.py`): 4.5 gen, excellent agility
- **Debug Plane** (`types/debug_plane.py`): Simplified for testing

---

## Integration Points

### With Physics Module

**Flight dynamics:**

```python
# In AircraftControlSystem.update_movement()
lat, lon, alt, spd, new_yaw, new_pitch, new_roll = self.parent.physics.compute_movement(
    position=self.parent.position,
    yaw_deg=self.yaw_deg,
    desired_yaw_deg=self.desired_yaw_deg,
    pitch_deg=self.pitch_deg,
    desired_pitch_deg=self.desired_pitch_deg,
    speed=self.parent.speed,
    throttle=self.throttle,
    dt=tick_secs
)
```

**Energy state:**

```python
# Physics computes Ps (specific energy rate)
Ps_current = self.parent.physics.get_specific_energy_rate()
Ps_min, Ps_max = self.parent.physics.get_ps_envelope(n_cmd)
```

### With Radar Module

**Radar updates:**

```python
# In AircraftSensorSystem.update_sensor_data()
self.sensor_tracks = self.radar.update_for_sensors(
    tick_secs=tick_secs,
    sim=sim,
    owner_position=self.parent.position,
    steer_h=5.0,  # Horizontal beam steering rate
    steer_p=3.0   # Vertical beam steering rate
)
```

**Lock management:**

```python
# Check radar lock
has_lock = self.parent.sensor.has_radar_lock(target)

# Get locked targets (multi-lock capable)
locked_targets = self.parent.sensor.get_locked_targets()  # Returns dict or set
```

### With Missiles Module

**Missile launch:**

```python
# In AircraftWeaponSystem.fire_missile()
missile = missile_cls(
    firing_time_s=sim.utc_time,
    target=target,
    source=self.parent,
    map_limits=self.parent.map_limits,
    group=self.parent.group
)

# Initialize with tracker state
missile.initial_tracked_position_enu = tracker_state[:3]
missile.initial_tracked_velocity_enu = tracker_state[3:6]
missile.tracker_reference_pos = self.parent.position.copy()

sim.add_unit(missile)
```

### With Simulator

**Registration:**

```python
# Aircraft registers on creation
sim.add_unit(aircraft)
```

**Update cycle:**

```python
# Simulator calls aircraft.update() each tick
for unit in sim.active_units.values():
    events = unit.update(tick_secs, sim)
```

**Removal:**

```python
# Boundary violation triggers removal
if aircraft.should_be_removed:
    sim.remove_unit(aircraft)
```

### With RL Environment

**Action application:**

```python
# RL environment calls apply_rl_action()
aircraft.apply_rl_action(action_vector, simulator)
```

**Observation extraction:**

```python
# RL environment gets state representation
state = aircraft.get_state_representation()

# Includes: position, attitude, speed, weapons, DLZ, SQI, etc.
```

**Reward signals:**

```python
# Boundary violation penalty
if aircraft.boundary_violation_active:
    reward += aircraft.boundary_violation_penalty_per_step  # -10.0 per step
```

---

## Configuration System

### Dataclass-Based Configuration

All aircraft types use dataclass configs for type safety and documentation:

```python
@dataclass
class Config:
    # Physics parameters
    mass_kg: float = 20000.0
    reference_area_m2: float = 27.87
    aspect_ratio: float = 4.0
    oswald_e: float = 0.78
    max_speed_mps: float = 680.0
    n_max: float = 9.0
    stall0_mps: float = 60.0

    # Radar parameters
    radar_max_range_m: float = 150_000.0
    radar_tx_power_w: float = 10_000.0
    radar_antenna_gain_db: float = 30.0

    # Weapons
    missile_types: Tuple = (AIM120_AMRAAM,)
    max_missiles: int = 8

    # Countermeasures
    flares: int = 60
    chaff: int = 120
    ecm: int = 1
    decoys: int = 2
```

**Usage:**

```python
cfg = F22.Config()
cfg.min_alt_m = custom_min_alt
cfg.max_alt_m = custom_max_alt

aircraft = F22(
    position=start_pos,
    yaw_deg=90.0,
    speed_mps=250.0,
    group="blue",
    map_limits=map_limits,
    min_alt_m=cfg.min_alt_m,
    max_alt_m=cfg.max_alt_m
)
```

---

## Recent Enhancements (October 2025)

### Critical Aircraft Responsiveness Fix

**Problem:** Aircraft appeared to fly straight up/forward, ignoring turn commands from RL policy.

**Root Cause:** Aircraft attitude (`yaw_deg`, `pitch_deg`, `roll_deg`) was never updated from the control system. The physics computed new attitudes, but the aircraft entity retained stale values.

**Solution:** Added critical sync lines in `movement_control.py:51-53`:

```python
# CRITICAL: Update aircraft attitude to match control system
self.parent.yaw_deg = new_yaw
self.parent.pitch_deg = new_pitch
self.parent.roll_deg = new_roll
```

This ensures the aircraft entity's attitude reflects the physics-computed values, making the aircraft responsive to control commands.

### Boundary Management System

**Delayed removal instead of immediate termination:**

```python
# When boundary violation detected
if not self.parent.boundary_violation_active:
    self.parent.boundary_violation_active = True
    self.parent.boundary_violation_countdown = 5  # 5-step grace period
    self.parent.removal_reason = "boundary_violation"

# In aircraft.update()
if self.boundary_violation_active:
    self.boundary_violation_countdown -= 1
    if self.boundary_violation_countdown <= 0:
        self.should_be_removed = True  # Remove after countdown
```

**Benefits:**
- Extended learning signal during violation period
- Continuous penalties guide policy away from boundaries
- Independent per-agent tracking (no team-wide termination)

### Gun System Implementation

**Realistic close-range combat:**

- Ballistic projectile physics with drag and gravity
- Ammunition management and burst control
- Lead angle calculation for moving targets
- Configurable per aircraft type (e.g., F-22 M61A2: 480 rounds, 6000 rpm)

### Enhanced NEZ/DLZ/SQI

**Comprehensive tactical overlays:**

- Full DLZ computation with R1/R2/R3/R4 zones
- NEZ visualization (No Escape Zone)
- SQI shot quality metric [0, 1]
- Integrated into state representation for RL observations

---

## Common Patterns and Best Practices

### 1. Subsystem Access

Always access subsystems through aircraft instance:

```python
# Correct
aircraft.control.set_throttle(0.8)
aircraft.weapons.fire_missile(sim, target, AIM120_AMRAAM)
aircraft.sensor.has_radar_lock(target)

# Incorrect - don't bypass aircraft
control_system.set_throttle(0.8)  # Missing context
```

### 2. Radar Lock Before Fire

Always verify radar lock before weapon employment:

```python
# Weapon system internally checks
if not self.parent.sensor.has_radar_lock(target):
    return None  # Veto fire command
```

### 3. Boundary Checking

Use delayed removal for learning signals:

```python
# Check violation
if self._check_boundary_violation(position):
    if not self.parent.boundary_violation_active:
        self.parent.boundary_violation_active = True
        self.parent.boundary_violation_countdown = 5

# Don't immediately remove - give grace period
```

### 4. State Representation

Include tactical overlays for RL:

```python
state = aircraft.get_state_representation()
# Returns: position, attitude, DLZ, SQI, weapons, countermeasures
```

### 5. Configuration Inheritance

Use dataclass configs with inheritance:

```python
@dataclass
class Config:
    mass_kg: float = 20000.0
    # ... base parameters

cfg = Config()
cfg.max_alt_m = custom_value  # Override specific fields
aircraft = Aircraft(config=cfg)
```

---

## Testing Strategy

### Unit Tests

**Movement control:**
- Attitude sync validation
- Boundary violation detection
- Altitude clamping

**Weapon systems:**
- Missile firing gates
- Gun projectile ballistics
- Ammunition management

**Sensor systems:**
- Radar lock logic
- Passive radar detection
- Missile warning

### Integration Tests

**Full aircraft system:**
- Physics integration
- Radar detection and tracking
- Weapon employment
- Countermeasure deployment

**Multi-aircraft scenarios:**
- 1v1 engagements
- DLZ/SQI accuracy
- Boundary handling

### Physics Validation

**Flight behavior:**
- Turn rates vs. speed/altitude
- Stall characteristics
- Envelope protection

**Weapon ballistics:**
- Gun projectile trajectories
- Hit detection accuracy
- Range limitations

---

## Performance Considerations

### Computational Costs

**Per aircraft per tick:**
- Physics computation: ~0.5ms
- Radar update: ~2-5ms (depends on contact count)
- Sensor fusion: ~0.2ms
- NEZ/DLZ calculation: ~0.1ms per target

**Optimization strategies:**
- Cache DLZ computations (update every N ticks)
- Limit radar beam steering rate
- Prioritize tracks before detailed processing

### Memory Usage

**Per aircraft:**
- Aircraft entity: ~5 KB
- Radar state: ~50 KB (tracks, detections)
- Physics state: ~2 KB
- Total: ~60 KB per aircraft

**Scalability:**
- 100 aircraft: ~6 MB
- 1000 aircraft: ~60 MB (practical limit for single machine)

---

## Future Enhancements

### Short-term

- [ ] Energy + Lift-Vector control integration (replace yaw/pitch deltas)
- [ ] Advanced countermeasure modeling (effectiveness vs. missile seekers)
- [ ] Improved RWR (Radar Warning Receiver) with threat prioritization

### Medium-term

- [ ] Datalink track fusion (multi-aircraft cooperative tracking)
- [ ] Electronic warfare interactions (ECM vs. radar)
- [ ] Helmet-mounted sight for off-boresight missile launches

### Long-term

- [ ] Multi-spectral sensors (EO/IR, ESM)
- [ ] Network-centric warfare (distributed kill chains)
- [ ] Swarm tactics for autonomous agents

---

## References

**Flight Dynamics:**
- Shaw, Robert L. "Fighter Combat: Tactics and Maneuvering" (1985)
- Stevens & Lewis, "Aircraft Control and Simulation" (2003)

**Radar Systems:**
- Skolnik, Merrill I. "Introduction to Radar Systems" (2001)
- Richards et al., "Principles of Modern Radar" (2010)

**Missile Guidance:**
- Zarchan, Paul. "Tactical and Strategic Missile Guidance" (2012)
- Yanushevsky, Rafael. "Modern Missile Guidance" (2007)

**Tactical Decision Making:**
- USAF Tactics, Techniques, and Procedures (TTP) - BVR Employment
- DLZ and NEZ concepts from academic air combat literature
