# Radar Module - Context

## Purpose
Implements realistic radar simulation including detection, tracking, electronic warfare (EW), and multi-platform data fusion. Models radar cross-section (RCS), atmospheric effects, electronic countermeasures (ECM), and cooperative engagement.

## Architecture Overview

```
radar/
├── radar.py              # Main Radar class and update loop
├── core/                 # Core utilities and building blocks
│   ├── utils.py         # Coordinate transformations (ENU/geodetic)
│   ├── lut.py           # Detection probability lookup tables
│   ├── data_link.py     # Multi-platform data fusion
│   └── parameter_policy.py  # Adaptive radar parameters
├── obs/                  # Observation generation
│   ├── observation.py   # Raw detection generation
│   ├── cluster.py       # Detection clustering (DBSCAN-like)
│   └── data_fusion.py   # Multi-radar data fusion
├── tracking/            # Target tracking and filtering
│   ├── tracker.py       # TrackerManager (orchestrates tracking)
│   ├── filter/          # Kalman filters (CV, CT, IMM)
│   └── helpers/         # ENU utils, measurement builder, track manager
├── lock/                # Target lock management
│   ├── base.py          # Base lock controller
│   ├── aircraft.py      # Aircraft radar lock
│   └── missile.py       # Missile seeker lock
├── ew/                  # Electronic Warfare
│   ├── ew_world.py      # Global EW coordinator
│   └── ecm_emitter.py   # ECM jammer implementation
├── units/               # Platform-specific radars
│   ├── aircraft.py      # Fighter radar systems
│   └── missile.py       # Missile seeker radars
└── graphics/            # Visualization and analysis scripts (NEW)
    ├── detection_probability_heatmap.py  # Heatmap visualization
    ├── detection_probability_f22_variants.py  # Aircraft comparison analysis
    ├── rcs_altitude_angle.py  # RCS pattern analysis
    └── rcs_horizontal_angle.py  # RCS horizontal patterns
```

## Key Components

### 1. Main Radar Class (`radar.py`)

The `Radar` class orchestrates all radar functionality:

```python
radar = Radar(
    horizontal_fov_deg=60,      # Azimuth field of view
    vertical_fov_deg=30,        # Elevation field of view
    max_range_m=120000,         # Maximum detection range
    radar_frequency_hz=10e9,    # X-band (10 GHz)
    tx_power_w=15000,           # Transmit power (15 kW)
    antenna_gain_db=40,         # Antenna gain
    snr_threshold_db=10,        # Detection threshold
    jam_susceptible=True        # Affected by ECM (aircraft only)
)
```

**Update Cycle:**
1. Generate raw detections from targets (observation.py)
2. Apply ECM effects (jamming, deception) if jam_susceptible
3. Convert detections to geodetic coordinates
4. Fuse with datalink partners (data_link.py)
5. Cluster detections (cluster.py)
6. Update tracks (tracker.py)
7. Update lock controller

### 2. Coordinate Systems (`core/utils.py`)

**Critical Functions:**
- `geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)` → ENU vector
  - Converts geodetic (lat/lon/alt) to local East-North-Up Cartesian
  - **Convention**: Returns (E, N, U) in meters relative to reference point

- `enu_to_geodetic(enu_vec, ref_lat, ref_lon, ref_alt)` → (lat, lon, alt)
  - Inverse transformation from ENU to geodetic

- `to_cart(az_deg, el_deg, range_m)` → (E, N, U)
  - Spherical to Cartesian conversion
  - **Azimuth Convention**: 0°=North, 90°=East (geodetic, clockwise)
  - **Changed in commit 30808bb** from mathematical (0°=East, CCW)

**Other Utilities:**
- `_angles_dist()`: Computes relative azimuth/elevation/range
- `_effective_rcs()`: RCS with aspect dependence
- `_doppler()`: Doppler shift calculation

### 3. Detection & Observation (`obs/`)

#### `observation.py` - RadarObsGenerator
Generates raw detections using radar equation:
```python
SNR = (P_tx * G_tx * G_rx * σ * λ²) / ((4π)³ * R⁴ * N)
P_det = probability from LUT(range, RCS)
```

Features:
- RCS aspect dependence (nose/beam/tail)
- Doppler computation
- FOV gating (azimuth/elevation)
- Probabilistic detection

#### `cluster.py` - Clusterer
Groups nearby detections using angular/range gating:
- Clusters detections within angular_resolution_deg
- Returns merged clusters with average az/el/d
- Handles both real targets and ECM ghosts

#### `data_fusion.py` - DataLink
Multi-platform data fusion:
- Combines detections from friendly radars
- Coordinate transformation between platforms
- Handles both own detections and remote observations

### 4. Tracking System (`tracking/`)

#### `tracker.py` - TrackerManager
Main tracking orchestrator:
```python
tracker = TrackerManager(assoc_dist=1000.0)
tracks = tracker.update_tracks(clusters, dt, default_ref, own_yaw_deg)
```

**Track State:**
- Position: (E, N, U) in ENU frame
- Velocity: (vE, vN, vU) in ENU frame
- Covariance: 6x6 matrix

**Key Features:**
- Auto-recentering: Shifts ENU reference when drift > 1km
- Per-track ENU references (prevents large coordinate values)
- Proper rotation when recentering (R @ state + Δ)
- Anisotropic measurement noise (xy >> z)
- Z-bias correction via EWMA

#### `filter/` - Kalman Filters
- `constant_velocity_filter.py`: CV model (default)
- `coordinated_turn_filter.py`: CT model for maneuvering targets
- `imm_filter.py`: Interacting Multiple Model (future)

#### `helpers/`
- `enu_utils.py`: ENU basis vectors and rotation matrices
- `recenter_logic.py`: Track recentering transformations
- `measurement_builder.py`: Converts clusters to ENU measurements
- `track_manager.py`: Track creation/pruning/output
- `noise_and_bias.py`: Measurement noise models

**Important Fix (Oct 2025):**
Fixed bug in `recenter_logic.py` where `geodetic_to_enu()` arguments were swapped:
```python
# CORRECT:
delta = geodetic_to_enu(
    new_ref.lat, new_ref.lon, new_ref.alt,  # Point to convert
    old_ref.lat, old_ref.lon, old_ref.alt   # Reference frame
)
```

### 5. Lock Management (`lock/`)

#### `base.py` - BaseLockController
- Tracks locked targets
- Handles lock acquisition/breaking
- Priority management

#### `aircraft.py` - AircraftRadarLockController
- TWS (Track While Scan) mode
- Single-target lock
- Multi-target tracking

#### `missile.py` - MissileLockController
- Terminal guidance lock
- High update rate
- Immune to ECM (jam_susceptible=False)

### 6. Electronic Warfare (`ew/`)

#### `ew_world.py` - EWWorld
Global EW coordinator that manages all jamming interactions:

```python
class EWWorld:
    def collect_incoming(self, victim_radar, t):
        """
        Collects jamming effects on victim radar.

        Returns:
            (jammer_info, ghosts) where:
            - jammer_info: List of (az_deg, el_deg, J_power)
            - ghosts: List of synthetic detections (is_deception=True)
        """
```

**Functionality:**
- Accumulates jamming power from multiple enemies
- Computes per-jammer azimuth/elevation for angular weighting
- Collects deception returns (ghosts) from DRFM systems
- Cooperative jamming assist (schedule_assist, future)

**Usage in `radar.py`:**
```python
if self.jam_susceptible and sim.ew_world:
    jammer_info, ghosts = sim.ew_world.collect_incoming(self, t)

    # Degrade real detections
    for detection in own_detections:
        J_total = sum_weighted_jamming(detection, jammer_info)
        SNR_eff = SNR / (1 + J/N)
        if SNR_eff < threshold:
            drop_detection()

    # Inject ghost detections
    own_detections.extend(ghosts)
```

#### `ecm_emitter.py` - ECMEmitter
Individual jammer on each aircraft:

```python
ecm = ECMEmitter(
    owner=aircraft,
    erp_w=10000,                    # Effective Radiated Power (10 kW)
    bw_hz=1e9,                      # Jamming bandwidth (1 GHz)
    hop_period_s=0.1,               # Frequency hopping
    techniques={"drfm_multi_false"} # DRFM deception
)
```

**Jamming Power Computation:**
```python
J = (ERP * G_rx * λ²) / ((4π)² * R² * L)
```

**Deception Techniques:**
- **DRFM Multi-False**: Generates 3-5 ghost targets near jammer
  - Random angular offsets (±5° az, ±3° el)
  - Range spread (±10%)
  - Realistic Doppler jitter (±30 m/s)
  - Plausible SNR (threshold + 2-10 dB)

**Ghost Detection Format:**
```python
ghost = {
    "az": ghost_az,                 # Azimuth (degrees)
    "el": ghost_el,                 # Elevation (degrees)
    "d": ghost_range,               # Range (meters)
    "dop": dop_hz,                  # Doppler (Hz)
    "snr_db": ghost_snr_db,        # SNR (dB)
    "is_deception": True,           # Flag for tracker
    "engagement_id": jammer.id,     # Jammer unit ID
    "jammer_id": jammer.id,         # Same
    "T": None                       # No ground truth
}
```

**Tracking Integration:**
- Ghosts flow through clustering and tracking like real contacts
- Tracker marks them with `is_deception=True`
- ECCM: Bearing-invariance test detects ghosts (yaw changes but bearing doesn't)
- Marked ghosts become non-engageable

### 7. Graphics & Visualization (`graphics/`) (NEW - November 2025)

Analysis and visualization tools for radar performance and RCS characteristics:

#### `detection_probability_heatmap.py`
Generates 2D heatmaps showing detection probability across range and RCS:
- Visualizes radar equation results
- Shows detection probability (0-100%) as color intensity
- Supports multiple aircraft types
- Outputs PNG heatmaps for analysis

#### `detection_probability_f22_variants.py`
Comparative analysis of detection probability across aircraft variants:
- F-22 vs F-35 vs Eurofighter vs Su-57
- Shows range vs aspect angle effects
- Helps validate RCS modeling
- Useful for training scenario design

#### `rcs_altitude_angle.py` & `rcs_horizontal_angle.py`
RCS pattern visualization:
- Aspect-dependent RCS models
- Altitude and angle effects
- Scattering center modeling
- Used for radar system tuning

### 8. Data Link & Fusion (`core/data_link.py`)

Multi-platform cooperative engagement:
- Share detections across friendly radars
- Triangulation for improved accuracy
- Distributed tracking
- Tactical data link protocols

## Key Algorithms

### Detection Processing
1. **Radar Equation**: SNR = f(range, RCS, power, gain)
2. **Probability Model**: P_det from LUT(range, RCS)
3. **FOV Gating**: Check azimuth/elevation limits
4. **Doppler Processing**: v_radial from velocity projection

### Tracking Algorithms
1. **Constant Velocity (CV)**: Default linear motion model
2. **Coordinated Turn (CT)**: For maneuvering targets (future)
3. **IMM**: Multiple model tracking (future)

### Data Association
- **Nearest Neighbor**: Simple 1:1 assignment
- Track ID from ground truth when available
- Hash-based ID for ghosts (per jammer, per position)

### ENU Coordinate Management
1. **Track Initialization**: Use measurement reference
2. **Recentering**: When drift > 1km, transform state
   ```python
   x_new = R @ x_old + Δ
   v_new = R @ v_old
   P_new = S @ P_old @ S^T  # S = blkdiag(R, R)
   ```
3. **Export**: Rotate tracks into missile ENU for guidance

## Integration Points

### With Aircraft Systems
- **Sensors**: Primary detection system
- **ECM**: Owned by aircraft, called via sim.ew_world
- **Weapons**: Target designation for missiles
- **Navigation**: Position reference for tracking

### With Simulator
- **Unit Registry**: Access to all active units
- **EW World**: Global ECM coordination
- **Event System**: Detection/lock events
- **Time Management**: Simulation tick_secs

### With RL Environment
- **Observation Space**: Radar tracks in observation
- **Action Space**: Scan patterns, lock commands (future)
- **Reward Shaping**: Detection/tracking bonuses
- **Info Dict**: Track quality metrics

## Coordinate System Conventions

### Geodetic
- **Latitude**: Degrees, -90 (S) to +90 (N)
- **Longitude**: Degrees, -180 (W) to +180 (E)
- **Altitude**: Meters above WGS84 ellipsoid

### ENU (East-North-Up)
- **East**: +X in meters
- **North**: +Y in meters
- **Up**: +Z in meters
- **Origin**: Reference point (lat, lon, alt)

### Azimuth/Elevation
- **Azimuth**: 0°=North, 90°=East (clockwise, geodetic convention)
  - **Changed in commit 30808bb** from mathematical convention
- **Elevation**: 0°=horizontal, +90°=up, -90°=down
- **Range**: Meters

## Configuration Examples

### Fighter Radar (F-22 APG-77)
```python
radar = Radar(
    horizontal_fov_deg=120,     # Wide scan
    vertical_fov_deg=60,
    max_range_m=200000,         # 200 km
    radar_frequency_hz=10e9,    # X-band
    tx_power_w=15000,           # 15 kW
    antenna_gain_db=40,
    snr_threshold_db=10,
    jam_susceptible=True        # Affected by ECM
)
```

### Missile Seeker (AIM-120 AMRAAM)
```python
radar = Radar(
    horizontal_fov_deg=60,      # Narrower FOV
    vertical_fov_deg=60,
    max_range_m=20000,          # 20 km
    radar_frequency_hz=16e9,    # Ku-band
    tx_power_w=1000,            # 1 kW
    antenna_gain_db=35,
    snr_threshold_db=8,
    jam_susceptible=False       # Immune to ECM
)
```

### ECM Jammer
```python
ecm = ECMEmitter(
    owner=aircraft,
    erp_w=10000,                        # 10 kW ERP
    bw_hz=1e9,                          # 1 GHz bandwidth
    hop_period_s=0.1,                   # 100ms hopping
    techniques={"drfm_multi_false"},    # DRFM deception
    assist_ahead_deg=45,                # Assist sector
    assist_behind_deg=45
)
```

## Testing

### Unit Tests
- `tests/radar/` - Component tests
- `tests/test_tracker_error.py` - End-to-end tracking validation

### Key Test Scenarios
1. **Detection Probability**: Range vs RCS curves
2. **Tracking Accuracy**: Position/velocity errors
3. **ENU Recentering**: Coordinate transform correctness
4. **ECM Effectiveness**: Jamming degradation, ghost injection
5. **Data Fusion**: Multi-platform coordination

## Recent Changes (October 2025)

### Coordinate System Fix
- **Issue**: Swapped arguments in `geodetic_to_enu()` causing tracker errors >2000m
- **Root Cause**: When RECENTER_THRESH_M changed from 10km to 1km, bug exposed
- **Fix**: Corrected argument order in `recenter_logic.py` lines 40-42 and 110-112
- **Impact**: Tracker now accurate with 1km recentering

### EW Integration
- **Added**: `ew/` subfolder with `ew_world.py` and `ecm_emitter.py`
- **Feature**: Aircraft-only jamming (missile seekers immune)
- **Capability**: Noise jamming + DRFM deception (multi-ghost)
- **ECCM**: Bearing-invariance test for ghost detection

## Performance Considerations

- **GPU Acceleration**: Clustering on CUDA when available
- **Caching**: Detections, clusters, tracks cached per update
- **Lazy Evaluation**: Tracks only computed when requested
- **Coordinate Transforms**: Expensive, minimize calls
- **Track Pruning**: Remove stale tracks after 5 missed updates

## Known Limitations

### RCS Modeling
- Simplified aspect dependence (azimuth only)
- No frequency dependence
- No polarization effects
- Single scattering center per target

### ECM
- Simplified angular weighting (cosine taper)
- No coherent jamming techniques
- No chaff/flare modeling
- Binary frequency overlap model

### Tracking
- Only CV filter implemented (CT/IMM planned)
- Simple nearest-neighbor association
- No MHT or JPDA
- No track smoothing/retrospective correction

## Future Enhancements

### Electronic Warfare
- Cross-eye jamming
- Chaff clouds and expendables
- Towed decoys
- Cooperative jamming with beam steering

### Tracking
- IMM filter for maneuvering targets
- JPDA/MHT for dense environments
- Track smoothing (RTS smoother)
- Track quality metrics

### Data Fusion
- Track-to-track fusion (not just measurement)
- Optimal sensor tasking
- Information-driven sensing
- Network-centric warfare

### Environmental Effects
- Atmospheric ducting
- Multipath propagation
- Weather clutter
- Ground clutter modeling

## References

- Skolnik, M. "Radar Handbook" (3rd Ed)
- Bar-Shalom, Y. "Estimation with Applications to Tracking"
- Adamy, D. "EW 101: A First Course in Electronic Warfare"
- Richards, M. "Fundamentals of Radar Signal Processing"
