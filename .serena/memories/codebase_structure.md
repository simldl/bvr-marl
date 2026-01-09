# Codebase Structure

## Directory Layout

### Core Simulation Components
- **`simulator/`** - Core simulation engine and utilities
  - `core/` - Events, hit calculation, substepping, units base classes
  - `utils/` - Angles, geodesics, map limits
  - `simulator.py` - Main Simulator class

- **`physics/`** - Physics models for aircraft and missiles
  - `physics.py` - Base physics classes, atmosphere model, force calculations
  - `aircraft.py` - AircraftPhysics with energy management
  - `missiles.py` - Missile physics
  - `flying_objects.py` - Generic flying object physics
  - `afterburner.py` - Afterburner thrust modeling

- **`aircrafts/`** - Aircraft implementation and control systems
  - `aircraft.py` - Main Aircraft class
  - `core/` - NEZ calculator, target prioritization
  - `control/` - Movement control, weapon systems, countermeasures, gun projectiles
  - `systems/` - Sensors, passive radar, missile warner, observation helpers, metrics
  - `types/` - Specific aircraft models (F-22, F-35, Eurofighter, Su-57, DebugPlane)

- **`missiles/`** - Missile systems and guidance algorithms
  - `missile.py` - Main Missile class
  - `core/` - Engine, movement, phases
  - `guidance/` - Base guidance, proportional navigation, lead, loft, terminal, FOV capture, direct
  - `fox1/` - Semi-Active Radar Homing (SARH) missiles - Sparrow, Skyflash
  - `fox2/` - Infrared (IR) missiles - Sidewinder, Python
  - `fox3/` - Active Radar Homing (ARH) missiles - AMRAAM, Meteor, R-77-1, R-37M, K-77M

- **`radar/`** - Radar simulation and tracking systems
  - `radar.py` - Main radar classes
  - `core/` - Data link, LUT, parameter policy, utils
  - `ew/` - Electronic warfare, ECM emitters, EW world
  - `lock/` - Lock management for aircraft and missiles
  - `obs/` - Observation clustering, data fusion
  - `tracking/` - Tracking filters (CV, CT, IMM), track manager, measurement builder
  - `units/` - Aircraft and missile radar units
  - `graphics/` - Visualization and analysis scripts for radar performance (RCS patterns, detection probability heatmaps)

### AI/ML Components
- **`reinforcement_learning/`** - RL environments, networks, and training
  - `train.py` - Main training script
  - `train_config.yaml` - Training configuration
  - `environment/` - Multi-agent RL environments
  - `networks/` - Neural network architectures (policy, value)
  - `rl_utils/` - Training utilities, evaluation, logging

- **`tacview/`** - Tactical view logging for scenario analysis and replay

### Utilities & Testing
- **`visualization/`** - Real-time visualization and plotting tools
- **`tests/`** - Comprehensive test suite
  - Component-specific folders: `aircrafts/`, `missiles/`, `physics/`, `radar/`, `reinforcement_learning/`, `simulator/`
  - `mocks/` - Test doubles and mock objects
  - Integration and unit tests for all major features

### Configuration
- **`.claude/`** - Claude Code IDE configuration
- **`.vscode/`** - VSCode settings (ignored in git)
- **`environment_gpu.yml`** - Conda environment specification
- **`requirements_gpu.txt`** - Python package dependencies

## Key Files
- `CONTEXT.md` - Root context documentation
- `reinforcement_learning/train_config.yaml` - Training hyperparameters and environment config
- Various `CONTEXT.md` files in subdirectories for module-specific documentation
