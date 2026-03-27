# Air-to-Air RL Simulation - Root Context

## Overview
This is a reinforcement learning simulation focused on air-to-air combat scenarios. The codebase implements a physics-based flight simulator with aircraft, missiles, radar systems, and multi-agent reinforcement learning environments.

## Project Structure

All source code lives under `src/air_to_air_rl/` as a single installable package.
Install with `pip install -e ".[dev]"` (editable) — no `PYTHONPATH` manipulation needed.

### Core Simulation (`src/air_to_air_rl/`)
- **`simulator/`** - Core simulation engine and utilities
- **`physics/`** - Physics models for aircraft and missiles
- **`aircrafts/`** - Aircraft implementation and control systems
- **`missiles/`** - Missile systems and guidance algorithms
- **`radar/`** - Radar simulation and tracking systems

### AI/ML Components
- **`rl/`** - RL environments, networks, and training (replaces old `reinforcement_learning/`)
- **`tacview/`** - Tactical view logging for scenario analysis

### Infrastructure
- **`core/paths.py`** - Package-aware path resolution (no CWD assumptions)
- **`services/training.py`** - Shared training launch logic for CLI and GUI
- **`visualization/`** - Real-time visualization and plotting
- **`gui/`** - Streamlit control panel
- **`configs/`** - User-editable configs at project root (training/, visualization/)
- **`tests/`** - Comprehensive test suite (pytest -m "not slow" for fast feedback)

## Key Features

### Physics Simulation
- High-fidelity aircraft flight dynamics with energy management
- Missile guidance systems (Fox-1, Fox-2, Fox-3)
- Realistic radar modeling with RCS, noise, and tracking
- Geographic coordinate system with proper geodesics

### Control Systems
- Rate-based aircraft control with physics-aware envelopes
- Direct speed control centered at Mach 0.7
- Boundary violation detection with delayed removal
- Weapon engagement zones and tactical indicators

### Reinforcement Learning
- Multi-agent environments with up to 4v4 scenarios
- Rich observation spaces with tactical information
- Advanced action processing with EMA filtering
- Reward shaping for tactical behavior

### Recent Improvements
- **Speed Control**: Action[0] now controls speed directly (Mach 0.7 reference)
- **Boundary Management**: Delayed removal system with continuous penalties
- **Aircraft Responsiveness**: Fixed attitude updates, improved rate limits
- **Action Processing**: Enhanced deadzone handling and physics awareness
- **Code Cleanup (Oct 2025)**: Comprehensive cleanup - translated all German comments to English, removed temporary files, cleaned whitespace, and removed obsolete code
- **Radar Analysis (Nov 2025)**: Added detection probability heatmaps and RCS pattern visualization tools in `radar/graphics/`

## Development Status (March 2026)
- Core simulation: ✅ Stable
- Aircraft control: ✅ Energy + Lift-Vector action space
- Missile systems: ✅ Functional (Fox-1, Fox-2, Fox-3)
- RL integration: ✅ Active PPO training with missile engagement rewards
- Visualization: ✅ Live 2D view, Tacview, symbol system
- Testing: ✅ 1602 tests; `pytest -m "not slow"` for fast 7 s feedback
- Package: ✅ Installable via `pip install -e .`; single `air_to_air_rl` namespace

## Getting Started
1. Install: `pip install -e ".[dev]"`
2. Run tests: `pytest -m "not slow"` (fast) or `pytest` (full)
3. Train: `air2air-train --config aggressive` (or `make test`)
4. Launch GUI: `air2air-gui`
5. Visualize: `air2air-view`

## Architecture Philosophy
- **Physics-first**: All systems grounded in realistic flight dynamics
- **Modular design**: Clear separation of concerns between components
- **Test-driven**: Extensive test coverage for reliability
- **Performance-oriented**: Optimized for RL training at scale