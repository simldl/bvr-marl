# Air-to-Air RL Simulation - Project Overview

## Purpose
This is a reinforcement learning simulation focused on air-to-air combat scenarios. The codebase implements a physics-based flight simulator with aircraft, missiles, radar systems, and multi-agent reinforcement learning environments for training tactical behavior.

## Project Type
- **Domain**: Air combat simulation & reinforcement learning
- **Language**: Python 3.12
- **Platform**: Windows (developed on Windows with conda environment)
- **Primary Goal**: Train RL agents for tactical air-to-air combat in multi-agent scenarios (1v1 to 4v4)

## Key Features
- High-fidelity aircraft flight dynamics with energy management
- Missile guidance systems (Fox-1 SARH, Fox-2 IR, Fox-3 ARH)
- Realistic radar modeling with RCS patterns, noise, and tracking
- Geographic coordinate system with proper geodesics
- Multi-agent RL environments with rich observation spaces
- Advanced action processing with physics-aware control
- Reward shaping for tactical behavior learning
- Real-time visualization and Tacview logging

## Development Status (as of November 2025)
- Core simulation: ✅ Stable
- Aircraft control: ✅ Energy + Lift-Vector action space
- Missile systems: ✅ Functional (Fox-1, Fox-2, Fox-3)
- RL integration: ✅ Active PPO training with reward adjustments for missile engagement
- Visualization: ✅ Working (2D live view, Tacview, radar performance analysis graphics)
- Radar/RCS Analysis: ✅ New detection probability heatmaps and RCS pattern visualization
- Testing: ✅ Comprehensive coverage with pytest
- Code quality: ✅ All comments in English, organized codebase
