# BVR-MARL: Public Baseline Platform for BVR Air-Combat Reinforcement Learning

[![CI](https://github.com/simldl/bvr-marl-core/actions/workflows/ci.yml/badge.svg)](https://github.com/simldl/bvr-marl-core/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The public baseline platform for BVR (Beyond Visual Range) air-combat simulation and reinforcement learning. A standalone package covering the complete pipeline from physics and scenario generation through RL environment, training, live visualization, Tacview export, and analysis  ready to use out of the box and designed to be extended via clean protocol interfaces.

## Quick Start

```bash
# Install from source
pip install -e .

# Launch the GUI control panel (recommended)
bvr-gui

# Or start training directly
bvr-train

# Watch live training visualization
bvr-view
```

## What This Package Provides

### Complete BVR Combat Simulator
- **Realistic aircraft models**: F-22, F-35, Eurofighter Typhoon, Su-57, AWACS
- **Advanced missile systems**: AIM-120 AMRAAM, Meteor, R-77, Python, Sidewinder
- **Radar and EW simulation**: RCS-based detection, ECM/ECCM, data fusion, AWACS datalink
- **Physics engine**: Energy-based flight dynamics with authentic performance models

### Training-Ready RL Environments
- **Gymnasium environments**: `BVRMultiAgentEnv` for full complexity, `SimplifiedMultiAgentEnv` for fast training
- **Ray RLlib integration**: Multi-agent PPO using the public RLlib baseline
- **Multi-agent framework**: Self-play and asymmetric training scenarios, 1v1 to NvN
- **Immediate training**: Start training with included baseline configurations

### Training Infrastructure
- **Config builder**: `build_ppo_config` for Ray RLlib PPO setup
- **Callbacks**: Progress tracking, smart checkpointing, weight loading
- **Checkpoint utilities**: Resume training, load weights from checkpoint
- **Adaptive config**: Automatic tier-based configuration scaling for CPU and GPU systems

### Visualization and Replay Tools
- **Live visualization**: Real-time 2D combat display with tactical overlays
- **RL commands panel**: Action and command visualization during inference
- **Tacview export**: Professional 3D replay in ACMI format (DCS-compatible)

### GUI and Analysis
- **Streamlit GUI**: Complete control panel for training, visualization, and analysis
- **Analysis tools**: Training metrics and episode performance plots
- **Batch training**: Multi-seed and multi-config batch execution

### Extension Framework
- **Protocol interfaces**: `bvr_marl_core.interfaces` defines Protocols for controllers, sensors, weapons, and flight models
- **Public API**: Clean import boundaries via `bvr_marl_core.public`
- **Schema system**: Versioned configuration models with migration support
- **Domain DTOs**: Typed state, command, and event objects

## Architecture Overview

The repository is organized as a single public platform with structured package areas:

```
bvr-marl-core/
|-- src/bvr_marl_core/
|   |-- aircraft/          # Aircraft models, systems, and observation helpers
|   |-- missiles/          # Missile guidance and physics
|   |-- radar/             # Radar simulation, RCS, tracking, data fusion
|   |-- physics/           # Flight dynamics and energy management
|   |-- simulator/         # Core simulation engine and event bus
|   |-- rl/                # RL environments, training helpers, callbacks
|   |-- training/          # CLI entry points (bvr-train, bvr-train-simple)
|   |-- visualization/     # Live views, scenario plotter, model wrapper
|   |-- tacview/           # Tacview ACMI export
|   |-- analysis/          # Post-training analysis and plot export
|   |-- gui/               # Streamlit control panel and components
|   |-- services/          # Process orchestration helpers
|   |-- domain/            # Typed state/command/event DTOs
|   |-- interfaces/        # Protocol definitions for extension points
|   |-- schema/            # Versioned config models with migration engine
|   |-- public/            # Public-safe training and visualization entry points
|   |-- utils/             # Config loader, geometry helpers, path helpers
|   `-- registry.py        # Aircraft and missile type registries
|-- configs/
|   |-- training/          # basic.yaml
|   |-- visualization/     # Live view and video export configs
|   |-- gui/               # GUI application config
|   `-- tacview/           # Tacview generation config
|-- scripts/               # Convenience launcher scripts
`-- tests/                 # Unit, integration, and smoke tests
```

## Installation

### Prerequisites
- Python 3.12+
- Node.js (for visualization symbols)
- CUDA 12.8+ (optional, for GPU acceleration)

### Install from Source
```bash
git clone https://github.com/simldl/bvr-marl-core
cd bvr-marl-core
pip install -e .

# Generate visualization symbols (required for proper tactical symbol display)
cd src/bvr_marl_core/visualization/symbols
npm install milsymbol canvas
node generate_symbols.js
```

### With GPU Support
```bash
pip install -e ".[gpu]"
```

### Development Install
```bash
pip install -e ".[dev]"   # Adds pytest and ruff
```

## Usage

### GUI Workflow (Recommended)

```bash
bvr-gui
```

The GUI opens a Streamlit control panel at `http://localhost:8501` and provides:

#### Training Dashboard
Start, monitor, and stop training runs from a browser interface.
- **Launch Training**  select a config from `configs/training/`, optionally override hyperparameters, then click Launch. The process runs in the background with logs captured to `runtime/gui/logs/`.
- **Batch Launch**  run multiple configs sequentially via the batch service.
- **Process Monitor**  live log tail, PID, runtime, and a stop button per training job.
- **Training Monitor**  real-time reward and metric curves from TensorBoard event files.

#### Visualization Panel
Launch live 2D battle visualizations with checkpoint and config selection.

| Mode | Command | Description |
|---|---|---|
| Standard 2D Live View | `bvr-view` | Aircraft, missiles, radar cones |
| RL Commands Panel | `bvr-view-commands` | Time-series RL action plots |

Behavior-tree visualization is available through the optional behavior package
via `bvr-view-bt`.

#### Tacview Generator
Generate `.acmi` files for 3D replay in [Tacview](https://www.tacview.net/).
- **RL Model** tab: run the environment with a trained checkpoint or random actions.
- **Behavior Tree** tab: run with the optional behavior package command when it
  is installed.

Output files land in `tacview/logs/` by default.

#### Config Builder & Validator
- **Config Builder**  GUI form for creating new training YAML configs, saved to `configs/training/`.
- **Config Validator**  load a YAML config and check it against the environment schema, showing warnings for unknown or out-of-range parameters.

#### Analysis
Visualize post-training metrics: episode reward distributions, kill/death ratios, energy efficiency over training iterations. Reads from `models/` and TensorBoard event files.

---

### Command Line Training

#### Standard Training (Full BVR Environment)
```bash
bvr-train
# or with explicit config
bvr-train --config configs/training/basic.yaml
```

#### Simplified Training (Faster Iteration)
```bash
bvr-train-simple
# or
bvr-train-simple --config configs/training/basic.yaml
```

#### Public Baseline Training
```bash
bvr-train-public
```

### Environment Configuration

Scenarios are configured via YAML files in `configs/training/`:

```yaml
env:
  num_agents_per_team: 2
  map_size_km: 300
  max_steps: 1200
  agent_aircraft_type: "F22"
  opponent_aircraft_type: "Su57"
  datalink_mode: full    # full | own | none | other | msl_support
```

### Tacview Export
```bash
# RL model scenario
bvr-tacview --checkpoint path/to/checkpoint

# Behavior-tree scenario, when the behavior package is installed
bvr-tacview-bt

# Via script (unified interface for both)
python scripts/tacview/generate_scenario.py --controller rl --checkpoint model.pkl
python scripts/tacview/generate_scenario.py --controller behavior-tree --num-scenarios 5
```

### Batch Training & Analysis
```bash
# Train with multiple seeds
python scripts/batch/batch_train.py --config configs/training/basic.yaml --seeds 42 123 456

# Compare training runs in TensorBoard
python scripts/batch/compare_runs.py --runs-dir outputs/ --tensorboard

# Launch TensorBoard
python scripts/analysis/launch_tensorboard.py

# Export analysis plots
bvr-export-plots
```

### Python API

```python
from bvr_marl_core.simulator import Simulator, MapLimits
from bvr_marl_core.schema import ScenarioConfig
from bvr_marl_core.rl.environment.gym import BVRMultiAgentEnv

# Direct simulator usage
sim = Simulator(tick_secs=1.0, random_seed=42)

# RL environment
env = BVRMultiAgentEnv(config)
obs, info = env.reset()
```

## Console Commands

All commands available after `pip install -e .`:

| Command | Description |
|---|---|
| `bvr-gui` | Launch Streamlit control panel |
| `bvr-train` | Full BVR environment training |
| `bvr-train-simple` | Simplified environment training |
| `bvr-view` | Standard 2D live visualization |
| `bvr-view-commands` | RL command panel |
| `bvr-export-plots` | Export training analysis plots |
| `bvr-tacview` | Generate Tacview ACMI (RL model) |

The optional behavior package adds `bvr-view-bt` and `bvr-tacview-bt`.

## Training Configurations

| Config | File | Description |
|---|---|---|
| Basic | `configs/training/basic.yaml` | Public RLlib PPO baseline for custom work |

## System Requirements

### Minimum Requirements
- Python 3.12+
- 8 GB RAM
- 4+ CPU cores

### Recommended for Training
- Python 3.12+
- 16 GB+ RAM
- NVIDIA GPU with 8 GB+ VRAM (CUDA 12.8+)
- SSD storage

## Troubleshooting

### Visualization Shows Basic Shapes Instead of Aircraft Symbols

The GUI and live views require pre-generated PNG symbol files:

```bash
cd src/bvr_marl_core/visualization/symbols
npm install milsymbol canvas
node generate_symbols.js
```

### Common Installation Issues

**Node.js not found**: Install Node.js from [nodejs.org](https://nodejs.org/)

**Canvas compilation errors on Windows**:
```bash
npm install --global windows-build-tools
```

**CUDA issues**: Install CUDA Toolkit 12.8+ or use CPU-only installation without `[gpu]` extra.

### Training Issues

**Out of memory**: Reduce `num_workers` and `train_batch_size` in your config.

**Training fails to start**: Check that Ray can allocate the requested resources:
```bash
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## Citation

If you use this environment in your research, please cite:

```bibtex
@inproceedings{inproceedings,
author = {Schosser, Simon and Retzlaff, Carl and Schulte, Axel},
year = {2026},
month = {01},
pages = {},
title = {Multi-Agent Reinforcement Learning Environment for Beyond Visual Range Air Combat (BVR-MARL)},
doi = {10.2514/6.2026-1592}
}
```

## Issues & Support

For bug reports or feature requests:

1. Check existing issues for similar problems
2. Provide minimal reproduction steps
3. Include environment details (OS, Python version, GPU)

**Contact**: simon.schosser@unibw.de

## License

MIT License  see [LICENSE](LICENSE) for details.

This public release is provided for research and educational purposes. Commercial use should verify compatibility with domain-specific regulations.

---

**Ready to start- `pip install -e . && bvr-gui`**
