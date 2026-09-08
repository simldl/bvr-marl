# BVR-MARL: Public Baseline Platform for BVR Air-Combat Reinforcement Learning

[![CI](https://github.com/simldl/bvr-marl/actions/workflows/ci.yml/badge.svg)](https://github.com/simldl/bvr-marl/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The public baseline platform for BVR (Beyond Visual Range) air-combat simulation and reinforcement learning. A standalone package covering the complete pipeline from physics and scenario generation through RL environment, training, live visualization, Tacview export, and analysis—ready to use out of the box and designed to be extended via clean protocol interfaces.

## About This Release

**This is an updated release, and it goes beyond the scope of the published paper.** The
version that corresponds to the paper is the previous commit on `main`; check it out if you
need the exact configuration the published results were produced with.

Since then the simulator has been corrected in several places and extended in others. The
corrections matter most, because a few of them change measured outcomes:

- **Terminal engagement.** An active seeker now re-associates onto its own track after
  acquisition instead of discarding it on an identity mismatch, which had left weapons
  dead-reckoning through the endgame. This moves kill rates throughout.
- **Launch authority.** The launch-range gate is the missile's dynamic launch zone rather
  than radar range alone, and accounts for target aspect, altitude delta, and time of
  flight. A long-range radar no longer implies a viable shot.
- **One fire-feasibility predicate.** The can-fire observation bit, the shot-opportunity
  metric, and the launch gate itself are now the same evaluation; previously three
  hand-written variants disagreed with each other.
- **Track confidence** is calibrated against the tracker's measured distribution, so
  thresholds are reachable at BVR range rather than silently acting as "never".
- **Contact-slot identity** is reset per episode and binned over occupied slots, fixing
  stale contacts accumulating across episodes in a reused environment.

Capabilities added beyond what the paper describes include passive IRST, radar emission
control (EMCON) as an agent decision with scripted baselines, a fuel model, radar
identification and battle-damage-assessment observation features, A-pole/F-pole timeline
estimates, an information firewall enforced at runtime, a networked-datalink weapon class,
and a seeded verification and validation suite (`python -m bvr_marl_core.validation.cli`).

Two notes on scope. The reward calculator shipped here is the terminal-only baseline —
kills, own losses, boundary violations, and a last-team-standing bonus — so the dense
shaping suite the paper sketches is not part of this package; the enemy-fighter
observation token exposes an extension point (`ef_extra_dim`) for widening it, and the
reward calculator is designed to be replaced wholesale.

Three training additions address failures observed in long curriculum runs, and each is
**active by default** rather than opt-in: critic warm-up (`critic_warmup_iterations: 15`),
a floored KL coefficient (`kl_coeff_floor: 0.003`), and separate actor/critic gradient
clipping (`policy_grad_clip: 10.0`). Set each to `0`, `0.0` and `null` respectively to
recover stock RLlib behaviour — worth doing if you are comparing against a stock PPO
baseline.

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
bvr-marl/
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
- Python 3.12
- Node.js (for visualization symbols)
- CUDA 12.8+ (optional, for GPU acceleration)

### Install from Source
```bash
git clone https://github.com/simldl/bvr-marl
cd bvr-marl
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

  random_map:
    min_separation_m: 40000
    # Optional upper bound on spawn separation. Omit for the historical
    # floor-only behaviour. Set it on stages that must train weapon employment
    # rather than transit — with a floor alone a large map routinely spawns both
    # teams beyond radar range, so an agent spends the episode in transit and
    # never reaches a position from which weapon employment can be learned.
    max_separation_m: 100000
```

```yaml
training:
  # Iterations at the start of each run during which only the value function
  # receives gradient. Curriculum promotion warm-starts the whole module, so the
  # next stage inherits a critic calibrated to the previous stage's return scale.
  # 0 restores stock RLlib behaviour. The rationale is documented in
  # `rl/training/critic_warmup_learner.py`.
  critic_warmup_iterations: 15
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

## Development

Install the dev extras first (`pip install -e ".[dev]"`), then run the same checks
CI runs. Running all of them locally reproduces the pipeline:

```bash
# Tests — the slow marker separates simulation-heavy suites
pytest tests/ -m "not slow"
pytest tests/ -m "slow"
pytest tests/                       # everything

# Lint and format
ruff check src/ tests/ scripts/
ruff check src/ tests/ scripts/ --fix
ruff format src/ tests/ scripts/

# Source encoding (UTF-8, no BOM)
python scripts/check_encoding.py

# Import-boundary enforcement
python scripts/check_import_boundaries.py
```

### Project conventions

- Python 3.12 syntax and type hints,
- `from __future__ import annotations` in modules,
- stdlib / third-party / first-party import ordering,
- **absolute imports only** — relative imports are rejected by `TID252`, so a
  module's dependencies read the same wherever the file sits in the tree. Type
  stubs (`*.pyi`) are exempt, since relative re-export is idiomatic there,
- catch the exceptions a call can actually raise rather than bare `except
  Exception`; where a broad catch is deliberate (a third-party boundary, or a
  hot loop that must never abort a tick) mark it `# noqa: BLE001` **with a
  reason**,
- diagnostics belong in `logging`, not `print`, anywhere that runs per tick or
  per episode — `print` is reserved for CLI and report output.

### Architecture decisions

The information firewall is the load-bearing one: operational decisions consume the
chain `SensorReport -> TrackSnapshot -> TacticalContact -> WeaponTrack`, and none of
those records carries a simulator entity handle. Sensor-limited mode is the default
and fails closed, so a policy or controller cannot reach ground truth by accident.
Read `domain/truth_access_guard.py` and the architecture tests under
`tests/architecture/` before changing how information flows between sensors, tracks,
and agents.

Validation study outputs (physics envelopes, radar detection, tracking, missile
effectiveness) are regenerated with `python scripts/validation/run_studies.py`, which
writes them under `docs/validation/results/`.

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

Note that this release has moved beyond the paper — see [About This Release](#about-this-release).
Work reproducing the published results should use the previous commit on `main`, which is the
version the paper describes.

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
