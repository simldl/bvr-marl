# Architecture

## Package Layout

All source code lives under `src/air_to_air_rl/`. The package is installed in editable mode (`pip install -e .`) so every sub-package is importable as `air_to_air_rl.*`.

```
src/air_to_air_rl/
├── core/           # Shared path helpers (paths.py)
├── simulator/      # Physics engine, map utilities
├── aircrafts/      # Aircraft types, flight dynamics, systems
│   ├── core/       # Base classes and energy model
│   ├── types/      # Concrete aircraft (Eurofighter, F22, F35, …)
│   ├── control/    # Flight control system, AWACS
│   └── systems/    # Countermeasures, radar cross section
├── missiles/       # Fox-1 / Fox-2 / Fox-3 seekers and guidance
│   ├── fox1/       # Semi-active radar homing (SARH)
│   ├── fox2/       # Infrared (IR) homing
│   ├── fox3/       # Active radar homing (ARH)
│   └── guidance/   # Proportional navigation, augmented PN
├── radar/          # Radar simulation, tracking, electronic warfare
│   ├── core/       # Detection, range equations
│   ├── tracking/   # Kalman filter tracker
│   ├── ew/         # Electronic warfare (jamming)
│   └── obs/        # Observation builders for RL
├── physics/        # Low-level physics helpers
├── automation/     # Scripted controllers (behavior trees)
│   ├── core/       # Base automation classes
│   ├── scripted_control/   # Behavior tree nodes, maneuvers, tactics
│   └── strategies/ # High-level engagement strategies
├── rl/             # Reinforcement learning layer
│   ├── environment/ # Gym environments, reward functions, action/obs spaces
│   │   ├── gym/     # BVRMultiAgentEnv, SimplifiedMultiAgentEnv
│   │   ├── rewards/ # Public baseline reward calculator
│   │   └── spaces/  # Observation builders, action processors
│   ├── utils/      # env_creator, type maps
│   ├── training/   # Callbacks, curriculum helpers
│   ├── configs/    # Default YAML training configs (shipped with package)
│   └── evaluation/ # Post-training evaluation utilities
├── training/       # Entry-point wrappers (train.py, train_simple.py)
├── visualization/  # 2D live view, Tacview exports, scenario plotter
│   ├── scenplotter/ # ScenarioPlotter (Cairo + Cartopy rendering engine)
│   ├── config/     # Viz config dataclasses
│   └── utils/      # Combined visualizer utilities
├── tacview/        # ACMI file generation (TacviewLogger, generate.py)
├── analysis/       # Post-run plot export
├── gui/            # Streamlit control panel
│   └── components/ # Per-tab UI components
├── services/       # Shared business logic used by GUI and CLI
│   ├── training.py    # Command building, config discovery, background launch
│   ├── visualization.py # Visualization command construction
│   ├── tacview.py     # Tacview command construction
│   ├── configs.py     # Unified config discovery wrappers
│   └── processes.py   # Generic background process monitor
└── utils/          # Config loader, logging helpers
```

## Data Flow

### Training

```
configs/training/*.yaml
        │
        ▼
utils/config_loader.load_train_config()
        │
        ▼
rl/utils/env_creator.create_env_creator(cfg)
        │   ┌─ BVRMultiAgentEnv (gym) ──────────┐
        │   │   ├─ Simulator (physics tick)       │
        │   │   ├─ Observation spaces             │
        │   │   ├─ Action processors              │
        │   │   └─ Reward modules                 │
        │   └────────────────────────────────────┘
        │   wrapped by RewardNormalizationWrapper
        │
        ▼
training/train.py  (PPO via Ray RLlib)
        │
        ▼
models/<run>/  (RLlib checkpoint)
```

### Visualization

```
models/<run>/checkpoint_*/
        │
        ▼
visualization/policy_inference
        │
        ▼
visualization/scenplotter/ScenarioPlotter.to_rgba()
        │  (Cairo surface → NumPy RGBA array, no display required)
        │
        ▼
Matplotlib FuncAnimation → video / live window
```

### Tacview

```
tacview/generate.py:run_tacview_scenario()
        │  (env step loop; no Ray required without a checkpoint)
        │
        ▼
tacview/logger.TacviewLogger  →  *.acmi  (open in Tacview)
```

## Key Design Decisions

### `src/` layout
All source lives under `src/air_to_air_rl/`. This prevents accidental imports from the project root and is a Python packaging best practice (`pip install -e .` makes the package editable).

### Services layer
`air_to_air_rl.services.*` contains shared logic (command construction, process monitoring, config discovery) that is used by both the GUI components and any future CLI tooling. GUI components must not implement their own subprocess launching — they call the relevant service.

### Runtime state separation
Generated files (logs, checkpoints, process state JSON) go to `runtime/` (via `core.paths.runtime_root()`), never into the source tree. See `core/paths.py` for the canonical directory layout.

### Reward modularity
Rewards are composed from small, independently testable components in `rl/environment/rewards/`. Each component inherits from `RewardComponent` and is assembled by `RewardConfig`. See `docs/configuration.md` for details.

### Headless rendering
`ScenarioPlotter.to_rgba([])` renders entirely in memory via pycairo + Cartopy. No display server is needed. This is what the smoke tests use.
