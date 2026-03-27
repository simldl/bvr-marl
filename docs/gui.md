# GUI — Control Panel

The GUI is a Streamlit application that provides a unified interface for training, visualization, Tacview generation, and analysis.

## Launching

```bash
# Via installed entry point (after pip install -e .)
air2air-gui

# Or directly
python -m air_to_air_rl.gui_launcher
streamlit run src/air_to_air_rl/gui/app.py
```

The GUI opens at `http://localhost:8501` by default.

## Tabs

### Training Dashboard

Start, monitor, and stop training runs.

- **Launch Training**: Select a config from `configs/training/`, optionally override hyperparameters, then click **Launch**. The process runs in the background with stdout/stderr captured to `runtime/gui/logs/`.
- **Batch Launch**: Run multiple configs sequentially via `scripts/batch/batch_train.py`.
- **Process Monitor**: Shows PID, runtime, live log tail, and a stop button for each running training process.
- **Training Monitor**: Real-time training curves and scalar metrics (reads TensorBoard event files from `models/`).

### Visualization Panel

Launch live 2D battle visualizations.

| Mode | Entry point | Description |
|------|-------------|-------------|
| Standard 2D Live View | `air2air-view` | Aircraft, missiles, radar cones |
| Behavior Tree Controllers | `air2air-view-behavior-tree` | Scripted behavior tree AI |
| RL Commands Panel | `air2air-view-commands` | Time-series RL action plots |

**Checkpoint selection**: choose from discovered `models/` files, or paste a custom path.

**Config selection**: training configs from `configs/training/`; visualization configs from `configs/visualization/`.

### Tacview Generator

Generate `.acmi` files for 3D replay in [Tacview](https://www.tacview.net/).

- **RL Model** tab: runs the environment with a trained checkpoint (or random actions).
- **Behavior Tree** tab: runs the environment with scripted behavior tree controllers.

Both tabs call `scripts/tacview/generate_scenario.py` which delegates to `air2air-tacview` / `air2air-tacview-bt`. Output files land in `tacview/logs/` by default.

### Config Builder

GUI form for creating new training YAML configs. Saved to `configs/training/`.

### Config Validator

Loads a YAML config and checks it against the environment schema. Shows warnings for unknown or out-of-range parameters.

### Analysis

Visualize post-training metrics:
- Episode reward distributions
- Kill/death ratios
- Energy efficiency over training iterations

Reads from `models/` and TensorBoard event files.

## Process Management

All background processes (training, visualization, Tacview) are tracked in JSON state files under `runtime/gui/`:

| File | Contents |
|------|----------|
| `training_processes.json` | Running/completed training jobs |
| `viz_processes.json` | Running/completed visualization jobs |
| `tacview_processes.json` | Running/completed Tacview jobs |

The GUI reads these files on each page refresh and updates liveness by checking the OS process table via `psutil`. Stale entries can be cleared with the **Cleanup** button in each monitor section.

## Architecture

The GUI follows a layered design:

```
gui/app.py                   ← Streamlit entry point, tab routing
gui/components/              ← Per-tab UI components
    training_dashboard.py
    visualization_panel.py
    tacview_generator.py
    config_builder.py
    config_validator.py
    visualization_config_builder.py
    analysis_interface.py
    output_paths.py
services/                    ← Business logic (no Streamlit imports)
    training.py              ← build_training_cmd, launch_background_process
    visualization.py         ← build_visualization_cmd, find_checkpoint_files
    tacview.py               ← build_tacview_cmd
    configs.py               ← list_training_configs, list_visualization_configs
    processes.py             ← ProcessMonitor, ProcessRecord
```

GUI components **only** handle rendering (Streamlit calls). All command building and process launching goes through `services/`. This makes the services independently testable without a live Streamlit session.
