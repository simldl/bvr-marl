# scripts/

Repository-local launch wrappers for common workflows.

These are thin Python scripts that call the installed `air_to_air_rl` package
entrypoints. They exist for convenience when working directly in the repository
without typing full module paths.

> **Tip:** After installing the package (`pip install -e .`), the `air2air-*`
> console commands work from anywhere and are the preferred way to run tasks.
> The scripts here are equivalent — use whichever is more convenient.

---

## Training

| Script | Equivalent command | Description |
|---|---|---|
| `scripts/training/train.py` | `air2air-train` | Standard PPO training |
| `scripts/training/train_simple.py` | `air2air-train-simple` | Simplified training |
| `scripts/batch/batch_train.py` | — | Run multiple training jobs from a YAML job list |

```bash
# Train with default config
python scripts/training/train.py

# Train with a specific config
python scripts/training/train.py --config train_config_v14_aggressive_center.yaml

# Train with config overrides
python scripts/training/train.py --config standard.yaml --overrides training.steps=500
```

---

## Visualization

| Script | Equivalent command | Description |
|---|---|---|
| `scripts/visualization/live_view.py` | (mode dispatcher) | Multi-mode 2D live view |

```bash
# Standard live view
python scripts/visualization/live_view.py --mode standard


# Behavior tree visualization
python scripts/visualization/live_view.py --mode behavior-tree
```

---

## Tacview export

| Script | Equivalent command | Description |
|---|---|---|
| `scripts/tacview/generate_scenario.py` | (mode dispatcher) | Generate ACMI scenario files |

```bash
# Generate standard scenario
python scripts/tacview/generate_scenario.py

# Generate with a trained model
python scripts/tacview/generate_scenario.py --checkpoint ./models/run1/checkpoint

# Generate behavior-tree scenario
python scripts/tacview/generate_scenario.py --mode behavior-tree
```

---

## Analysis

| Script | Equivalent command | Description |
|---|---|---|
| `scripts/analysis/export_plots.py` | `air2air-export-plots` | Export TensorBoard plots to PNG |
| `scripts/analysis/launch_tensorboard.py` | — | Launch TensorBoard with auto-discovery |
| `scripts/analysis/evaluate_model.py` | — | Evaluate a trained model |
| `scripts/batch/compare_runs.py` | — | Compare multiple training runs |

---

## GUI

```bash
python scripts/launch_gui.py   # equivalent to: air2air-gui
```

---

## Standard argument names

| Argument | Purpose |
|---|---|
| `--config` | Path to configuration file or config name |
| `--config-path` | Path to configuration directory |
| `--checkpoint` | Path to trained model checkpoint |
| `--output-dir` | Directory for outputs |
| `--seed` | Random seed |
| `--overrides` | `key=value` config overrides |
