# Migration Notes

> **Historical record.** Documents the namespace consolidation that occurred in March 2026.
> All packages are now unified under `air_to_air_rl.*`. These notes are retained as reference
> for contributors who encounter old import patterns in git history.

## Phase 1 — Namespace Consolidation (2026-03-20)

All production code has been moved into the unified `air_to_air_rl` package namespace.

### Package moves

| Old import | New import |
|---|---|
| `from aircrafts.X import Y` | `from air_to_air_rl.aircrafts.X import Y` |
| `from missiles.X import Y` | `from air_to_air_rl.missiles.X import Y` |
| `from physics.X import Y` | `from air_to_air_rl.physics.X import Y` |
| `from radar.X import Y` | `from air_to_air_rl.radar.X import Y` |
| `from simulator.X import Y` | `from air_to_air_rl.simulator.X import Y` |
| `from utils.X import Y` | `from air_to_air_rl.utils.X import Y` |
| `from reinforcement_learning.X import Y` | `from air_to_air_rl.rl.X import Y` |
| `from visualization.X import Y` | `from air_to_air_rl.visualization.X import Y` |
| `from tacview.X import Y` | `from air_to_air_rl.tacview.X import Y` |
| `from automation.X import Y` | `from air_to_air_rl.automation.X import Y` |
| `from gui.X import Y` | `from air_to_air_rl.gui.X import Y` |

### Directory changes

- `src/reinforcement_learning/` → `src/air_to_air_rl/rl/`
- `src/aircrafts/` → `src/air_to_air_rl/aircrafts/`
- `src/missiles/` → `src/air_to_air_rl/missiles/`
- `src/physics/` → `src/air_to_air_rl/physics/`
- `src/radar/` → `src/air_to_air_rl/radar/`
- `src/simulator/` → `src/air_to_air_rl/simulator/`
- `src/utils/` → `src/air_to_air_rl/utils/`
- `src/automation/` → `src/air_to_air_rl/automation/`
- `src/visualization/` → merged into `src/air_to_air_rl/visualization/`
- `src/tacview/` → merged into `src/air_to_air_rl/tacview/`
- `src/gui/` (code) → `src/air_to_air_rl/gui/`

### Note for `patch()` strings in tests

All `unittest.mock.patch()` string arguments have been updated to use the new
module paths. For example:
- `patch('simulator.core.units.normalize_angle')` →
  `patch('air_to_air_rl.simulator.core.units.normalize_angle')`

## Phase 2 — Remove Path Hacks (2026-03-20)

### Central path module

`src/air_to_air_rl/core/paths.py` now provides package-aware helpers:
- `package_root()` — installed package directory
- `rl_configs_root()` — default training configs
- `visualization_root()` / `visualization_config(name)` — viz configs and assets
- `tacview_config()` — default Tacview config
- `symbols_root()` — visualization symbol assets
- `gui_app()` — Streamlit app entry point

### Removed sys.path hacks (27 files)

16 graphics analysis scripts, 2 automation test scripts, `gui/app.py`,
`rl/configs/test_reward_config.py`, and 6 test files no longer manipulate
`sys.path` — the installed package is found automatically.

### Fixed gui_launcher.py

- Removed `os.chdir(project_root)` (process CWD is no longer mutated)
- GUI app path resolved via `core.paths.gui_app()` instead of a
  hardcoded `src/gui/app.py` string

### Intentional os.getcwd() calls that remain

`training/train.py` and `training/train_simple.py` still use `os.getcwd()`
to resolve user-supplied relative checkpoint paths (e.g. `--checkpoint
./models/run1`). This is correct behavior — it resolves relative to wherever
the user runs the command, not the repository root.

---

> **Status as of 2026-03-24:** All items above have been resolved. Runtime state
> (process-tracking JSON, logs, generated plots) is now written to `runtime/` via
> `core.paths.runtime_root()` and is gitignored. No legacy directories remain in
> version control.
