# Development Guide

## Setup

```bash
git clone https://github.com/simldl/air_to_air_rl
cd air_to_air_rl

# Conda (recommended — includes CUDA-enabled PyTorch)
conda env create -f environment_gpu.yml
conda activate rlenv

# Install in editable mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

The `[dev]` extra installs pytest and ruff.

## Running Tests

```bash
# Fast tests only (< 0.5 s each)
make test-fast
# or: pytest -m "not slow"

# All tests including slow physics/integration tests
make test
# or: pytest

# Smoke tests only (all four main programs, ~4 s total)
pytest -m smoke

# Single subsystem
pytest tests/missiles/
pytest tests/radar/
pytest tests/rl/
```

### Test Markers

| Marker | Meaning | Applied via |
|--------|---------|-------------|
| `smoke` | End-to-end entry-point checks (fast, ~4 s) | `pytestmark = pytest.mark.smoke` at top of file |
| `slow` | Long-running tests (physics simulations, full training) | `@pytest.mark.slow` on specific functions |
| `integration` | Tests spanning multiple subsystems | `pytestmark = pytest.mark.integration` at top of file, or auto-applied for `tests/simulator/` |
| `unit` | Pure unit tests with no external I/O | auto-applied for `tests/aircrafts/`, `tests/missiles/`, `tests/physics/`, `tests/radar/`, `tests/rl/` |

All markers are registered in `pyproject.toml` under `[tool.pytest.ini_options].markers`.

Directory-level default markers are applied automatically by `tests/conftest.py::pytest_collection_modifyitems` — no annotation needed on individual test files in those directories.

**Execution strategies:**

```bash
# Fast feedback (skip slow tests)
pytest -m "not slow"

# Only smoke tests
pytest -m smoke

# Only integration tests
pytest -m integration

# Specific subsystem
pytest tests/missiles/
pytest tests/radar/
pytest tests/rl/
```

Deselect slow tests: `pytest -m "not slow"`.

## Code Quality

```bash
# Lint + auto-fix
make lint
# or: ruff check --fix src/ tests/

# Format
make fmt
# or: ruff format src/ tests/

# Lint + format check (CI mode, no writes)
make check
# or: ruff check src/ tests/ && ruff format --check src/ tests/
```

Ruff is configured in `pyproject.toml` (`[tool.ruff]`). Line length is 100. Selected rule sets: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade).

Pre-commit runs ruff lint+format on every commit automatically.

## Project Structure

```
air_to_air_rl/
├── src/air_to_air_rl/   ← All source (see docs/architecture.md)
├── tests/               ← pytest test suite
│   ├── smoke/           ← Entry-point smoke tests (@pytest.mark.smoke)
│   ├── integration/     ← Multi-subsystem integration tests (@pytest.mark.integration)
│   ├── missiles/        ← Missile guidance unit tests
│   ├── radar/           ← Radar simulation unit tests
│   ├── aircrafts/       ← Aircraft dynamics unit tests
│   ├── rl/              ← RL environment and network tests
│   ├── physics/         ← Low-level physics unit tests
│   ├── simulator/       ← Simulator core unit tests
│   └── mocks/           ← Shared test mock objects
├── configs/             ← User-created config overrides (canonical config root)
│   ├── training/        ← Training YAML overrides
│   └── visualization/   ← Visualization YAML overrides
├── scripts/             ← Thin CLI wrappers (delegate to the package)
├── tools/               ← Development tools
│   └── profiling/       ← cProfile simulation and training profiling scripts
├── docs/                ← This documentation
└── runtime/             ← Generated at runtime (gitignored)
    └── gui/logs/        ← Background process log files
```

## Adding a New Aircraft Type

1. Create `src/air_to_air_rl/aircrafts/types/my_aircraft.py` inheriting from `Airplane`.
2. Set `aircraft_type = "my_aircraft"` as a class attribute.
3. Register it in `rl/utils/type_maps.py`.
4. Add a test in `tests/aircrafts/types/`.


## Adding a New Visualization Mode

1. Add a new `live_view_*.py` entry point in `src/air_to_air_rl/visualization/`.
2. Register it in `pyproject.toml` under `[project.scripts]`.
3. Add its display name and `--mode` key to `services/visualization.VISUALIZATION_MODES`.
4. Add it to the mode list in `gui/components/visualization_panel.py`.

## Profiling

```bash
# Profile the simulation (no Ray, fast)
python tools/profiling/profile_simulation.py --episodes 5 --steps 200

# Profile a full training run (starts Ray)
python tools/profiling/profile_training.py --iterations 20
```

Output goes to `profiling_results/` (gitignored).

## Runtime Output Conventions

See [docs/runtime.md](runtime.md) for the full artifact → location mapping and directory layout.

All runtime output paths are resolved relative to `project_root()` from `air_to_air_rl.core.paths`.
The root for generated state can be overridden via the `AIR2AIR_RUNTIME_DIR` environment variable.

## CI

GitHub Actions runs on push to `main`, `refactor/**`, `claude/**`, `feature/**`, and `fix/**`
branches, and on all pull requests targeting `main`.  Matrix: Ubuntu + Windows, Python 3.12.


1. **Lint** — `ruff check`
2. **Format check** — `ruff format --check`
3. **Fast tests** — `pytest -m "not slow"`
4. **Slow tests** — `pytest -m slow`

See `.github/workflows/ci.yml`.

## Release Checklist

- [ ] All smoke tests pass: `pytest -m smoke`
- [ ] All fast tests pass: `pytest -m "not slow"`
- [ ] No lint errors: `make check`
- [ ] README badges reflect current CI status
- [ ] `pyproject.toml` version bumped
- [ ] CHANGELOG updated
