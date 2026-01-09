# Code Style and Conventions

## General Style

### Language
- **All code and comments must be in English** (recent cleanup in October 2025 removed all German comments)
- Docstrings and inline comments should be clear and descriptive

### Python Style
- Generally follows PEP 8 conventions (no explicit linter config found)
- Uses type hints extensively (from `typing` and modern Python 3.10+ syntax)
- Uses `from __future__ import annotations` for forward references

### Import Organization
1. Standard library imports
2. Third-party library imports
3. Local application imports
4. Blank lines between groups

Example:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

from simulator.simulator import Simulator
from physics.aircraft import AircraftPhysics
from aircrafts.control.movement_control import AircraftControlSystem
```

## Naming Conventions

### Variables and Functions
- **snake_case** for variables, functions, methods: `speed_mps`, `get_density()`, `apply_rl_action()`
- **SCREAMING_SNAKE_CASE** for constants: `SEA_LEVEL_DENSITY`, `AIRSPEED_OF_SOUND`, `STANDARD_GRAVITY`
- Descriptive names with units included: `alt_m`, `speed_mps`, `range_km`, `yaw_deg`, `throttle` (0-1)

### Classes
- **PascalCase** for class names: `Aircraft`, `Missile`, `Simulator`, `ActionProcessor`
- Nested classes/dataclasses also use PascalCase: `AircraftPhysics.Params`, `PhysicsParams`

### Files and Directories
- **snake_case** for Python files: `aircraft.py`, `movement_control.py`, `hit_calculation.py`
- **snake_case** for directories: `reinforcement_learning/`, `missile_warner.py`

### Private/Protected Members
- Single underscore prefix for "internal use" methods: `_get_target_candidates()`, `_density_scalar()`
- Double underscore for name mangling (rare): not commonly used in this codebase

## Code Patterns

### Dataclasses
- Heavy use of `@dataclass` decorator for configuration and parameter objects
- Often with `field(default_factory=...)` for mutable defaults
```python
@dataclass
class PhysicsParams:
    mass_kg: float
    reference_area_m2: float
    gravity_m_s2: float = STANDARD_GRAVITY
    air: Any = field(default_factory=AirLayer)
```

### Type Hints
- Extensive use of type hints for function signatures
- Uses `Optional[T]` for nullable types
- Uses `List[T]`, `Dict[K, V]` from typing module
```python
def get_speed_of_sound(alt_m: float, temp_k: float = None) -> float:
    ...
```

### Dictionary Configuration
- Configuration objects often converted to dicts with `asdict()` from dataclasses
- Uses `.get()` with defaults for safe dictionary access:
```python
cfg_mass = config.get("mass_kg", 20000.0)
cfg_nmax = float(config.get("n_max", 9.0))
```

## Testing Conventions

### Test Files
- Test files named `test_*.py` in `tests/` directory
- Mirror source structure in test structure
- Descriptive test names: `test_speed_control_initialization()`, `test_ema_filtering_of_speed_commands()`
- **MUST be pytest-compatible**: NO `exit()` calls at module level, NO test execution code at module level
- Tests should be importable without side effects

### Pytest Fixtures
- Use `@pytest.fixture` for reusable test components
- Common fixtures: `simulator`, `aircraft`, `map_limits`
```python
@pytest.fixture
def simulator():
    return Simulator(tick_secs=0.1)
```

### Assertions
- Descriptive test function names
- Clear assertion messages (implicit via function names)
- Tolerance-based comparisons for floats: `abs(value - expected) < tolerance`

### Print Statements in Tests
- **CRITICAL: DO NOT use Unicode/special characters in print statements**
- Windows console encoding (cp1252) cannot handle ✓, ✗, ⚠, etc.
- Use ASCII alternatives:
  - Instead of ✓: Use "OK", "PASS", or "[SUCCESS]"
  - Instead of ✗: Use "ERROR", "FAIL", or "[FAILED]"  
  - Instead of ⚠: Use "WARNING" or "[WARN]"
  - Instead of →: Use "->" or "=>"
  - Instead of •: Use "*" or "-"

**Bad:**
```python
print(f"✓ Test passed")
print(f"✗ Test failed")
```

**Good:**
```python
print(f"OK Test passed")
print(f"ERROR Test failed")
```

## Documentation

### Docstrings
- Present but not universally used
- When used, typically one-line format for simple functions:
```python
def get_speed_of_sound(alt_m: float, temp_k: float = None) -> float:
    """Get local speed of sound based on altitude and temperature."""
```

### Comments
- Inline comments for complex logic
- Section headers with `# ---` style:
```python
# --- Flight Physics & Envelopes ---
# --- Radar & DataLink ---
# --- Subsystems ---
```

### Context Files
- `CONTEXT.md` files in major directories provide high-level documentation
- Markdown format with clear sections and examples

## Special Patterns

### Units in Variable Names
- Always include units in variable names: `_m`, `_mps`, `_deg`, `_km`, `_kg`, `_m2`, `_s`
- Makes physics code self-documenting and prevents unit errors

### Numeric Literals
- Use underscores for readability: `25_000.0`, `11_000.0`
- Explicit float literals: `1.0` not `1` when dealing with floats

### Configuration Handling
- Dataclass configs converted to dicts for flexible access
- Nested `.get()` with sensible defaults throughout
