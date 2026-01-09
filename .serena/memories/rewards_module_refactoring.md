# Rewards Module Refactoring (December 2025)

## Overview
The `reinforcement_learning/environment/rewards/` module was refactored from large monolithic files into a modular structure with smaller, focused modules organized in subfolders.

## Structure

```
rewards/
├── __init__.py                    # Main exports
├── calculator.py                  # Core reward aggregation (310 lines)
├── terminal_rewards.py            # Terminal/sparse rewards (51 lines)
│
├── tactical/                      # Tactical engagement rewards (4 modules)
│   ├── base.py                   # Positioning & tracking
│   ├── shooting.py               # Shot quality & timing
│   ├── engagement.py             # NEZ closure & maneuvering
│   └── shaping.py                # SQI delta & detection rewards
│
├── control/                       # Flight control rewards (3 modules)
│   ├── flight_control.py         # Lift balance
│   ├── heading_alignment.py      # Heading rewards
│   └── passivity.py              # Passivity & orbit penalties
│
├── defensive/                     # Defensive maneuvering rewards (3 modules)
│   ├── boundary.py               # Boundary penalties
│   ├── evasion.py                # Missile evasion
│   └── zone_control.py           # Center zone control
│
└── energy/                        # Energy management rewards (2 modules)
    ├── management.py             # Energy state rewards
    └── advantage.py              # Relative energy advantage
```

## Key Changes
- Split 816-line `tactical_rewards.py` into 4 focused modules
- Split 399-line `control_rewards.py` into 3 focused modules
- Split 288-line `defensive_rewards.py` into 3 focused modules
- Split 188-line `energy_rewards.py` into 2 focused modules
- Split 460-line `rewards.py` into `calculator.py` (orchestrator)
- All new modules are under 250 lines (most under 200)

## Usage
```python
from reinforcement_learning.environment.rewards import RewardCalculator

calc = RewardCalculator(
    enable_terminal=True,
    enable_tactical=True,
    destruction_penalty=-200.0,
    kill_reward=200.0,
)
```

## Benefits
- Better organization with clear subfolder structure
- Smaller, more maintainable files
- Each module has single, well-defined purpose
- Easier to navigate and modify specific reward components

## Testing & Integration
- All 17 existing reward tests pass successfully
- Config integration works correctly with `reward_config.py`
- Import paths updated in all test files
- No breaking changes to existing training code
- Returns tuple format: `(reward, sqi, tactical_potential, energy_advantage)` from `compute_total_reward()`
