# Task Completion Checklist

When completing a development task, follow this checklist to ensure quality and consistency.

## Code Quality

### 1. Code Style
- [ ] All code follows snake_case naming for variables/functions
- [ ] Classes use PascalCase naming
- [ ] Constants use SCREAMING_SNAKE_CASE
- [ ] All comments and docstrings are in English (no German)
- [ ] Type hints added to function signatures
- [ ] Units included in variable names (e.g., `speed_mps`, `alt_m`, `yaw_deg`)

### 2. Documentation
- [ ] Added/updated docstrings for new public functions
- [ ] Added inline comments for complex logic
- [ ] Updated relevant CONTEXT.md files if architecture changed
- [ ] Clear variable names that are self-documenting

## Testing

### 3. Test Coverage
- [ ] Run relevant unit tests: `pytest tests/test_<feature>.py`
- [ ] Run full test suite: `pytest tests/`
- [ ] All tests pass without errors
- [ ] Added new tests for new functionality (if applicable)
- [ ] Edge cases covered in tests
- [ ] Physics limits respected and tested

### 4. Integration Testing
- [ ] Test with actual RL environment if RL-related changes
- [ ] Verify simulation still runs correctly
- [ ] Check for regressions in related components
- [ ] Test boundary conditions and error handling

## Verification

### 5. Functional Verification
- [ ] Feature works as intended
- [ ] No unintended side effects
- [ ] Performance is acceptable (no significant slowdowns)
- [ ] Memory leaks checked (if long-running operations)

### 6. Code Review Self-Check
- [ ] No debug print statements left in production code
- [ ] No commented-out code blocks (unless explicitly needed)
- [ ] No hardcoded paths or magic numbers (use constants or config)
- [ ] Proper error handling added
- [ ] Configuration changes reflected in YAML files if needed

## Physics/Simulation Specific

### 7. Physics Validation (if applicable)
- [ ] Units are consistent throughout
- [ ] Physics equations are correct
- [ ] Numerical stability verified
- [ ] Tested at boundary conditions (min/max speed, altitude, etc.)
- [ ] Energy conservation checked (if relevant)

### 8. RL Environment (if applicable)
- [ ] Observation space dimensions correct
- [ ] Action space properly bounded [0, 1]
- [ ] Reward signals make sense and are scaled appropriately
- [ ] Episode termination conditions work correctly
- [ ] Info dict contains useful debug information

## Final Steps

### 9. Clean Up
- [ ] Remove temporary files and debug outputs
- [ ] Format code consistently
- [ ] Remove unused imports
- [ ] Remove obsolete TODO comments (or track them properly)

### 10. Git Workflow
- [ ] Changes staged: `git add <files>`
- [ ] Meaningful commit message prepared
- [ ] Commit follows project conventions
- [ ] Branch is up to date with main (if on feature branch)

## Optional (for Major Features)

### 11. Documentation Update
- [ ] Update README.md if user-facing changes
- [ ] Update CONTEXT.md with architectural changes
- [ ] Add examples to docstrings if complex API
- [ ] Update train_config.yaml comments if config options changed

### 12. Training Verification (for RL changes)
- [ ] Run short training: `python reinforcement_learning/train.py`
- [ ] Check TensorBoard for reasonable metrics
- [ ] Verify no NaN/Inf in training logs
- [ ] Check that GPU is being utilized (if available)

## Common Issues to Check

- **Import errors**: Verify all imports are correct and available
- **Path issues**: Use OS-agnostic paths or Path library
- **Type errors**: Check type hints match actual usage
- **Unit mismatches**: Verify all physics quantities have correct units
- **Boundary violations**: Test map limits, speed limits, altitude limits
- **Null references**: Check for None before accessing attributes
- **Array dimensions**: Verify NumPy array shapes are correct
- **Action space**: Ensure actions are clipped to [0, 1]
- **Observation space**: Ensure observations are properly normalized

## Notes

- Not all items apply to every task
- Use judgment to determine which checks are relevant
- When in doubt, run the full test suite
- Keep tests fast to encourage frequent testing
- Document any deviations from the checklist with rationale
