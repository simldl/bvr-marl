.PHONY: install test test-fast lint fmt check clean

# ── Install ────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Test ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -q

# Fast: skip the slow physics-simulation tests (1591 tests, ~7 s)
test-fast:
	pytest tests/ -q -m "not slow"

# ── Lint / Format ──────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# Check (no writes) — suitable for CI
check:
	ruff check src/ tests/
	ruff format --check src/ tests/

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
