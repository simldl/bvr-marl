"""Runtime guard against forbidden truth access in the sensor-limited path.

The information firewall requires that a sensor-limited policy, behavior tree,
target-selection, weapon-guidance, or terminal-resolution path never resolves an
anonymous track/contact identity back to a live world-truth entity. Own-state access
(an agent reading its OWN unit) is legitimate and is NOT routed through this guard;
only the track-identity -> truth-entity resolution is.

Usage
-----
Wrap sensor-limited work in :func:`forbidden_truth_access`; perform any legitimate
oracle/evaluator truth resolution through :func:`resolve_truth_unit` with a documented
``reason`` (which raises if it happens inside a forbidden scope), or inside an explicit
:func:`allow_truth_access` boundary. A regression that adds a truth lookup to a
sensor-limited path then fails loudly instead of silently leaking privileged state.

The guard state is thread-local, so parallel env runners do not interfere.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

__all__ = [
    "TruthAccessViolation",
    "forbidden_truth_access",
    "allow_truth_access",
    "truth_access_forbidden",
    "resolve_truth_unit",
]


class TruthAccessViolation(RuntimeError):
    """Raised when a sensor-limited path attempts to resolve live world truth."""


_state = threading.local()


def _stack() -> list[tuple[bool, str]]:
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = []
        _state.stack = stack
    return stack


def truth_access_forbidden() -> bool:
    """True when the innermost active scope forbids truth resolution."""
    stack = _stack()
    return bool(stack and stack[-1][0])


@contextlib.contextmanager
def forbidden_truth_access(reason: str = "") -> Iterator[None]:
    """Scope in which truth resolution raises (e.g. sensor-limited observation build)."""
    _stack().append((True, reason))
    try:
        yield
    finally:
        _stack().pop()


@contextlib.contextmanager
def allow_truth_access(reason: str = "") -> Iterator[None]:
    """Explicitly labelled oracle/evaluator boundary that re-permits truth resolution."""
    _stack().append((False, reason))
    try:
        yield
    finally:
        _stack().pop()


def resolve_truth_unit(simulator, entity_id, *, reason: str):
    """Resolve a track/contact identity to a live truth unit, guarded.

    Returns ``simulator.active_units.get(entity_id)`` (or ``None``). Raises
    :class:`TruthAccessViolation` when called inside a :func:`forbidden_truth_access`
    scope, unless an inner :func:`allow_truth_access` re-permits it. ``reason`` names
    the oracle/evaluator boundary and appears in the violation message.
    """
    if truth_access_forbidden():
        raise TruthAccessViolation(
            f"Sensor-limited path attempted truth resolution of entity {entity_id!r} "
            f"(reason={reason!r}). Live world truth is reachable only at a labelled "
            f"oracle/evaluator boundary (allow_truth_access)."
        )
    units = getattr(simulator, "active_units", None)
    if units is None:
        return None
    return units.get(entity_id)
