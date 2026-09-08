"""Double-buffered kinematic state for simultaneous simulation ticks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _KinematicState:
    position: object
    scalars: dict[str, float]


def _capture(unit) -> _KinematicState:
    position = getattr(unit, "position", None)
    position_copy = (
        position.copy() if position is not None and hasattr(position, "copy") else position
    )
    scalars = {
        name: getattr(unit, name)
        for name in ("yaw_deg", "pitch_deg", "roll_deg", "speed")
        if hasattr(unit, name)
    }
    return _KinematicState(position_copy, scalars)


def _restore(unit, state: _KinematicState) -> None:
    if state.position is not None:
        current = getattr(unit, "position", None)
        if current is not None and all(hasattr(current, name) for name in ("lat", "lon", "alt")):
            current.lat = state.position.lat
            current.lon = state.position.lon
            current.alt = state.position.alt
        else:
            unit.position = (
                state.position.copy() if hasattr(state.position, "copy") else state.position
            )
    for name, value in state.scalars.items():
        setattr(unit, name, value)
    if hasattr(unit, "_velocity_cache"):
        unit._velocity_cache = None


class TickStateBuffer:
    """Present one immutable-time kinematic view while units update independently."""

    def __init__(self, roster):
        self._units = {unit.id: unit for unit in roster}
        self._start = {unit.id: _capture(unit) for unit in roster}
        self._next: dict[object, _KinematicState] = {}

    def restore_start(self) -> None:
        for unit_id, unit in self._units.items():
            _restore(unit, self._start[unit_id])

    def capture_next(self, unit) -> None:
        self._next[unit.id] = _capture(unit)

    def next_position(self, unit):
        state = self._next.get(getattr(unit, "id", None))
        if state is None or state.position is None:
            return None
        return state.position.copy() if hasattr(state.position, "copy") else state.position

    def publish(self) -> None:
        for unit_id, unit in self._units.items():
            _restore(unit, self._next.get(unit_id, self._start[unit_id]))
