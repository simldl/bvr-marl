"""Per-episode radar-emission tracker for active sensing.

Records the ownship radar on/off state each step so the policy can observe its own
emission behavior (duty cycle, time since last toggle) and so episode-end analysis can
report duty cycle, switch count, and exposure for the sensing Pareto frontier. One
tracker lives on each aircraft; because units are re-spawned every episode it is
naturally fresh per episode.
"""

from __future__ import annotations


class EmissionTracker:
    """Running radar-emission statistics for one aircraft over one episode."""

    __slots__ = ("steps", "emitting_steps", "transitions", "steps_since_toggle", "_last")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.emitting_steps = 0
        self.transitions = 0
        self.steps_since_toggle = 0
        self._last: bool | None = None

    def record(self, emitting: bool) -> None:
        """Record the radar-emission state for one simulation step."""
        emitting = bool(emitting)
        self.steps += 1
        if emitting:
            self.emitting_steps += 1
        if self._last is not None and emitting != self._last:
            self.transitions += 1
            self.steps_since_toggle = 0
        else:
            self.steps_since_toggle += 1
        self._last = emitting

    @property
    def duty_cycle(self) -> float:
        """Fraction of recorded steps the radar was emitting (1.0 before any step)."""
        return self.emitting_steps / self.steps if self.steps else 1.0

    def summary(self) -> dict[str, float]:
        """Episode-level emission metrics for logging / Pareto analysis."""
        return {
            "emission_duty_cycle": self.duty_cycle,
            "emission_steps": float(self.steps),
            "emission_transitions": float(self.transitions),
        }


__all__ = ["EmissionTracker"]
