"""Per-episode aggregation of missile launch geometry and terminal outcomes.

The capture side and the metric keys live in
``bvr_marl_core.domain.launch_geometry`` so the action space can import them without
pulling the environment in through ``gym``.
"""

from __future__ import annotations

from typing import Any

from bvr_marl_core.domain.launch_geometry import (
    DETONATION_RATE,
    LAUNCH_ALT_DELTA_M,
    LAUNCH_ASPECT_DEG,
    LAUNCH_CLOSURE_MPS,
    LAUNCH_RANGE_KM,
    LAUNCH_SHOOTER_SPEED,
    MISSILE_DIAGNOSTIC_KEYS,
    NO_TERMINAL_RATE,
    TERMINAL_MISS_M,
    TERMINAL_PK,
    capture_launch_geometry,
)

__all__ = ["MissileDiagnosticsCollector", "MISSILE_DIAGNOSTIC_KEYS", "capture_launch_geometry"]


class MissileDiagnosticsCollector:
    """Per-episode accumulator for launch geometry and terminal outcomes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._launches: list[dict[str, float]] = []
        self._launch_count = 0
        self._seen_events: set[int] = set()
        self._terminal_records: list[dict[str, Any]] = []

    def record_launch(self, geometry: dict[str, float] | None) -> None:
        """Count a launch, with its geometry when it could be captured."""
        self._launch_count += 1
        if geometry:
            self._launches.append(geometry)

    def collect_terminal_events(self, simulator, shooter_group: str | None = None) -> None:
        """Drain new ``MissileTerminalEvent`` records from the simulator.

        Idempotent by event identity, so it is safe to call every step -- the event list
        is cumulative over the episode and the alternative (reading it once at episode
        end) breaks whenever an episode terminates early.
        """
        events = getattr(simulator, "events", None) or ()
        for event in events:
            if type(event).__name__ != "MissileTerminalEvent":
                continue
            key = id(event)
            if key in self._seen_events:
                continue
            self._seen_events.add(key)
            record = getattr(event, "record", None)
            if not isinstance(record, dict):
                continue
            if shooter_group is not None:
                group = record.get("shooter_group") or record.get("launch_group")
                if group is not None and group != shooter_group:
                    continue
            self._terminal_records.append(record)

    def episode_metrics(self) -> dict[str, float]:
        """Aggregate to per-episode values, omitting keys with nothing to report.

        Omission is deliberate: logging 0.0 for "median launch range" on an episode
        where nobody fired would drag the mean toward zero and read as "fires at point
        blank", which is the opposite of the truth.
        """
        out: dict[str, float] = {}

        if self._launches:
            for key in (
                LAUNCH_RANGE_KM,
                LAUNCH_ASPECT_DEG,
                LAUNCH_ALT_DELTA_M,
                LAUNCH_SHOOTER_SPEED,
                LAUNCH_CLOSURE_MPS,
            ):
                values = [entry[key] for entry in self._launches if key in entry]
                if values:
                    out[key] = sum(values) / len(values)

        if self._launch_count > 0:
            detonations = len(self._terminal_records)
            out[DETONATION_RATE] = detonations / self._launch_count
            # The complement, logged explicitly: "never arrived" is the failure mode
            # that carries no terminal event and therefore no other trace at all.
            out[NO_TERMINAL_RATE] = max(0.0, 1.0 - detonations / self._launch_count)

        if self._terminal_records:
            misses = [
                float(record["miss_distance_m"])
                for record in self._terminal_records
                if record.get("miss_distance_m") is not None
            ]
            if misses:
                out[TERMINAL_MISS_M] = sum(misses) / len(misses)
            pks = [
                float(record["pk"])
                for record in self._terminal_records
                if record.get("pk") is not None
            ]
            if pks:
                out[TERMINAL_PK] = sum(pks) / len(pks)

        return out
