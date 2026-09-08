"""Immutable friendly-force information shared over the operational data link."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FriendlyDatalinkReport:
    """Receiver-relative status report broadcast by a friendly platform."""

    source_id: Hashable
    receiver_id: Hashable
    acquisition_time_s: float
    relative_state_enu: tuple[float, ...]
    platform_kind: str
    phase: float = 0.0
    seeker_locked: bool = False
    target_track_id: Hashable | None = None
    lock_track_id: Hashable | None = None
    age_s: float = 0.0

    def __post_init__(self) -> None:
        state = tuple(float(value) for value in self.relative_state_enu)
        if len(state) != 6 or not all(math.isfinite(value) for value in state):
            raise ValueError("A friendly data-link report requires six finite ENU values.")
        kind = str(self.platform_kind).strip().lower()
        if kind not in {"aircraft", "missile"}:
            raise ValueError(f"Unsupported friendly platform kind: {self.platform_kind!r}")
        object.__setattr__(self, "relative_state_enu", state)
        object.__setattr__(self, "platform_kind", kind)
        object.__setattr__(self, "phase", min(1.0, max(0.0, float(self.phase))))
        object.__setattr__(self, "age_s", max(0.0, float(self.age_s)))

    @property
    def is_missile(self) -> bool:
        return self.platform_kind == "missile"
