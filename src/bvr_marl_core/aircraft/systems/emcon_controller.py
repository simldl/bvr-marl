"""Scripted radar-emission (EMCON) policies for the active-sensing baselines.

The paper compares the LEARNED radar policy against fixed/heuristic sensing policies
on the same information-fair interface. This controller produces the radar on/off
decision for those baselines; ``learned`` returns ``None`` so the policy's own action
(index 9) is used unchanged. Config-driven via ``env.emcon_policy`` so a baseline is an
eval-time config, not a code change.

Policies
--------
- ``learned``     : no override (use the policy action).
- ``always_on``   : radar always emitting (diagnostic upper bound on information).
- ``always_off``  : radar always silent (rely on data-link / passive).
- ``periodic``    : emit for the first ``duty`` fraction of each ``period_steps`` window.
- ``random``      : emit with per-step probability ``duty`` (matched-duty-cycle control).
- ``heuristic``   : emit when the best track is stale (age > threshold) or uncertain
                    (covariance trace > threshold); stay silent otherwise.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("learned", "always_on", "always_off", "periodic", "random", "heuristic")


class EmconController:
    """Compute the scripted radar-emission decision for one aircraft each step."""

    def __init__(
        self,
        policy: str = "learned",
        *,
        period_steps: int = 10,
        duty: float = 0.5,
        heuristic_age_s: float = 6.0,
        heuristic_cov_trace: float = 5.0e6,
    ) -> None:
        policy = str(policy or "learned").strip().lower()
        if policy not in POLICIES:
            raise ValueError(f"unknown emcon_policy {policy!r}; expected one of {POLICIES}")
        self.policy = policy
        self.period_steps = max(1, int(period_steps))
        self.duty = float(np.clip(duty, 0.0, 1.0))
        self.heuristic_age_s = float(heuristic_age_s)
        self.heuristic_cov_trace = float(heuristic_cov_trace)

    @property
    def is_override(self) -> bool:
        """True when this controller overrides the policy's radar action."""
        return self.policy != "learned"

    def emitting(
        self, *, step: int, unit=None, rng: np.random.Generator | None = None
    ) -> bool | None:
        """Return the forced radar-emission state, or None to defer to the policy."""
        if self.policy == "learned":
            return None
        if self.policy == "always_on":
            return True
        if self.policy == "always_off":
            return False
        if self.policy == "periodic":
            on_steps = round(self.duty * self.period_steps)
            return (int(step) % self.period_steps) < on_steps
        if self.policy == "random":
            gen = rng if rng is not None else np.random.default_rng()
            return bool(gen.random() < self.duty)
        # heuristic
        return self._heuristic_emit(unit)

    def _heuristic_emit(self, unit) -> bool:
        """Emit when the tracked picture is stale or uncertain."""
        tracks = getattr(getattr(unit, "sensor", None), "sensor_tracks", None) or ()
        if not tracks:
            return True  # no picture at all -> emit to acquire
        best_age = min((float(getattr(t, "age_s", 0.0)) for t in tracks), default=0.0)
        if best_age > self.heuristic_age_s:
            return True
        min_trace = min(
            (self._cov_trace(getattr(t, "covariance", None)) for t in tracks),
            default=float("inf"),
        )
        return min_trace > self.heuristic_cov_trace

    @staticmethod
    def _cov_trace(cov) -> float:
        if cov is None:
            return float("inf")
        try:
            arr = np.asarray(cov, dtype=float)
            return float(np.trace(arr[:3, :3]))  # position-block uncertainty
        except (ValueError, TypeError):
            return float("inf")


__all__ = ["EmconController", "POLICIES"]
