"""Warhead-lethality submodel (P_wh)."""

from __future__ import annotations

import math
from typing import Any

from bvr_marl_core.simulator.core.effectiveness.common import attr_float, clamp01


class WarheadModel:
    """P_wh — probability the warhead produces a lethal effect at the achieved
    closest-approach miss distance.

    Carries the Gaussian fall-off in miss distance that the legacy scalar model
    used directly:

        P_wh(d) = warhead_effectiveness * exp(-(d / lethal_radius_m)^2)

    ``warhead_effectiveness`` defaults to 1.0. The legacy base
    ``hit_probability`` is intentionally not applied after a geometric
    intercept; otherwise a successful intercept is penalized a second time.
    A non-positive ``lethal_radius_m`` (or a missing miss distance) yields flat
    effectiveness.
    """

    def probability(self, missile: Any, miss_distance_m: float | None = None) -> float:
        eff = getattr(missile, "warhead_effectiveness", None)
        if eff is None:
            eff = 1.0
        else:
            eff = attr_float(missile, "warhead_effectiveness", 1.0)

        if miss_distance_m is None:
            return clamp01(eff)

        lethal_r = attr_float(missile, "lethal_radius_m", 0.0)
        if lethal_r <= 0.0:
            return clamp01(eff)

        d = max(0.0, float(miss_distance_m))
        return clamp01(eff * math.exp(-((d / lethal_r) ** 2)))
