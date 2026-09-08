"""Proximity-fuze submodel (P_fuze)."""

from __future__ import annotations

from typing import Any

from bvr_marl_core.simulator.core.effectiveness.common import attr_float, clamp01


class FuzeModel:
    """P_fuze — probability the proximity fuze functions correctly.

    The arming gate (an unarmed fuze cannot detonate) is enforced upstream in
    the substepper before ``on_hit`` is ever called, so by the time this runs
    the fuze is armed and the missile is within the fuse radius. What remains is
    the fuze's intrinsic reliability, read from ``missile.fuze_reliability``
    (default 1.0 — set per missile in Phase 5).
    """

    def probability(self, missile: Any, miss_distance_m: float | None = None) -> float:
        return clamp01(attr_float(missile, "fuze_reliability", 1.0))
