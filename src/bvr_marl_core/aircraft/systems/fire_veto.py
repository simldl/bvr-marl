"""Attribution of a wasted trigger press to the launch gate that rejected it.

``vetoed_missile_wasted`` counts a permissible trigger press that the weapon model
then rejected. As a single number it says the shot died on geometry but not WHICH
gate killed it, and the three gates have unrelated fixes: FOV is a pointing problem,
range is a closure problem, lock is a sensor problem. A collapsed counter sitting at
tens of presses per episode with zero launches is enough to rule out target selection
and nothing else.

The three conditions come off ``FireGates``, the SAME shared evaluator that produces
the observation's ``can_fire`` bit and the launch gate itself, so the attribution
cannot drift from what actually blocked the shot.

**These are conditions, not a partition.** A press taken out of FOV *and* beyond
radar range increments both, so the subtotals do NOT sum to ``vetoed_missile_wasted``
and each is instead bounded by it. That is deliberate: a partition would have to pick
a winner by gate ordering, which hides that two gates are failing together and makes
fixing the reported one move the number nowhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bvr_marl_core.aircraft.systems.fire_feasibility import FireGates

VETO_CATEGORY_FOV = "fov"
VETO_CATEGORY_RANGE = "range"
VETO_CATEGORY_LOCK = "lock"

WASTED_VETO_CATEGORIES: tuple[str, ...] = (
    VETO_CATEGORY_FOV,
    VETO_CATEGORY_RANGE,
    VETO_CATEGORY_LOCK,
)


def wasted_categories_from_gates(gates: FireGates | None) -> dict[str, int]:
    """Which of the three gates were unmet at the moment of a vetoed press.

    ``fov`` folds in the gimbal limit: both are "the weapon cannot be brought to
    bear", and both are fixed by pointing the aircraft differently. ``range`` uses
    ``launch_range_ok`` rather than ``radar_range_ok`` so a datalink-cued shot beyond
    own radar range is not miscounted as too far -- that shot is legal.

    A missing ``gates`` yields all-zero rather than raising: an attribution failure
    must never be able to drop the press from the collapsed counter.
    """
    if gates is None:
        return {category: 0 for category in WASTED_VETO_CATEGORIES}
    return {
        VETO_CATEGORY_FOV: int(not (gates.target_in_fov and gates.gimbal_ok)),
        VETO_CATEGORY_RANGE: int(not gates.launch_range_ok),
        VETO_CATEGORY_LOCK: int(not gates.has_lock),
    }


def team_wasted_info_key(team: str, category: str) -> str:
    """Episode-``info`` key for one team's per-gate wasted count."""
    return f"team_{team}_vetoed_missile_wasted_{category}"


# The exact set the training callback reserves in progress.csv's header. Both repos
# must agree on it or the columns silently go missing (see ADDING_A_METRIC.md,
# trap 2), so it is exported through bvr_marl_core.rl.api rather than rebuilt.
TEAM_A_WASTED_VETO_KEYS: tuple[str, ...] = tuple(
    team_wasted_info_key("a", category) for category in WASTED_VETO_CATEGORIES
)


def wasted_category_key(category: str) -> str:
    """Per-step agent-state key for one gate."""
    return f"vetoed_missile_wasted_{category}_this_step"


def wasted_signal_key(category: str) -> str:
    """Training-signal key for one gate (the per-step flag, unsuffixed)."""
    return f"vetoed_missile_wasted_{category}"
