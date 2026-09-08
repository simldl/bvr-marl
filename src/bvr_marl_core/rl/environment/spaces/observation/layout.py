"""The resolved observation layout, made explicit so it cannot diverge silently.

The observation width is a product of slot counts and per-token widths, and the slot
counts come from ``config.get(..., <default>)`` -- values that live only in code. So a
checkpoint and a config can disagree about the observation space with nothing recording
it, and the disagreement surfaces as a matrix-shape error deep inside a forward pass:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x175 and 197x512)

That is what happens replaying a checkpoint trained under a different slot count: the
checkpoint wants 197, every config path reproduces 175, and identifying the cause means
reconstructing the arithmetic by hand -- ``num_fm`` had been 6 where the current default
is 4 (2 extra friendly-missile slots x d_FM 11 = the missing 22). No config file recorded
it, because ``num_fm`` was never written to one.

Two traps make this easy to hit again, both in ``gym_components/config.py``:

* ``num_ef`` AUTO-BUMPS by one when an opponent AWACS is present and non-engageable, so
  toggling a scenario flag silently rewrites the observation width and invalidates every
  earlier checkpoint.
* ``num_pr`` defaults to ``num_enemy``, so changing the enemy count moves a second block
  as a side effect.

:func:`observation_layout` computes the whole layout from one config in one place, so it
can be recorded next to a checkpoint, compared across runs, and asserted on load --
turning a shape error into "expected num_fm=6, got 4".
"""

from __future__ import annotations

from bvr_marl_core.rl.environment.spaces.observation.constants import (
    d_EF,
    d_EM,
    d_FF,
    d_FM,
    d_MWS,
    d_PR,
    own_state_dim,
)

__all__ = ["observation_layout", "observation_width", "describe_layout_mismatch"]


def observation_layout(config: dict | None) -> dict[str, int]:
    """Resolve every slot count and block width the observation is built from.

    Mirrors the resolution order in ``gym_components/config.py`` deliberately: this is a
    description of what that code produces, so if the two drift the layout test fails.
    """
    config = config or {}
    scenario = config.get("scenario_config") or {}
    awacs = scenario.get("awacs_config") or {}
    # Non-engageable defaults True to match the environment config dataclass.
    awacs_non_engageable = bool(awacs.get("awacs_non_engageable", True))
    opponent_awacs = bool(awacs.get("opponent_awacs", False))

    num_enemy = int(config.get("num_enemy", 2))
    num_fm = int(config.get("num_fm", 4))
    num_ff = int(config.get("num_ff", 2))
    num_em = int(config.get("num_em", 4))
    # The auto-bump: a dedicated enemy slot for a non-engageable opponent AWACS.
    num_ef = num_enemy + (1 if opponent_awacs and awacs_non_engageable else 0)
    num_pr = int(config.get("pr_slots", num_enemy))
    num_warn = int(config.get("warn_sectors", 4))
    d_own = own_state_dim(bool(config.get("emcon_action_enabled", False)))

    blocks = {
        "own_state": d_own,
        "friendly_missiles": num_fm * d_FM,
        "friendly_fighters": num_ff * d_FF,
        "enemy_missiles": num_em * d_EM,
        "enemy_fighters": num_ef * d_EF,
        "missile_warnings": num_warn * d_MWS,
        "passive_radar": num_pr * d_PR,
    }
    return {
        "num_fm": num_fm,
        "num_ff": num_ff,
        "num_em": num_em,
        "num_ef": num_ef,
        "num_pr": num_pr,
        "num_warn": num_warn,
        "own_state_dim": d_own,
        **{f"block_{name}": size for name, size in blocks.items()},
        "total": sum(blocks.values()),
    }


def observation_width(config: dict | None) -> int:
    """Total flattened observation width implied by ``config``."""
    return observation_layout(config)["total"]


def describe_layout_mismatch(expected: dict[str, int], actual: dict[str, int]) -> str:
    """Human-readable diff between two layouts, or ``""`` when they agree.

    The point is to name the responsible knob. "expected num_fm=6, got 4" is actionable;
    "1x175 and 197x512" costs an afternoon.
    """
    if expected == actual:
        return ""
    keys = sorted(set(expected) | set(actual))
    differing = [
        f"  {key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in keys
        if expected.get(key) != actual.get(key)
    ]
    return "Observation layout mismatch:\n" + "\n".join(differing)
