"""The can-fire flag's index must stay pinned to what the policy side reads.

``OWN_IDX_CAN_FIRE`` is consumed by the fire-action gradient mask, which slices it out
of a flattened observation. If the ownship layout is reordered and this constant is not
updated, the mask silently starts keying on the wrong feature -- a failure that would
look like "the masking didn't help" rather than like a bug.
"""

from __future__ import annotations

from bvr_marl_core.rl.environment.spaces.observation.constants import (
    OWN_IDX_CAN_FIRE,
    d_OWN,
    own_state_dim,
)


def test_can_fire_index_is_inside_the_ownship_vector():
    assert 0 <= OWN_IDX_CAN_FIRE < d_OWN


def test_can_fire_index_matches_the_documented_layout():
    # The block comment under d_OWN is the layout contract: "[8]: Can-fire-now flag".
    # Read it from SOURCE, not __doc__: d_OWN is an int, so the string literal after the
    # assignment is discarded at runtime and __doc__ returns int's own docstring.
    import inspect

    from bvr_marl_core.rl.environment.spaces.observation import constants

    source = inspect.getsource(constants)
    marker = f"[{OWN_IDX_CAN_FIRE}]:"
    line = next((ln for ln in source.splitlines() if ln.strip().startswith(marker)), None)

    assert line is not None, f"ownship layout documents no {marker} entry"
    assert "can-fire" in line.lower(), f"index {OWN_IDX_CAN_FIRE} documents: {line.strip()!r}"


def test_index_is_stable_under_the_emcon_extension():
    # EMCON widens the vector by appending, so the can-fire slot must not move.
    assert own_state_dim(True) > own_state_dim(False)
    assert OWN_IDX_CAN_FIRE < own_state_dim(False)
