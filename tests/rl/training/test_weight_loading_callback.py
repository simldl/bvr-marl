from __future__ import annotations

import pytest

from bvr_marl_core.rl.training.callbacks.weight_loading import WeightLoadingCallback


class _Module:
    def __init__(self) -> None:
        self.state = None

    def set_state(self, state):
        self.state = state


def test_weight_loading_maps_shared_policy_to_team_policies() -> None:
    target = {
        "attacker_policy": _Module(),
        "defender_policy": _Module(),
    }
    callback = WeightLoadingCallback(
        "unused",
        policy_mapping={
            "from_checkpoint_policy": "shared_policy",
            "to_stage_policies": ["attacker_policy", "defender_policy"],
        },
        strict_policy_mapping=True,
    )

    callback._load_policy_states(target, {"shared_policy": {"w": 1}})

    assert target["attacker_policy"].state == {"w": 1}
    assert target["defender_policy"].state == {"w": 1}


def test_weight_loading_raises_when_strict_mapping_matches_no_policy() -> None:
    callback = WeightLoadingCallback(
        "unused",
        policy_mapping={"mode": "direct_policy_id_match"},
        strict_policy_mapping=True,
    )

    with pytest.raises(RuntimeError, match="did not find source weights"):
        callback._load_policy_states({"attacker_policy": _Module()}, {"shared_policy": {"w": 1}})
