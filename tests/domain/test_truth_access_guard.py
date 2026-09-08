"""Runtime truth-access guard (information firewall, paper item 6)."""

import pytest

from bvr_marl_core.domain.truth_access_guard import (
    TruthAccessViolation,
    allow_truth_access,
    forbidden_truth_access,
    resolve_truth_unit,
    truth_access_forbidden,
)


class _Sim:
    def __init__(self):
        self.active_units = {7: "unit-7"}


def test_resolve_works_outside_forbidden_scope():
    assert not truth_access_forbidden()
    assert resolve_truth_unit(_Sim(), 7, reason="oracle") == "unit-7"
    assert resolve_truth_unit(_Sim(), 999, reason="oracle") is None


def test_resolve_raises_inside_forbidden_scope():
    with forbidden_truth_access("sensor_limited_observation"):
        assert truth_access_forbidden()
        with pytest.raises(TruthAccessViolation):
            resolve_truth_unit(_Sim(), 7, reason="oracle_enemy_identity")


def test_allow_re_permits_inside_forbidden():
    with forbidden_truth_access("sensor_limited"):
        with allow_truth_access("evaluator_terminal"):
            assert not truth_access_forbidden()
            assert resolve_truth_unit(_Sim(), 7, reason="evaluator") == "unit-7"
        # forbidden restored after the allow scope exits
        assert truth_access_forbidden()


def test_nested_scopes_restore_state():
    assert not truth_access_forbidden()
    with forbidden_truth_access("a"):
        with forbidden_truth_access("b"):
            assert truth_access_forbidden()
        assert truth_access_forbidden()
    assert not truth_access_forbidden()


def test_violation_message_names_entity_and_reason():
    with forbidden_truth_access(), pytest.raises(TruthAccessViolation) as exc:
        resolve_truth_unit(_Sim(), 42, reason="oracle_enemy_identity")
    assert "42" in str(exc.value) and "oracle_enemy_identity" in str(exc.value)
