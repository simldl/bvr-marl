from __future__ import annotations

from types import SimpleNamespace

from bvr_marl_core.rl.environment.spaces.action_space.automation.missile_automation import (
    MissileAutomation,
)


def test_missile_automation_can_be_suppressed_per_unit():
    automation = MissileAutomation(enable=True, sqi_threshold=0.1)
    automation.set_suppressed_unit_ids([42])

    should_fire, reason = automation.should_auto_fire(
        SimpleNamespace(id=42),
        selected_target=SimpleNamespace(id=7),
    )

    assert should_fire is False
    assert reason == "automation_suppressed_for_unit"
