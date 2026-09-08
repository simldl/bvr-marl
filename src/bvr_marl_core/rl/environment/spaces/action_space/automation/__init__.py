"""Automation modules for weapon systems."""

from bvr_marl_core.rl.environment.spaces.action_space.automation.missile_automation import (
    MissileAutomation,
)
from bvr_marl_core.rl.environment.spaces.action_space.automation.weapon_cooldowns import (
    WeaponCooldowns,
)

__all__ = ["MissileAutomation", "WeaponCooldowns"]
