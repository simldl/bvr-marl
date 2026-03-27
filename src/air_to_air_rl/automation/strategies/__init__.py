"""Automation strategies."""

from air_to_air_rl.automation.strategies.aggressive import AggressiveStrategy
from air_to_air_rl.automation.strategies.balanced import BalancedStrategy
from air_to_air_rl.automation.strategies.defensive import DefensiveStrategy

__all__ = ["DefensiveStrategy", "AggressiveStrategy", "BalancedStrategy"]
