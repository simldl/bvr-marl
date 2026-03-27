"""Advanced tactical maneuvers for air-to-air combat."""

from air_to_air_rl.automation.scripted_control.maneuvers.bfm_maneuvers import BFMManeuvers
from air_to_air_rl.automation.scripted_control.maneuvers.defensive_maneuvers import (
    DefensiveManeuvers,
)
from air_to_air_rl.automation.scripted_control.maneuvers.offensive_maneuvers import (
    OffensiveManeuvers,
)

__all__ = ["DefensiveManeuvers", "OffensiveManeuvers", "BFMManeuvers"]
