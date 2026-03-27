"""
Semi-automated helper system for air-to-air combat.

This module provides intelligent automation for all aircraft actions except:
- Throttle control
- Yaw (heading) control
- Pitch control
- Missile firing decision

The system handles:
- Target selection and prioritization
- Countermeasure deployment (flares, chaff, ECM, decoys)
- Radar and sensor management
- Threat assessment and defensive actions
"""

from air_to_air_rl.automation.core.auto_helper import AutoHelper

__all__ = ["AutoHelper"]
