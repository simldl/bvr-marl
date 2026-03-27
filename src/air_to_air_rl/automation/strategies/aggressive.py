"""
Aggressive strategy configuration for the automation system.
Prioritizes offensive engagement and target elimination.
Optimized for scenarios with AIM-120 AMRAAM (Fox-3) and AIM-9 Sidewinder (Fox-2).
"""

from air_to_air_rl.automation.core.auto_helper import AutoHelperConfig


class AggressiveStrategy:
    """
    Aggressive automation strategy focused on engagement.

    Designed for scenarios where the enemy uses:
    - Fox-3 missiles (AIM-120 AMRAAM, active radar homing)
    - Fox-2 missiles (AIM-9 Sidewinder, infrared homing)

    Countermeasure priorities:
    - Chaff for Fox-3 threats (radar-guided)
    - Flares for Fox-2 threats (IR-guided)
    """

    @staticmethod
    def create_config(enemy_missile_types: list = None) -> AutoHelperConfig:
        """
        Create an aggressive configuration.

        Args:
            enemy_missile_types: List of expected enemy missile types
                                ['fox3', 'fox2', 'fox1'] or None for defaults
        """
        config = AutoHelperConfig()

        # Default enemy loadout (modern BVR scenario)
        if enemy_missile_types is None:
            enemy_missile_types = ["fox3", "fox2"]  # AIM-120 + AIM-9

        # Aggressive target priority - focus on engagement geometry
        config.target_priority_weights = {
            "range": 0.4,  # Range important for engagement
            "threat_level": 0.2,  # Less focus on threat, more on opportunity
            "angle": 0.3,  # Geometry very important for engagement
            "lock_quality": 0.1,  # Basic lock requirement
        }

        # Missile-type aware countermeasure deployment
        # Aggressive strategy deploys later to conserve supplies
        if "fox3" in enemy_missile_types or "fox1" in enemy_missile_types:
            # Radar-guided missiles present - prioritize chaff
            config.threat_response_thresholds = {
                "incoming_missile_distance_m": 2500,  # Deploy later (aggressive)
                "radar_lock_time_s": 4.0,  # Tolerate longer locks
                "multiple_threats": 3,  # Need more threats to deploy
                "fox3_optimal_timing_min_s": 1.5,  # Deploy chaff 1.5-3.5s before impact
                "fox3_optimal_timing_max_s": 3.5,
            }
        else:
            config.threat_response_thresholds = {
                "incoming_missile_distance_m": 3000,
                "radar_lock_time_s": 5.0,
                "multiple_threats": 3,
            }

        if "fox2" in enemy_missile_types:
            # IR-guided missiles present - configure flare timing
            config.threat_response_thresholds.update(
                {
                    "fox2_optimal_timing_min_s": 1.5,  # Deploy flares 1.5-4.0s before impact
                    "fox2_optimal_timing_max_s": 4.0,
                }
            )

        # Store missile type awareness
        config.expected_enemy_missiles = enemy_missile_types

        config.automation_level = "aggressive"
        config.enable_countermeasures = True  # Use countermeasures but conservatively
        config.enable_target_selection = True  # Aggressive target selection
        config.enable_radar_management = True

        return config

    @staticmethod
    def get_description() -> str:
        """Get strategy description."""
        return """
        Aggressive Strategy (BVR Optimized):
        - Prioritizes offensive engagement over defense
        - Focuses on target engagement geometry
        - Conservative countermeasure usage to preserve supplies
        - Seeks engagement opportunities
        - Higher risk tolerance for better positioning

        Missile-Type Awareness:
        - Deploys chaff against Fox-3 (AIM-120) threats
        - Deploys flares against Fox-2 (AIM-9) threats
        - Uses precise timing based on missile tracking data
        - Optimizes countermeasure expenditure for maximum effectiveness
        """
