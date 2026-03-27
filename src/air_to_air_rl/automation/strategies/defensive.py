"""
Defensive strategy configuration for the automation system.
Prioritizes survival and defensive actions over offensive engagement.
Optimized for scenarios with AIM-120 AMRAAM (Fox-3) and AIM-9 Sidewinder (Fox-2).
"""

from air_to_air_rl.automation.core.auto_helper import AutoHelperConfig


class DefensiveStrategy:
    """
    Defensive automation strategy focused on survival.

    Designed for scenarios where the enemy uses:
    - Fox-3 missiles (AIM-120 AMRAAM, active radar homing)
    - Fox-2 missiles (AIM-9 Sidewinder, infrared homing)

    Countermeasure priorities:
    - Early deployment to maximize survival
    - Chaff for Fox-3/Fox-1 threats (radar-guided)
    - Flares for Fox-2 threats (IR-guided)
    """

    @staticmethod
    def create_config(enemy_missile_types: list = None) -> AutoHelperConfig:
        """
        Create a defensive configuration.

        Args:
            enemy_missile_types: List of expected enemy missile types
                                ['fox3', 'fox2', 'fox1'] or None for defaults
        """
        config = AutoHelperConfig()

        # Default enemy loadout (modern BVR scenario)
        if enemy_missile_types is None:
            enemy_missile_types = ["fox3", "fox2"]  # AIM-120 + AIM-9

        # Defensive target priority - focus on immediate threats
        config.target_priority_weights = {
            "range": 0.1,  # Range less important
            "threat_level": 0.6,  # Prioritize dangerous targets
            "angle": 0.1,  # Angle less important
            "lock_quality": 0.2,  # Need good lock but not critical
        }

        # Missile-type aware countermeasure deployment
        # Defensive strategy deploys earlier for maximum survival
        if "fox3" in enemy_missile_types or "fox1" in enemy_missile_types:
            # Radar-guided missiles present - liberal chaff usage
            config.threat_response_thresholds = {
                "incoming_missile_distance_m": 7000,  # Deploy early (defensive)
                "radar_lock_time_s": 2.0,  # Low tolerance for locks
                "multiple_threats": 1,  # Deploy with any threat
                "fox3_optimal_timing_min_s": 2.0,  # Deploy chaff 2.0-5.0s before impact
                "fox3_optimal_timing_max_s": 5.0,
            }
        else:
            config.threat_response_thresholds = {
                "incoming_missile_distance_m": 8000,
                "radar_lock_time_s": 2.0,
                "multiple_threats": 1,
            }

        if "fox2" in enemy_missile_types:
            # IR-guided missiles present - liberal flare usage
            config.threat_response_thresholds.update(
                {
                    "fox2_optimal_timing_min_s": 2.0,  # Deploy flares 2.0-5.0s before impact
                    "fox2_optimal_timing_max_s": 5.0,
                }
            )

        # Store missile type awareness
        config.expected_enemy_missiles = enemy_missile_types

        config.automation_level = "defensive"
        config.enable_countermeasures = True
        config.enable_target_selection = True  # Still select targets but defensively
        config.enable_radar_management = True

        return config

    @staticmethod
    def get_description() -> str:
        """Get strategy description."""
        return """
        Defensive Strategy (BVR Optimized):
        - Prioritizes survival over engagement
        - Early countermeasure deployment for maximum effectiveness
        - Focus on immediate high-threat targets
        - Conservative engagement parameters
        - Emphasizes threat neutralization over kill scoring

        Missile-Type Awareness:
        - Deploys chaff against Fox-3 (AIM-120) threats early
        - Deploys flares against Fox-2 (AIM-9) threats early
        - Uses precise timing based on missile tracking data
        - Liberal countermeasure expenditure for survival
        """
