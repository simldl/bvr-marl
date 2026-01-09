"""
Balanced strategy configuration for the automation system.
Balances offensive and defensive capabilities for general-purpose combat.
Optimized for scenarios with AIM-120 AMRAAM (Fox-3) and AIM-9 Sidewinder (Fox-2).
"""
from automation.core.auto_helper import AutoHelperConfig


class BalancedStrategy:
    """
    Balanced automation strategy for general combat.

    Designed for scenarios where the enemy uses:
    - Fox-3 missiles (AIM-120 AMRAAM, active radar homing)
    - Fox-2 missiles (AIM-9 Sidewinder, infrared homing)

    Countermeasure priorities:
    - Balanced deployment timing
    - Chaff for Fox-3/Fox-1 threats (radar-guided)
    - Flares for Fox-2 threats (IR-guided)
    """

    @staticmethod
    def create_config(enemy_missile_types: list = None) -> AutoHelperConfig:
        """
        Create a balanced configuration.

        Args:
            enemy_missile_types: List of expected enemy missile types
                                ['fox3', 'fox2', 'fox1'] or None for defaults
        """
        config = AutoHelperConfig()

        # Default enemy loadout (modern BVR scenario)
        if enemy_missile_types is None:
            enemy_missile_types = ['fox3', 'fox2']  # AIM-120 + AIM-9

        # Balanced target priority
        config.target_priority_weights = {
            'range': 0.3,           # Reasonable range consideration
            'threat_level': 0.4,    # Moderate threat focus
            'angle': 0.2,           # Some geometry consideration
            'lock_quality': 0.1     # Basic lock requirement
        }

        # Missile-type aware countermeasure deployment
        # Balanced strategy uses optimal timing windows
        if 'fox3' in enemy_missile_types or 'fox1' in enemy_missile_types:
            # Radar-guided missiles present - balanced chaff usage
            config.threat_response_thresholds = {
                'incoming_missile_distance_m': 5000,    # Standard deployment
                'radar_lock_time_s': 3.0,              # Standard tolerance
                'multiple_threats': 2,                  # Deploy with 2+ threats
                'fox3_optimal_timing_min_s': 1.5,      # Deploy chaff 1.5-4.0s before impact
                'fox3_optimal_timing_max_s': 4.0,
            }
        else:
            config.threat_response_thresholds = {
                'incoming_missile_distance_m': 5000,
                'radar_lock_time_s': 3.0,
                'multiple_threats': 2,
            }

        if 'fox2' in enemy_missile_types:
            # IR-guided missiles present - balanced flare usage
            config.threat_response_thresholds.update({
                'fox2_optimal_timing_min_s': 1.5,  # Deploy flares 1.5-4.0s before impact
                'fox2_optimal_timing_max_s': 4.0,
            })

        # Store missile type awareness
        config.expected_enemy_missiles = enemy_missile_types

        config.automation_level = 'balanced'
        config.enable_countermeasures = True
        config.enable_target_selection = True
        config.enable_radar_management = True

        return config

    @staticmethod
    def get_description() -> str:
        """Get strategy description."""
        return """
        Balanced Strategy (BVR Optimized):
        - Balances offensive and defensive actions
        - Optimal countermeasure deployment timing
        - Considers multiple factors in target selection
        - Adaptable to various combat situations
        - Good general-purpose configuration

        Missile-Type Awareness:
        - Deploys chaff against Fox-3 (AIM-120) threats optimally
        - Deploys flares against Fox-2 (AIM-9) threats optimally
        - Uses precise timing based on missile tracking data
        - Balanced countermeasure expenditure for effectiveness
        """