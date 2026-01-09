"""
Observation information provider.
Describes recommended observation components for neural networks.
"""
from typing import Dict, Any


class ObservationInfoProvider:
    """Provide information about observation structure."""

    @staticmethod
    def get_observation_info() -> Dict[str, Any]:
        """
        Get information about what should be included in observations.

        Returns:
            Dict describing recommended observation components
        """
        return {
            'aircraft_state': [
                'position (lat, lon, alt)',
                'velocity (vx, vy, vz)',
                'attitude (yaw, pitch, roll)',
                'speed',
                'throttle_setting'
            ],
            'sensors': [
                'radar_detections',
                'passive_radar_contacts',
                'missile_warning_system',
                'rwr_threats'
            ],
            'weapons': [
                'remaining_missiles',
                'missile_types_available',
                'gun_ammunition',
                'weapons_ready'
            ],
            'countermeasures': [
                'flares_remaining',
                'chaff_remaining',
                'ecm_available',
                'decoys_remaining'
            ],
            'tactical': [
                'current_target_info',
                'threat_summary',
                'dlz_zones',
                'nez_status'
            ],
            'automation_feedback': [
                'recommended_target',
                'threat_level',
                'countermeasure_recommendations',
                'automation_status'
            ]
        }
