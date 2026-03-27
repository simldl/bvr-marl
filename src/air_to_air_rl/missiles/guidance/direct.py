import math

from air_to_air_rl.simulator.core.helpers import geodetic_bearing_deg, geodetic_distance_km


class DirectPursuitGuidance:
    def __init__(self, missile):
        self.missile = missile

    def compute(
        self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
    ):
        """
        Direct pursuit guidance - aims directly at target.
        Simple and reliable for most scenarios.
        """
        desired_yaw_deg = geodetic_bearing_deg(
            missile_position.lat, missile_position.lon, target_position.lat, target_position.lon
        )
        horizontal_distance = (
            geodetic_distance_km(
                missile_position.lat,
                missile_position.lon,
                0.0,
                target_position.lat,
                target_position.lon,
                0.0,
            )
            * 1000.0
        )
        vertical_diff = target_position.alt - missile_position.alt
        desired_pitch_deg = math.degrees(math.atan2(vertical_diff, max(1e-3, horizontal_distance)))
        return desired_yaw_deg, desired_pitch_deg
