import math

from air_to_air_rl.missiles.guidance.base import BaseGuidanceMode
from air_to_air_rl.simulator.core.helpers import geodetic_bearing_deg
from air_to_air_rl.simulator.utils.angles import yaw_math_to_geo


class LoftGuidance(BaseGuidanceMode):
    MIN_LOFT_ALT = 15000.0
    MAX_LOFT_ALT = 30000.0
    FORCED_PITCH_CLIMB = 75.0

    def compute(
        self, missile, target_pos, min_loft_range_m: float = 5000.0, alt_guard_m: float = 1500.0
    ):
        """
        Simple loft: fixed nose-up pitch + geodetic yaw to target, with basic safety checks.
        """
        try:
            if getattr(missile.phase_manager, "current_phase", "") != "boost":
                return None
        except Exception:
            return None

        try:
            # Local planar approximation (consistent with the rest of the sim)
            dlat = (target_pos.lat - missile.position.lat) * 111000.0
            dlon = (
                (target_pos.lon - missile.position.lon)
                * 111000.0
                * math.cos(math.radians(missile.position.lat))
            )
            dz = target_pos.alt - missile.position.alt
            slant = math.sqrt(dlat * dlat + dlon * dlon + dz * dz)
            if slant < float(min_loft_range_m):
                return None
        except Exception:
            pass

        try:
            if missile.position.alt > (target_pos.alt + float(alt_guard_m)):
                return None
        except Exception:
            pass

        try:
            yaw_geo = geodetic_bearing_deg(
                missile.position.lat, missile.position.lon, target_pos.lat, target_pos.lon
            )
        except Exception:
            yaw_math = math.degrees(math.atan2(dlat, dlon))
            yaw_geo = yaw_math_to_geo(yaw_math)

        pitch_deg = 75.0
        return float(yaw_geo), float(pitch_deg)
