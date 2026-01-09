import math
from missiles.guidance.base import BaseGuidanceMode
from simulator.core.helpers import geodetic_bearing_deg
from simulator.utils.angles import yaw_math_to_geo

class LoftGuidance(BaseGuidanceMode):
    MIN_LOFT_ALT = 15000.0
    MAX_LOFT_ALT = 30000.0
    FORCED_PITCH_CLIMB = 75.0

    def compute(missile, target_pos, min_loft_range_m: float = 5000.0, alt_guard_m: float = 1500.0):
        """
        Simple loft: fixed nose-up pitch + geodetic yaw to target, with basic safety checks.
        """
        # 1) Phase check: only loft in boost
        try:
            if getattr(missile.phase_manager, "current_phase", "") != "boost":
                return None  # signal "do not loft"
        except Exception:
            return None

        # 2) Range check
        try:
            # local planar approx for small deltas (consistent with rest of sim)
            dlat = (target_pos.lat - missile.position.lat) * 111000.0
            dlon = (target_pos.lon - missile.position.lon) * 111000.0 * math.cos(math.radians(missile.position.lat))
            dz   = target_pos.alt - missile.position.alt
            slant = math.sqrt(dlat*dlat + dlon*dlon + dz*dz)
            if slant < float(min_loft_range_m):
                return None
        except Exception:
            pass  # if we cannot compute, fall back to allowing loft

        # 3) Altitude guard: don't loft if already far above target
        try:
            if missile.position.alt > (target_pos.alt + float(alt_guard_m)):
                return None
        except Exception:
            pass

        # Loft commands: yaw to geodetic bearing, fixed pitch
        try:
            yaw_geo = geodetic_bearing_deg(missile.position.lat, missile.position.lon,
                                        target_pos.lat, target_pos.lon)
        except Exception:
            # fallback: compute bearing from EN deltas
            yaw_math = math.degrees(math.atan2(dlat, dlon))  # atan2(N,E) -> math yaw
            yaw_geo = yaw_math_to_geo(yaw_math)

        pitch_deg = 75.0
        return float(yaw_geo), float(pitch_deg)
