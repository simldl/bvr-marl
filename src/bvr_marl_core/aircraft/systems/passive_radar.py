import math

import numpy as np

from bvr_marl_core.simulator.core.units import Unit


class PassiveRadar:
    def __init__(
        self, angular_error_deg: float = 5.0, range_error_m: float = 2000.0, max_age_s: float = 3.0
    ):
        self.angular_error_deg = angular_error_deg
        self.range_error_m = range_error_m
        self.detections = []
        self.max_age_s = max_age_s

    def receive_emission(self, emitter: Unit, receiver: Unit, timestamp: float):
        """
        Bearing & slant range from receiver to emitter using local ENU approximation.
        - Properly scales longitudinal difference by cos(latitude)
        - Produces azimuth in geodetic yaw convention (0°=North, CW+)
        - Adds Gaussian errors per configured stddevs
        """
        lat_rcv = float(getattr(receiver.position, "lat", 0.0))
        dlat_deg = float(getattr(emitter.position, "lat", 0.0) - lat_rcv)
        dlon_deg = float(
            getattr(emitter.position, "lon", 0.0) - getattr(receiver.position, "lon", 0.0)
        )
        dz_m = float(getattr(emitter.position, "alt", 0.0) - getattr(receiver.position, "alt", 0.0))

        # Flat-earth small-angle ENU (meters)
        cos_lat = math.cos(math.radians(lat_rcv))
        dE = dlon_deg * 111_000.0 * cos_lat
        dN = dlat_deg * 111_000.0

        # Geodetic yaw: atan2(E, N) → 0°=North, CW positive
        azimuth_deg = (math.degrees(math.atan2(dE, dN)) + 360.0) % 360.0

        # Relative angle in receiver’s nose frame + noise, wrapped to [-180,180]
        err = float(np.random.normal(0.0, self.angular_error_deg))
        rel_angle = (
            azimuth_deg - float(getattr(receiver, "yaw_deg", 0.0)) + err + 540.0
        ) % 360.0 - 180.0

        # Slant range with vertical component + range noise
        slant = math.sqrt(dE * dE + dN * dN + dz_m * dz_m)
        dist_m = max(0.0, slant + float(np.random.normal(0.0, self.range_error_m)))

        detection = {
            "emitter_id": getattr(emitter, "id", None),
            "rel_angle_deg": rel_angle,
            "distance_m": dist_m,
            "timestamp": float(timestamp),
        }
        self.detections.append(detection)
        return detection

    def get_observation(self, current_time: float) -> list[dict]:
        obs = []
        for det in self.detections:
            age = current_time - det["timestamp"]
            if 0 <= age <= self.max_age_s:
                obs.append(
                    {
                        "emitter_id": det["emitter_id"],
                        "rel_angle_deg": det["rel_angle_deg"],
                        "distance_m": det["distance_m"],
                        "age_s": age,
                    }
                )
        return obs

    def clear_old(self, current_time: float):
        self.detections = [
            d for d in self.detections if (current_time - d["timestamp"]) <= self.max_age_s
        ]
