"""Infrared Search and Track (IRST): a passive, angle-only IR sensor.

Unlike the radar, the IRST emits nothing (no RWR exposure), does not care about a
target's radar cross-section (so it sees stealth aircraft), and measures only a
bearing, not range. Detection range is driven by the target's IR signature:

  * aspect  — the exhaust is hottest from the rear, coolest head-on;
  * afterburner — a lit burner is a huge IR source;
  * speed   — skin friction warms a fast airframe.

Each detection is emitted as a bearing-only "strobe" (range denied) so it flows
through the same datalink fusion + cross-radar triangulation as a jammer strobe:
one IRST gives a bearing-only track, two datalinked IRST fighters triangulate a
full 3-D track. The IRST does not exist on every airframe (e.g. the F-22 omits it).
"""

import math

from bvr_marl_core.radar.core.utils import enu_to_geodetic, to_cart
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff


def in_afterburner(target) -> bool:
    """True if the target's afterburner is lit (spool blend high, else throttle)."""
    ab = getattr(getattr(target, "physics", None), "afterburner", None)
    if ab is not None and hasattr(ab, "_blend"):
        try:
            return float(ab._blend) > 0.5
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
    return float(getattr(getattr(target, "control", None), "throttle", 0.0)) > 0.85


def _aspect_deg(target, obs_lat, obs_lon) -> float:
    """Angle between the target's tail (exhaust) direction and the LOS to the IRST;
    ~0 = looking up the tailpipe (hot), ~180 = head-on (cold)."""
    from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg

    bearing_tgt_to_obs = geodetic_bearing_deg(
        target.position.lat, target.position.lon, obs_lat, obs_lon
    )
    tail_heading = (float(getattr(target, "yaw_deg", 0.0)) + 180.0) % 360.0
    return abs(signed_yaw_deg_diff(tail_heading, bearing_tgt_to_obs))


class IRSTSensor:
    def __init__(
        self, owner, fov_deg=140.0, base_range_m=45_000.0, angular_noise_deg=0.4, np_rng=None
    ):
        self.owner = owner
        self.fov_deg = float(fov_deg)
        self.base_range_m = float(base_range_m)
        self.angular_noise_deg = float(angular_noise_deg)
        self.np_rng = np_rng

    def ir_detection_range(self, target, obs_lat, obs_lon) -> float:
        """Range (m) at which this target's IR signature is detectable."""
        aspect = _aspect_deg(target, obs_lat, obs_lon)
        if aspect < 45.0:
            asp = 2.2  # rear: hot exhaust
        elif aspect < 135.0:
            asp = 1.0  # beam
        else:
            asp = 0.4  # nose: cold intake
        ab = 2.2 if in_afterburner(target) else 1.0
        spd = 0.8 + float(getattr(target, "speed", 250.0)) / 600.0
        return self.base_range_m * asp * ab * spd

    def generate(self, owner_position, targets, yaw_deg, pitch_deg) -> list[dict]:
        """Bearing-only IR detections of enemy aircraft, as range-denied strobes."""
        dets = []
        h_half = self.fov_deg * 0.5
        v_half = self.fov_deg * 0.5
        cos_lat = math.cos(math.radians(owner_position.lat))
        own_group = getattr(self.owner, "group", None)
        own_id = getattr(self.owner, "id", None)
        for tgt in targets:
            if getattr(tgt, "is_missile", False) or getattr(tgt, "is_countermeasure", False):
                continue
            if own_group is not None and getattr(tgt, "group", None) == own_group:
                continue
            if own_id is not None and getattr(tgt, "id", None) == own_id:
                continue
            tp = tgt.position
            dN = (tp.lat - owner_position.lat) * 111_000.0
            dE = (tp.lon - owner_position.lon) * 111_000.0 * cos_lat
            dU = tp.alt - owner_position.alt
            hd = math.hypot(dE, dN)
            dist = math.hypot(hd, dU)

            if dist > self.ir_detection_range(tgt, owner_position.lat, owner_position.lon):
                continue

            az_abs = math.degrees(math.atan2(dE, dN))
            el_abs = math.degrees(math.atan2(dU, max(hd, 1e-12)))
            if (
                abs(signed_yaw_deg_diff(yaw_deg, az_abs)) > h_half
                or abs(el_abs - pitch_deg) > v_half
            ):
                continue

            if self.np_rng is not None and self.angular_noise_deg > 0.0:
                az_abs += float(self.np_rng.normal(0.0, self.angular_noise_deg))
                el_abs += float(self.np_rng.normal(0.0, self.angular_noise_deg))

            # Bearing-only: range is denied (angle-only sensor). Placeholder range so
            # the strobe machinery can build a bearing / triangulate across IRSTs.
            strobe_range = self.base_range_m
            enu = to_cart(az_abs, el_abs, strobe_range)
            lat, lon, alt = enu_to_geodetic(
                enu, owner_position.lat, owner_position.lon, owner_position.alt
            )
            dets.append(
                {
                    "T": tgt,
                    "az": az_abs,
                    "el": el_abs,
                    "d": strobe_range,
                    "dop": 0.0,
                    "snr_db": 20.0,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "range_denied": True,
                    "strobe_az": az_abs,
                    "strobe_el": el_abs,
                    "obs_lat": owner_position.lat,
                    "obs_lon": owner_position.lon,
                    "obs_alt": owner_position.alt,
                    "ir_contact": True,
                }
            )
        return dets
