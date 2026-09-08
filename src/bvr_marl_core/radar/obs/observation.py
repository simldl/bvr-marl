import math

import numpy as np

from bvr_marl_core.radar.core.utils import (
    _doppler,
    _effective_rcs,
    has_effective_earth_line_of_sight,
)
from bvr_marl_core.simulator.core.helpers import Position
from bvr_marl_core.simulator.core.units import Unit
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff

# Default half-width (m/s) of the pulse-Doppler "main-lobe clutter" notch enabled
# on operational radars/seekers. A target whose LOS-projected closing rate falls
# below this is hidden in the zero-Doppler notch (the basis of the beaming/notch
# defensive maneuver). The base Radar keeps the notch disabled (0.0) so generic
# unit tests are unaffected; operational aircraft/missile radars opt in.
DEFAULT_NOTCH_VELOCITY_MPS = 50.0

# Convert a resolution-cell *width* into the 1-sigma measurement uncertainty it
# implies. A measurement known only to lie somewhere inside a cell of width w is
# uniformly distributed over it, so its standard deviation is w/sqrt(12) -- not the
# half-width w/2, which is the *maximum* error and overstates sigma by sqrt(3).
# Reports feed a chi-square association gate, so inflating sigma this way makes
# well-separated aircraft look statistically compatible and merges their tracks.
CELL_WIDTH_TO_SIGMA = 1.0 / math.sqrt(12.0)


def resolution_cell_sigma(cell_width: float) -> float:
    """1-sigma measurement uncertainty implied by a resolution cell of this width."""
    return abs(float(cell_width)) * CELL_WIDTH_TO_SIGMA


class RadarObsGenerator:
    def __init__(
        self,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        max_range_m: float,
        lut,
        snr_threshold_db: float,
        false_alarm_rate: float = 0.0,  # Deprecated - kept for backward compatibility
        np_rng=None,
        device=None,  # Accepted for backwards compatibility; ignored.
        notch_velocity_mps: float = 0.0,
        meas_angular_noise_deg: float = 0.0,
        meas_range_noise_m: float = 0.0,
        doppler_noise_hz: float = 0.0,
    ):
        self.h_fov_deg = horizontal_fov_deg
        self.v_fov_deg = vertical_fov_deg
        self.max_range_m = max_range_m
        self.lut = lut
        self.snr_threshold_db = snr_threshold_db
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng(0)
        self.notch_velocity_mps = float(notch_velocity_mps)
        # Reported-measurement noise (0 = perfect report). Angular noise gives a
        # range-scaled cross-range error; range noise is an along-LOS floor.
        self.meas_angular_noise_deg = float(meas_angular_noise_deg)
        self.meas_range_noise_m = float(meas_range_noise_m)
        self.doppler_noise_hz = max(0.0, float(doppler_noise_hz))
        # Acquisition-only evaluator side channel. It is consumed immediately by
        # Radar for attribution diagnostics and never enters a report or datalink.
        self.last_detection_targets: tuple[object, ...] = ()

    @staticmethod
    def _vel_xyz(v) -> tuple[float, float, float]:
        """Extract ENU components from a velocity (namedtuple, array, or None)."""
        if v is None:
            return (0.0, 0.0, 0.0)
        try:
            return (
                float(getattr(v, "vx", v[0])),
                float(getattr(v, "vy", v[1])),
                float(getattr(v, "vz", v[2])),
            )
        except Exception:
            return (0.0, 0.0, 0.0)

    def _apply_measurement_noise(
        self, az_deg: float, el_deg: float, dist_m: float
    ) -> tuple[float, float, float]:
        """Perturb the reported az/el/d with Gaussian measurement noise.

        Returns the true values unchanged when no noise is configured. Angular
        noise produces a cross-range error that scales with range (the dominant
        surveillance-radar effect); range noise is an along-LOS floor.
        """
        s_ang = self.meas_angular_noise_deg
        s_rng = self.meas_range_noise_m
        if s_ang <= 0.0 and s_rng <= 0.0:
            return az_deg, el_deg, dist_m
        if s_ang > 0.0:
            az_deg = az_deg + float(self.np_rng.normal(0.0, s_ang))
            el_deg = el_deg + float(self.np_rng.normal(0.0, s_ang))
        if s_rng > 0.0:
            dist_m = max(1.0, dist_m + float(self.np_rng.normal(0.0, s_rng)))
        return az_deg, el_deg, dist_m

    def _notch_detection_factor(
        self, dE: float, dN: float, dU: float, dist: float, tgt, own_velocity
    ) -> float:
        """Detection-probability multiplier for the Doppler notch.

        Returns 1.0 outside the notch and falls smoothly to 0.0 as the
        LOS-projected relative (range-rate) velocity approaches zero, so a
        target beaming the radar (range rate ~ 0) becomes effectively invisible.
        """
        vn = self.notch_velocity_mps
        if vn <= 0.0 or own_velocity is None:
            return 1.0
        inv = 1.0 / max(dist, 1e-6)
        r_e, r_n, r_u = dE * inv, dN * inv, dU * inv
        tvx, tvy, tvz = self._vel_xyz(getattr(tgt, "velocity", None))
        ovx, ovy, ovz = self._vel_xyz(own_velocity)
        # Range rate = d||R||/dt = (v_tgt - v_own) · r_hat. Small |.| => in notch.
        range_rate = (tvx - ovx) * r_e + (tvy - ovy) * r_n + (tvz - ovz) * r_u
        a = abs(range_rate)
        if a >= vn:
            return 1.0
        return (a / vn) ** 2

    def generate(
        self,
        pos,
        targets,
        yaw_deg,
        pitch_deg,
        own_group=None,
        own_id=None,
        own_velocity=None,
        dwell_time_s: float = 1.0,
    ):
        dets = []
        detected_targets = []
        h_half = self.h_fov_deg * 0.5
        v_half = self.v_fov_deg * 0.5
        max_rng = self.max_range_m
        cos_lat = math.cos(math.radians(pos.lat))
        for tgt in targets:
            if own_group is not None and hasattr(tgt, "group") and tgt.group == own_group:
                continue
            if own_id is not None and hasattr(tgt, "id") and tgt.id == own_id:
                continue

            # Compute ENU deltas once; reuse for both _angles_dist result and _effective_rcs
            tp = tgt.position
            dN = (tp.lat - pos.lat) * 111_000.0
            dE = (tp.lon - pos.lon) * 111_000.0 * cos_lat
            dU = tp.alt - pos.alt
            hd = math.hypot(dE, dN)
            dist = math.hypot(hd, dU)

            if dist > max_rng:
                continue
            if not has_effective_earth_line_of_sight(hd, pos.alt, tp.alt):
                continue

            az_abs_rad = math.atan2(dE, dN)
            az_abs = math.degrees(az_abs_rad)
            el_abs = math.degrees(math.atan2(dU, max(hd, 1e-12)))
            az_rel = signed_yaw_deg_diff(yaw_deg, az_abs)
            el_rel = el_abs - pitch_deg

            if abs(az_rel) > h_half or abs(el_rel) > v_half:
                continue

            # Pass pre-computed ENU deltas to skip redundant bearing+ENU in _effective_rcs
            eff_rcs = _effective_rcs(tgt, pos, _dE_r2t=dE, _dN_r2t=dN, _dU_r2t=dU)
            p_reference = float(self.lut.get_probability(dist, eff_rcs))
            # Interpret the LUT value as a one-second reference probability and
            # convert it to a time-consistent dwell hazard.
            rate = -math.log(max(1.0 - min(p_reference, 1.0 - 1e-12), 1e-12))
            p = 1.0 - math.exp(-rate * max(float(dwell_time_s), 0.0))
            # Doppler notch: suppress targets beaming the radar (range rate ~ 0).
            p *= self._notch_detection_factor(dE, dN, dU, dist, tgt, own_velocity)
            if self.np_rng.random() > p:
                continue

            dop = _doppler(
                tgt,
                az_abs,
                el_abs,
                self.lut.freq_hz,
                radar_velocity=own_velocity,
            )
            if self.doppler_noise_hz > 0.0:
                dop += float(self.np_rng.normal(0.0, self.doppler_noise_hz))

            # Measurement noise on the *reported* az/el/d (the detection decision
            # above used the true range). Angular noise dominates: cross-range
            # error = sigma_angular(rad) * range, so a long-range surveillance
            # radar (AWACS) reports a position whose error grows with range,
            # while a short-range fighter radar (noise ~ 0) stays crisp.
            az_rep, el_rep, d_rep = self._apply_measurement_noise(az_abs, el_abs, dist)

            snr_db = self.snr_threshold_db + self.np_rng.uniform(2.0, 8.0)
            dets.append({"az": az_rep, "el": el_rep, "d": d_rep, "dop": dop, "snr_db": snr_db})
            detected_targets.append(tgt)

        self.last_detection_targets = tuple(detected_targets)
        return dets

    def detect(
        self, pos: Position, targets: list[Unit], dt: float, steer_h: float, steer_p: float
    ) -> list[dict]:
        if self.use_beam_steering:
            self._scan_time += dt
            period = 10.0
            phase = 2 * math.pi * self._scan_time / period
            self.yaw_offset_deg = steer_h * math.sin(phase)
            self.pitch_offset_deg = steer_p * math.sin(phase)
        else:
            self.yaw_offset_deg = 0.0
            self.pitch_offset_deg = 0.0
        return self._raw_detections(pos, targets)
