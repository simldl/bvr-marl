import math

import numpy as np
import torch

from air_to_air_rl.radar.core.utils import _angles_dist, _doppler, _effective_rcs
from air_to_air_rl.simulator.core.helpers import Position
from air_to_air_rl.simulator.core.units import Unit
from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff, yaw_geo_to_math

_CUDA_AVAILABLE: bool = torch.cuda.is_available()


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
        device=None,
    ):
        self.h_fov_deg = horizontal_fov_deg
        self.v_fov_deg = vertical_fov_deg
        self.max_range_m = max_range_m
        self.lut = lut
        self.snr_threshold_db = snr_threshold_db
        self.device = device or torch.device("cuda" if _CUDA_AVAILABLE else "cpu")
        self.np_rng = np_rng or np.random.default_rng()

    def generate(self, pos, targets, yaw_deg, pitch_deg, own_group=None, own_id=None):
        dets = []
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

            az_abs_rad = math.atan2(dE, dN)
            az_abs = math.degrees(az_abs_rad)
            el_abs = math.degrees(math.atan2(dU, max(hd, 1e-12)))
            az_rel = signed_yaw_deg_diff(yaw_deg, az_abs)
            el_rel = el_abs - pitch_deg

            if abs(az_rel) > h_half or abs(el_rel) > v_half:
                continue

            # Pass pre-computed ENU deltas to skip redundant bearing+ENU in _effective_rcs
            eff_rcs = _effective_rcs(tgt, pos, _dE_r2t=dE, _dN_r2t=dN, _dU_r2t=dU)
            p = self.lut.get_probability(dist, eff_rcs)
            if self.np_rng.random() > p:
                continue

            dop = _doppler(tgt, az_abs, el_abs, self.lut.freq_hz)

            snr_db = self.snr_threshold_db + self.np_rng.uniform(2.0, 8.0)
            dets.append(
                {"T": tgt, "az": az_abs, "el": el_abs, "d": dist, "dop": dop, "snr_db": snr_db}
            )

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
