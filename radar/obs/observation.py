import numpy as np
import torch
import math
from typing import List
from radar.core.utils import _doppler, _angles_dist, _effective_rcs
from simulator.utils.angles import yaw_geo_to_math, signed_yaw_deg_diff
from simulator.core.helpers import Position
from simulator.core.units import Unit

class RadarObsGenerator:
    def __init__(
        self,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        max_range_m: float,
        lut,                 # DetectionLUT-Objekt!
        snr_threshold_db: float,
        false_alarm_rate: float = 0.0,  # Deprecated - kept for backward compatibility
        np_rng=None,
        device=None
    ):
        self.h_fov_deg = horizontal_fov_deg
        self.v_fov_deg = vertical_fov_deg
        self.max_range_m = max_range_m
        self.lut = lut
        self.snr_threshold_db = snr_threshold_db
        # false_alarm_rate is deprecated - ECM system now handles deception/ghosts
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.np_rng = np_rng or np.random.default_rng()

    def generate(self, pos, targets, yaw_deg, pitch_deg, own_group=None, own_id=None):
        dets = []
        for tgt in targets:
            if own_group is not None and hasattr(tgt, "group") and tgt.group == own_group:
                continue
            if own_id is not None and hasattr(tgt, "id") and tgt.id == own_id:
                continue

            # RELATIVE for gating
            az_rel, el_rel, dist = _angles_dist(pos, yaw_deg, pitch_deg, tgt.position)

            # FOV and range gating in RELATIVE domain
            if abs(az_rel) > self.h_fov_deg / 2 or abs(el_rel) > self.v_fov_deg / 2 or dist > self.max_range_m:
                continue

            # ABSOLUTE angles for outputs & geometry
            az_abs = yaw_deg + az_rel
            el_abs = pitch_deg + el_rel

            eff_rcs = _effective_rcs(tgt, pos)
            p = self.lut.get_probability(dist, eff_rcs)
            if torch.rand(1, device=self.device).item() > p:
                continue

            dop = _doppler(tgt, az_abs, el_abs, self.lut.freq_hz)

            # Store ABSOLUTE az/el (so downstream geometry is correct)
            snr_db = self.snr_threshold_db + self.np_rng.uniform(2.0, 8.0)
            dets.append({'T': tgt, 'az': az_abs, 'el': el_abs, 'd': dist, 'dop': dop, 'snr_db': snr_db})

        return dets

    def detect(self, pos: Position, targets: List[Unit],
            dt: float, steer_h: float, steer_p: float) -> List[dict]:
        if self.use_beam_steering:
            self._scan_time += dt
            period = 10.0  # Seconds for left-right-left cycle
            phase = 2 * math.pi * self._scan_time / period
            self.yaw_offset_deg = steer_h * math.sin(phase)
            self.pitch_offset_deg = steer_p * math.sin(phase)
        else:
            self.yaw_offset_deg = 0.0
            self.pitch_offset_deg = 0.0
        return self._raw_detections(pos, targets)