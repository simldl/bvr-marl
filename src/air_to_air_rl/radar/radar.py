import math
import os

import numpy as np
import torch

from air_to_air_rl.radar.core.data_link import DataLink
from air_to_air_rl.radar.core.lut import DetectionLUT
from air_to_air_rl.radar.core.parameter_policy import RadarParamPolicy
from air_to_air_rl.radar.core.utils import enu_to_geodetic, to_cart
from air_to_air_rl.radar.lock.base import BaseLockController
from air_to_air_rl.radar.obs.cluster import Clusterer
from air_to_air_rl.radar.obs.observation import RadarObsGenerator
from air_to_air_rl.radar.tracking.tracker import TrackerManager

# Device selection resolved once at import time — avoids repeated torch.cuda.is_available()
# syscall on every Radar/LUT construction during resets and spawns.
_RADAR_FORCE_CPU: bool = os.environ.get("RADAR_FORCE_CPU", "0") == "1"
_CUDA_AVAILABLE: bool = (not _RADAR_FORCE_CPU) and torch.cuda.is_available()
_DEFAULT_RADAR_DEVICE: torch.device = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")


def estimate_noise_power(radar) -> float:
    """
    Estimate radar thermal noise power (Watts) for jamming calculations.

    Simple model: N = k*T*B*F where
      k = Boltzmann constant (1.38e-23 J/K)
      T = system temperature (~290 K)
      B = bandwidth (Hz)
      F = noise figure (~3 dB = 2x linear)

    Args:
        radar: Radar instance

    Returns:
        Noise power in Watts
    """
    k_boltzmann = 1.38e-23  # J/K
    T_sys = 290.0  # System temperature (K)
    B_hz = getattr(radar, "bandwidth_hz", 1e6)  # Default 1 MHz if not specified
    F_linear = 2.0  # Noise figure ~3 dB

    N = k_boltzmann * T_sys * B_hz * F_linear
    return N


def lin2db(ratio: float) -> float:
    """Convert linear ratio to decibels."""
    return 10.0 * math.log10(max(ratio, 1e-12))


class Radar:
    def __init__(
        self,
        horizontal_fov_deg,
        vertical_fov_deg,
        max_range_m,
        radar_frequency_hz,
        tx_power_w,
        antenna_gain_db,
        snr_threshold_db,
        false_alarm_rate=0.01,
        range_resolution_m=5000.0,
        angular_resolution_deg=10.0,
        lut_bins=(200, 200),
        owner=None,
        use_beam_steering=False,
        lock_ctrl=None,
        data_link=None,
        jam_susceptible=True,
    ):

        self.owner = owner
        self.use_beam_steering = use_beam_steering
        self.jam_susceptible = jam_susceptible
        # Device resolved at module import time to avoid per-instance syscall
        self.device = torch.device("cpu") if _RADAR_FORCE_CPU else _DEFAULT_RADAR_DEVICE
        owner_id = getattr(owner, "id", 0) if owner is not None else 0
        seed = hash(owner_id) % (2**32) if isinstance(owner_id, str) else owner_id
        self.np_rng = np.random.default_rng(seed)
        if data_link is None or isinstance(data_link, str):
            self.data_link = DataLink(data_link if isinstance(data_link, str) else "own")
        else:
            self.data_link = data_link

        self.cached_detections = None
        self.cached_clusters = None
        self.cached_tracks = None

        self.h_fov_deg = horizontal_fov_deg
        self.v_fov_deg = vertical_fov_deg
        self.max_range_m = max_range_m
        self.freq_hz = radar_frequency_hz
        self.tx_power_w = tx_power_w
        self.antenna_gain_db = antenna_gain_db
        self.snr_threshold_db = snr_threshold_db
        self.false_alarm_rate = false_alarm_rate
        self.r_res_m = range_resolution_m
        self.a_res_deg = angular_resolution_deg

        self.lock_ctrl = lock_ctrl if lock_ctrl is not None else BaseLockController()
        self.param_policy = RadarParamPolicy(
            search_h_fov_deg=horizontal_fov_deg,
            search_v_fov_deg=vertical_fov_deg,
            search_gain_db=antenna_gain_db,
            search_snr_db=snr_threshold_db,
            search_false_alarm_rate=false_alarm_rate,
        )
        self.param_policy.lock_ctrl = self.lock_ctrl

        # Reuse cached LUT for identical configurations — LUT is read-only after construction
        self.lut = DetectionLUT.get_or_create(
            radar_frequency_hz,
            tx_power_w,
            10 ** (antenna_gain_db / 10),
            max_range_m,
            snr_threshold_db,
            max_rcs=20.0,
            rcs_bins=lut_bins[1],
            dist_bins=lut_bins[0],
            device=self.device,
        )
        self.obsgen = RadarObsGenerator(
            horizontal_fov_deg,
            vertical_fov_deg,
            max_range_m,
            self.lut,
            snr_threshold_db,
            false_alarm_rate,
            np_rng=self.np_rng,
            device=self.device,
        )
        self.clusterer = Clusterer(angular_resolution_deg, range_resolution_m, device=self.device)
        self.tracker_manager = TrackerManager(max(range_resolution_m, angular_resolution_deg))

        self._scan_time = 0.0
        self.yaw_offset_deg = 0.0
        self.pitch_offset_deg = 0.0

    @property
    def yaw_deg(self):
        if self.owner is not None:
            return float(self.owner.yaw_deg) + self.yaw_offset_deg
        return self.yaw_offset_deg

    @property
    def pitch_deg(self):
        if self.owner is not None:
            return float(self.owner.pitch_deg) + self.pitch_offset_deg
        return self.pitch_offset_deg

    def get_locked_target(self):
        tgt = getattr(self.owner, "target", None)
        return getattr(tgt, "id", None)

    def get_locked_targets(self):
        tid = self.get_locked_target()
        return {tid} if tid is not None else set()

    def get_default_group_radars(self, sim):
        return DataLink.update_group_radars(sim, owner=self.owner)

    def update(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None
    ):
        self.param_policy.update_dynamic_params()
        self._update_beam_steering(tick_secs, steer_h, steer_p)
        own_detections = self._generate_and_transform_detections(owner_position, targets)

        if self.jam_susceptible and sim is not None and hasattr(sim, "ew_world"):
            t = getattr(sim, "elapsed_time_s", 0.0)
            jammer_info, ghosts = sim.ew_world.collect_incoming(self, t)

            N0 = estimate_noise_power(self)

            for d in own_detections:
                det_az = d.get("az", 0.0)
                det_el = d.get("el", 0.0)

                # Sum per-detection jamming using cosine taper over angular separation.
                # Cosine drops to zero at 90° (sector burn-through effect).
                J_total = 0.0
                for jam_az, jam_el, J_power in jammer_info:
                    angle_sep_rad = math.radians(
                        math.sqrt((det_az - jam_az) ** 2 + (det_el - jam_el) ** 2)
                    )
                    J_total += J_power * max(0.0, math.cos(angle_sep_rad))

                # SNR_eff = S/(N+J)
                jamming_degradation_db = lin2db(1.0 + J_total / (N0 + 1e-12))
                snr_eff_db = d.get("snr_db", 0.0) - jamming_degradation_db

                if snr_eff_db < self.snr_threshold_db:
                    d["drop"] = True
                else:
                    d["snr_db"] = snr_eff_db

            own_detections = [d for d in own_detections if not d.get("drop", False)]
            own_detections.extend(ghosts)

            for d in own_detections:
                if ("lat" not in d) or ("lon" not in d) or ("alt" not in d):
                    enu_xyz = to_cart(d["az"], d["el"], d["d"])
                    lat, lon, alt = enu_to_geodetic(
                        enu_xyz, owner_position.lat, owner_position.lon, owner_position.alt
                    )
                    d["lat"], d["lon"], d["alt"] = lat, lon, alt

        self.cached_detections = own_detections

        if group_radars is None and sim is not None:
            group_radars = self.get_default_group_radars(sim)

        clusters = self._fusion_and_clustering(owner_position, group_radars, own_detections)
        self.cached_clusters = clusters

        if self.owner is not None and hasattr(self.owner, "position"):
            self.tracker_manager.set_export_reference(self.owner.position)

        self.cached_tracks = self.tracker_manager.update_tracks(clusters, tick_secs, owner_position)
        self._update_lock_ctrl(self.cached_tracks)
        return self.cached_tracks

    def _update_beam_steering(self, tick_secs, steer_h, steer_p):
        if self.use_beam_steering:
            self._scan_time += tick_secs
            period = 10.0
            phase = 2 * math.pi * self._scan_time / period
            self.yaw_offset_deg = steer_h * math.sin(phase)
            self.pitch_offset_deg = steer_p * math.sin(phase)
        else:
            self.yaw_offset_deg = 0.0
            self.pitch_offset_deg = 0.0

    def _generate_and_transform_detections(self, owner_position, targets):
        own_group = getattr(self.owner, "group", None)
        own_id = getattr(self.owner, "id", None)
        detections = self.obsgen.generate(
            owner_position,
            targets,
            self.yaw_deg,
            self.pitch_deg,
            own_group=own_group,
            own_id=own_id,
        )
        for det in detections:
            enu_xyz = to_cart(det["az"], det["el"], det["d"])
            lat, lon, alt = enu_to_geodetic(
                enu_xyz, owner_position.lat, owner_position.lon, owner_position.alt
            )
            det["lat"] = lat
            det["lon"] = lon
            if det.get("T") is not None and hasattr(det["T"], "position"):
                det["alt"] = float(det["T"].position.alt)
            else:
                det["alt"] = alt
        return detections

    def _fusion_and_clustering(self, owner_position, group_radars, own_detections):
        if self.data_link is not None:
            fused_measurements = self.data_link.get_fused_clusters(
                own_radar=self,
                group_radars=group_radars,
                own_position=owner_position,
                clusterer=self.clusterer,
            )
        else:
            fused_measurements = self.clusterer.cluster(own_detections)
        return fused_measurements

    def _update_lock_ctrl(self, tracks):
        detected_target_ids = set()
        for (
            tid,
            state,
            cov,
            tgt,
            utype,
            ref,
            confidence,
            n_obs,
            lifetime,
            update_count,
            is_deception,
            suspect_deception,
            engagement_id,
            jammer_id,
            engageable,
        ) in tracks:
            if engageable and tgt is not None and hasattr(tgt, "id"):
                detected_target_ids.add(tgt.id)
        self.lock_ctrl.update_locks(detected_target_ids)

    def get_tracks(self):
        return self.cached_tracks if hasattr(self, "cached_tracks") else []
