import math
from collections import deque

import numpy as np

from bvr_marl_core.domain.classification import signature_class_probabilities
from bvr_marl_core.domain.information import FrameReference, SensorReport, SensorType
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.core.lut import DetectionLUT
from bvr_marl_core.radar.core.parameter_policy import RadarParamPolicy
from bvr_marl_core.radar.core.utils import enu_to_geodetic, to_cart
from bvr_marl_core.radar.lock.base import BaseLockController
from bvr_marl_core.radar.obs.cluster import Clusterer
from bvr_marl_core.radar.obs.observation import RadarObsGenerator, resolution_cell_sigma
from bvr_marl_core.radar.scan_scheduler import ScanScheduler
from bvr_marl_core.radar.tracking.tracker import TrackerManager


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
        notch_velocity_mps=0.0,
        meas_angular_noise_deg=0.0,
        meas_range_noise_m=0.0,
        dl_delay_base_s=0.0,
        dl_delay_per_km_s=0.0,
        processing_gain_db=0.0,
        max_report_age_s=30.0,
        scan_sector_count=4,
        doppler_noise_hz=0.0,
        tracker_confirmation_hits=3,
        tracker_motion_model="cv",
    ):

        self.owner = owner
        self.processing_gain_db = float(processing_gain_db)
        self.max_report_age_s = float(max_report_age_s)
        # Per-scan probability that an own active detection is missed (dropped before
        # tracking). 0 = perfect scan; the imperfect-information benchmark raises it so
        # tracks coast between updates (paper missed-scan axis). Seeded per episode.
        self.missed_scan_probability = 0.0
        self._current_time_s = 0.0
        self.last_datalink_status = "live"
        # Datalink staleness: when this radar shares its track over the datalink,
        # consumers receive a snapshot delayed by base + per_km * (range to this
        # radar). 0/0 = instantaneous (fighters). A surveillance platform (AWACS)
        # sets these so a missile flown on its picture aims where the target was.
        self.dl_delay_base_s = float(dl_delay_base_s)
        self.dl_delay_per_km_s = float(dl_delay_per_km_s)
        # Time-stamped history of shared detections, populated only when a delay
        # is configured (avoids buffering overhead for instantaneous links).
        self._dl_history: deque = deque(maxlen=64)
        self._strobe_history: deque = deque(maxlen=32)
        self.use_beam_steering = use_beam_steering
        self.jam_susceptible = jam_susceptible
        owner_id = getattr(owner, "id", 0) if owner is not None else 0
        seed = hash(owner_id) % (2**32) if isinstance(owner_id, str) else owner_id
        self.np_rng = np.random.default_rng(0 if seed is None else seed)
        if data_link is None or isinstance(data_link, str):
            self.data_link = DataLink(data_link if isinstance(data_link, str) else "own")
        else:
            self.data_link = data_link

        self.cached_detections = None
        self.cached_clusters = None
        self.cached_tracks = None
        # Count of offboard tracks in the most recent picture; see the update path.
        self.network_tracks_received = 0
        self._next_report_id = 1
        # Evaluator compatibility bridge.  Truth identity is kept outside report,
        # cluster, filter, and track payloads and is never used for measurement or
        # association.  Legacy target-object launch APIs consult it only to resolve
        # an already selected operational contact.
        self._acquisition_target_by_detection: dict[int, object] = {}

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

        self.device = None  # No GPU dependency in core; accepted by LUT/obs for compatibility
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
            processing_gain_db=self.processing_gain_db,
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
            notch_velocity_mps=notch_velocity_mps,
            meas_angular_noise_deg=meas_angular_noise_deg,
            meas_range_noise_m=meas_range_noise_m,
            doppler_noise_hz=doppler_noise_hz,
        )
        self._dl_delay_enabled = self.dl_delay_base_s > 0.0 or self.dl_delay_per_km_s > 0.0
        self.clusterer = Clusterer(angular_resolution_deg, range_resolution_m)
        self.tracker_manager = TrackerManager(
            max(range_resolution_m, angular_resolution_deg),
            confirmation_hits=tracker_confirmation_hits,
            motion_model=tracker_motion_model,
        )

        self._scan_time = 0.0
        self.scan_scheduler = ScanScheduler(
            horizontal_fov_deg, vertical_fov_deg, sectors=scan_sector_count
        )
        self.current_dwell = None
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

    def get_delayed_detections(self, delay_s: float):
        """Return the shared detections as they were ``delay_s`` seconds ago.

        Returns the newest buffered snapshot at least ``delay_s`` old (the oldest
        retained snapshot if the history doesn't reach that far back). Falls back
        to the live detections when no delay is configured or no history exists.
        """
        if delay_s <= 0.0 or not self._dl_history:
            return self.cached_detections or []
        target_ts = self._dl_history[-1][0] - float(delay_s)
        if target_ts < self._dl_history[0][0]:
            self.last_datalink_status = "rejected_before_history"
            return []
        chosen = self._dl_history[0][1]
        for ts, dets in self._dl_history:
            if ts <= target_ts:
                chosen = dets
            else:
                break
        self.last_datalink_status = "delayed"
        return chosen or []

    def update(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0, group_radars=None
    ):
        self.stage_reports(tick_secs, sim, targets, owner_position, steer_h, steer_p)
        return self.update_from_staged_reports(
            tick_secs, sim, owner_position, group_radars=group_radars
        )

    def stage_reports(
        self, tick_secs, sim, targets, owner_position, steer_h=0.0, steer_p=0.0
    ) -> tuple[SensorReport, ...]:
        """Generate and freeze this radar's reports without consuming peer state."""
        self._current_time_s = float(getattr(sim, "elapsed_time_s", 0.0) or 0.0)
        self.param_policy.update_dynamic_params()
        self._update_beam_steering(tick_secs, steer_h, steer_p)

        # EMCON: a silent radar (owner.radar_emitting is False) generates no active
        # returns of its own. The datalink fusion below still runs, so the platform
        # keeps the shared picture from emitting friendlies. Missile seekers have no
        # radar_emitting attribute, so getattr defaults True and they always emit.
        self._acquisition_target_by_detection.clear()
        emitting = getattr(self.owner, "radar_emitting", True)
        if emitting:
            own_detections = self._generate_and_transform_detections(owner_position, targets)
            own_detections = self._apply_missed_scan(own_detections, sim)
            own_detections.extend(self._generate_false_alarms(owner_position, tick_secs))
        else:
            own_detections = []

        if emitting and self.jam_susceptible and sim is not None and hasattr(sim, "ew_world"):
            t = getattr(sim, "elapsed_time_s", 0.0)
            denial = sim.ew_world.collect_range_denial(self, t)
            if denial:
                self._apply_range_denial(own_detections, denial, owner_position)

        # Passive IRST contacts (bearing-only): added regardless of EMCON because the
        # IRST does not radiate. They fuse/triangulate through the strobe path below.
        irst = getattr(self.owner, "irst", None)
        if irst is not None:
            ir_detections = irst.generate(owner_position, targets, self.yaw_deg, self.pitch_deg)
            for detection in ir_detections:
                target = detection.pop("T", None)
                detection["_sensor_type"] = SensorType.IRST.value
                if target is not None:
                    self._acquisition_target_by_detection[id(detection)] = target
            own_detections.extend(ir_detections)

        own_reports = self._freeze_sensor_reports(own_detections, owner_position, sim)
        self._acquisition_target_by_detection.clear()
        self.cached_detections = own_reports
        self._strobe_history.extend(
            report for report in own_reports if bool(report.metadata.get("range_denied", False))
        )

        # Record this tick's shared detections for delayed datalink playback.
        if self._dl_delay_enabled:
            now_s = getattr(sim, "elapsed_time_s", None)
            if now_s is not None:
                try:
                    self._dl_history.append((float(now_s), own_reports))
                except (TypeError, ValueError):
                    pass

        return own_reports

    def get_recent_strobes(
        self, current_time_s: float, max_age_s: float = 3.0, *, exclude_irst: bool = False
    ):
        """Return bounded recent anonymous bearing reports for time-consistent fusion.

        ``exclude_irst`` drops passive IRST bearings from the replayed history. A few
        seconds of history lets asynchronously dwelling radars triangulate a single
        jammer across ticks, but replaying every platform's IRST bearings into the
        shared picture re-triangulates a temporally-mixed set of many-target bearings
        each tick, manufacturing ghost intersections. IRST is therefore fused only
        from the current tick (see ``network_picture``), while jammer strobes keep
        their cross-tick history.
        """
        cutoff = float(current_time_s) - max(0.0, float(max_age_s))
        return tuple(
            report
            for report in self._strobe_history
            if report.acquisition_time_s >= cutoff
            and not (exclude_irst and report.sensor_type == SensorType.IRST)
        )

    def update_from_staged_reports(self, tick_secs, sim, owner_position, *, group_radars=None):
        """Update the local (own-radar) tracker, then reconcile with the network picture.

        Cross-platform fusion is no longer redone here per receiver: it happens once per
        team in the shared Link-16 network picture (see ``core.network_picture``). This
        radar tracks only its own reports (fresh, dropout-proof) and merges the shared
        picture on top (``core.track_reconcile``).
        """
        own_reports = tuple(self.cached_detections or ())

        # Fuse the explicit live reports, bypassing datalink history entirely. A
        # source's net-entry latency belongs only to the shared network picture;
        # its own tracker always consumes the freshest onboard measurement.
        local_fuser = self.data_link if self.data_link is not None else DataLink("own")
        clusters = local_fuser.fuse_reports(
            own_reports,
            owner_position,
            self.clusterer,
            cooperative=False,
            current_time_s=self._current_time_s,
            max_report_age_s=self.max_report_age_s,
        )
        self.cached_clusters = clusters

        if self.owner is not None and hasattr(self.owner, "position"):
            self.tracker_manager.set_export_reference(self.owner.position)

        missed_increment = None
        if self.use_beam_steering:
            # Track deletion is defined in scan-update opportunities, not macro
            # simulation ticks. Fractional ticks must therefore contribute only
            # their fraction of a physical dwell to the missed-update counter.
            missed_increment = float(tick_secs) / self.scan_scheduler.dwell_duration_s
        local_tracks = self.tracker_manager.update_tracks(
            clusters,
            tick_secs,
            owner_position,
            missed_increment=missed_increment,
        )
        # Merge the shared team network picture; own-radar fallback when off the net.
        from bvr_marl_core.radar.core.track_reconcile import reconcile_local_and_network

        network_tracks = self._read_network_picture(sim, owner_position)
        # Whether offboard tracks are arriving is an OPERATIONAL fact the platform
        # can observe from its own receiver. Recording it here lets consumers answer
        # "do I have datalink support?" without scanning the truth registry for a
        # live teammate, which is what the behaviour-tree picture builder was doing.
        self.network_tracks_received = len(network_tracks or ())
        self.cached_tracks = reconcile_local_and_network(local_tracks, network_tracks)
        refresh_attribution = getattr(sim, "refresh_contact_truth_associations", None)
        sensor_id = getattr(getattr(self, "owner", None), "id", None)
        if sensor_id is not None and callable(refresh_attribution):
            # Attribute the IDs actually exposed by this receiver. In particular,
            # a locally refreshed estimate may now carry a shared Network Track
            # Number, which weapon launch must resolve at the evaluator boundary.
            refresh_attribution(
                sensor_id,
                {track.track_id: track.report_lineage for track in self.cached_tracks},
            )
        self._update_lock_ctrl(self.cached_tracks)
        return self.cached_tracks

    def _read_network_picture(self, sim, owner_position):
        """Return the shared team network tracks in this platform's frame, or [].

        Empty when there is no picture or the platform is cut off from the net
        (own-radar fallback).
        """
        pictures = getattr(sim, "network_pictures", None)
        if not pictures:
            return []
        owner = self.owner
        picture = pictures.get(getattr(owner, "group", None))
        if picture is None or not self._on_datalink_net(sim, owner):
            return []
        return picture.tracks_in_frame(owner_position)

    def _on_datalink_net(self, sim, owner) -> bool:
        """True if any teammate can transmit to this platform (it can read the net)."""
        if owner is None:
            return False
        owner_link = getattr(getattr(owner, "radar", None), "data_link", None)
        if owner_link is None or owner_link.get_mode() != "full":
            return False
        is_datalink_up = getattr(sim, "is_datalink_up", None)
        if not callable(is_datalink_up):
            return True
        owner_id = getattr(owner, "id", None)
        team = getattr(owner, "group", None)
        for unit in sim.active_units.values():
            if (
                unit is owner
                or getattr(unit, "group", None) != team
                or bool(getattr(unit, "is_missile", False))
            ):
                continue
            link = getattr(getattr(unit, "radar", None), "data_link", None)
            if link is None or link.get_mode() != "full":
                continue
            if is_datalink_up(getattr(unit, "id", None), owner_id):
                return True
        return False

    def _freeze_sensor_reports(self, detections, owner_position, sim) -> tuple[SensorReport, ...]:
        """Seal mutable acquisition scratch data before it crosses the radar boundary."""
        reports = []
        source_id = getattr(self.owner, "id", 0) or 0
        acquisition_time_s = float(getattr(sim, "elapsed_time_s", 0.0) or 0.0)
        angular_variance = (
            max(self.obsgen.meas_angular_noise_deg, resolution_cell_sigma(self.a_res_deg)) ** 2
        )
        range_variance = (
            max(self.obsgen.meas_range_noise_m, resolution_cell_sigma(self.r_res_m)) ** 2
        )
        covariance = np.diag([angular_variance, angular_variance, range_variance])
        frame = FrameReference(owner_position.lat, owner_position.lon, owner_position.alt)
        for detection in detections:
            report_id = self._next_report_id
            target = self._acquisition_target_by_detection.get(id(detection))
            truth_id = getattr(target, "id", None)
            register_attribution = getattr(sim, "register_sensor_report_truth_association", None)
            sensor_id = getattr(getattr(self, "owner", None), "id", None)
            if truth_id is not None and sensor_id is not None and callable(register_attribution):
                register_attribution(sensor_id, report_id, truth_id)
            metadata = {
                key: value
                for key, value in detection.items()
                if key not in {"T", "_sensor_type"} and isinstance(value, (str, int, float, bool))
            }
            metadata.pop("jammer_id", None)
            metadata.pop("engagement_id", None)
            if bool(metadata.get("range_denied", False)):
                metadata["strobe_id"] = f"{source_id}:strobe:{report_id}"
            sensor_type = SensorType(detection.get("_sensor_type", SensorType.RADAR.value))
            reports.append(
                SensorReport(
                    report_id=report_id,
                    source_id=source_id,
                    acquisition_time_s=acquisition_time_s,
                    measurement=(detection["az"], detection["el"], detection["d"]),
                    covariance=covariance,
                    frame=frame,
                    sensor_type=sensor_type,
                    classification_probabilities=signature_class_probabilities(target),
                    metadata=metadata,
                )
            )
            self._next_report_id += 1
        return tuple(reports)

    def _update_beam_steering(self, tick_secs, steer_h, steer_p):
        if self.use_beam_steering:
            self._scan_time += tick_secs
            self.current_dwell = self.scan_scheduler.next_dwell(tick_secs)
            self.yaw_offset_deg = self.current_dwell.center_azimuth_offset_deg
            self.pitch_offset_deg = self.current_dwell.center_elevation_offset_deg
        else:
            self.current_dwell = None
            self.yaw_offset_deg = 0.0
            self.pitch_offset_deg = 0.0

    def _apply_missed_scan(self, detections, sim):
        """Drop each own detection with probability ``missed_scan_probability``.

        Seeded from the simulator's per-episode named RNG streams so realized misses
        are reproducible from the episode seed. No-op at probability 0.
        """
        p = float(getattr(self, "missed_scan_probability", 0.0))
        if p <= 0.0 or not detections:
            return detections
        streams = getattr(sim, "random_streams", None)
        if streams is None:
            return detections
        rng = streams.generator("missed_scan", getattr(self.owner, "id", 0))
        return [det for det in detections if rng.random() >= p]

    def _generate_and_transform_detections(self, owner_position, targets):
        own_group = getattr(self.owner, "group", None)
        own_id = getattr(self.owner, "id", None)
        old_h, old_v = self.obsgen.h_fov_deg, self.obsgen.v_fov_deg
        if self.current_dwell is not None:
            self.obsgen.h_fov_deg = self.current_dwell.horizontal_width_deg
            self.obsgen.v_fov_deg = self.current_dwell.vertical_width_deg
        try:
            detections = self.obsgen.generate(
                owner_position,
                targets,
                self.yaw_deg,
                self.pitch_deg,
                own_group=own_group,
                own_id=own_id,
                own_velocity=getattr(self.owner, "velocity", None),
                dwell_time_s=(
                    self.current_dwell.duration_s if self.current_dwell is not None else 1.0
                ),
            )
        finally:
            self.obsgen.h_fov_deg, self.obsgen.v_fov_deg = old_h, old_v
        self._acquisition_target_by_detection = {
            id(detection): target
            for detection, target in zip(
                detections, self.obsgen.last_detection_targets, strict=True
            )
        }
        for det in detections:
            enu_xyz = to_cart(det["az"], det["el"], det["d"])
            lat, lon, alt = enu_to_geodetic(
                enu_xyz, owner_position.lat, owner_position.lon, owner_position.alt
            )
            det["lat"] = lat
            det["lon"] = lon
            det["alt"] = alt
        return detections

    def _generate_false_alarms(self, owner_position, tick_secs: float) -> list[dict]:
        """Generate uncorrelated search-cell clutter for the current dwell."""
        dwell_duration_s = (
            self.current_dwell.duration_s
            if self.current_dwell is not None
            else max(float(tick_secs), 0.0)
        )
        expected = max(0.0, float(self.false_alarm_rate)) * dwell_duration_s
        count = int(self.np_rng.poisson(expected))
        if count <= 0:
            return []
        dwell = self.current_dwell
        center_az = self.yaw_deg
        center_el = self.pitch_deg
        width_h = dwell.horizontal_width_deg if dwell is not None else self.h_fov_deg
        width_v = dwell.vertical_width_deg if dwell is not None else self.v_fov_deg
        detections = []
        for _ in range(count):
            azimuth = center_az + float(self.np_rng.uniform(-width_h / 2.0, width_h / 2.0))
            elevation = center_el + float(self.np_rng.uniform(-width_v / 2.0, width_v / 2.0))
            distance = self.max_range_m * float(self.np_rng.random()) ** (1.0 / 3.0)
            enu = to_cart(azimuth, elevation, distance)
            lat, lon, alt = enu_to_geodetic(
                enu, owner_position.lat, owner_position.lon, owner_position.alt
            )
            detections.append(
                {
                    "az": azimuth,
                    "el": elevation,
                    "d": distance,
                    "dop": float(self.np_rng.normal(0.0, self.obsgen.doppler_noise_hz or 50.0)),
                    "snr_db": self.snr_threshold_db,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "is_deception": True,
                    "is_false_alarm": True,
                }
            )
        return detections

    def _apply_range_denial(self, detections, denial, owner_position):
        """Turn detections of jamming targets beyond burn-through into range strobes.

        The bearing (az/el) is kept, but the range is denied: the reported range is
        replaced by a strobe placeholder at this radar's max range and ``lat/lon/alt``
        are recomputed there, while the *true* bearing and this observer's position
        are stashed (``strobe_az/strobe_el/obs_lat/obs_lon/obs_alt``) so cooperating
        radars can later triangulate the real range. Inside burn-through the radar
        keeps the true range (skin echo dominates), so the detection is untouched.
        """
        for d in detections:
            tgt = self._acquisition_target_by_detection.get(id(d))
            tid = getattr(tgt, "id", None)
            burn_through_m = denial.get(tid)
            if burn_through_m is None:
                continue
            if d.get("d", 0.0) <= burn_through_m:
                continue  # inside burn-through: range recovered, leave as-is

            d["range_denied"] = True
            # Preserve the true bearing and observer for downstream triangulation.
            d["strobe_az"] = d["az"]
            d["strobe_el"] = d["el"]
            d["obs_lat"] = owner_position.lat
            d["obs_lon"] = owner_position.lon
            d["obs_alt"] = owner_position.alt
            # Deny range: place the strobe at max range along the true bearing.
            strobe_range = float(self.max_range_m)
            d["d"] = strobe_range
            enu_xyz = to_cart(d["az"], d["el"], strobe_range)
            lat, lon, alt = enu_to_geodetic(
                enu_xyz, owner_position.lat, owner_position.lon, owner_position.alt
            )
            d["lat"], d["lon"], d["alt"] = lat, lon, alt

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
        for track in tracks:
            if track.engageable:
                detected_target_ids.add(track.track_id)
        self.lock_ctrl.update_locks(detected_target_ids)

    def get_tracks(self):
        return self.cached_tracks if hasattr(self, "cached_tracks") else []
