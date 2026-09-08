import math

import numpy as np

from bvr_marl_core.domain.information import SensorReport
from bvr_marl_core.radar.core.utils import _angles_dist, enu_to_geodetic, geodetic_to_enu, to_cart
from bvr_marl_core.radar.ew.triangulation import pairwise_bearing_candidates, triangulate
from bvr_marl_core.radar.obs.common_frame import cluster_reports_common_frame
from bvr_marl_core.radar.tracking.helpers.enu_utils import enu_rotation
from bvr_marl_core.simulator.core.helpers import Position

# Where a bearing-only (untriangulated) jammer strobe is placed along its bearing
# when the true range is unknown; the track is given a large covariance so this
# guess barely constrains the estimate.
_BEARING_ONLY_NOMINAL_RANGE_M = 60_000.0


def _codes(values, missing_is_wildcard=False):
    """Map hashable labels to small integer codes for outer-product mask building.

    With ``missing_is_wildcard`` a ``None`` label is coded as ``-1`` so callers can
    treat it as compatible with everything (an unknown RF band matches any band).
    """
    codes = np.empty(len(values), dtype=np.int64)
    lookup: dict = {}
    for index, value in enumerate(values):
        if missing_is_wildcard and value is None:
            codes[index] = -1
            continue
        code = lookup.get(value)
        if code is None:
            code = len(lookup)
            lookup[value] = code
        codes[index] = code
    return codes


class DataLink:
    VALID_MODES = ("full", "own", "none", "other", "msl_support")

    def __init__(self, mode="own"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {self.VALID_MODES}")
        self.mode = mode

    def set_mode(self, mode):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {self.VALID_MODES}")
        self.mode = mode

    def get_mode(self):
        return self.mode

    def __repr__(self):
        return f"<DataLink mode={self.mode}>"

    def get_fused_clusters(
        self, own_radar, group_radars, own_position, clusterer, cooperative=True
    ):
        raw_detections = self._gather_raw_detections(
            self.get_mode(), own_radar, group_radars, own_position
        )
        return self.fuse_reports(
            raw_detections,
            own_position,
            clusterer,
            cooperative=cooperative,
            current_time_s=getattr(own_radar, "_current_time_s", None),
            max_report_age_s=float(getattr(own_radar, "max_report_age_s", 30.0)),
        )

    def fuse_reports(
        self,
        reports,
        own_position,
        clusterer,
        *,
        cooperative=True,
        current_time_s=None,
        max_report_age_s=30.0,
    ):
        """Fuse an explicit frozen report set without gathering peer state."""
        # Noise-jammer strobes carry only a bearing (range denied); pull them out
        # before clustering — they are resolved separately by cross-radar
        # triangulation and must not be binned by their placeholder range.
        strobes = [d for d in reports if d.get("range_denied")]
        normal = [d for d in reports if not d.get("range_denied")]

        if normal and all(isinstance(report, SensorReport) for report in normal):
            # Production reports are transformed out of their source-local
            # spherical frames before any grouping decision is made.
            fused = cluster_reports_common_frame(
                normal,
                own_position,
                current_time_s=current_time_s,
                max_report_age_s=float(max_report_age_s),
            )
        else:
            # Transitional compatibility for legacy tests and integrations that
            # still provide mutable detection dictionaries.
            clusters = clusterer.cluster(normal)
            fused = self._fuse_clusters_by_id_and_false_alarm(clusters)

        if strobes:
            fused.extend(self._resolve_jammer_strobes(strobes, own_position, cooperative))

        return fused

    def net_entry_reports(self, src_radar, src_position, *, exclude_irst_history=False):
        """Return a source's contribution after its one-way net-entry latency.

        The distance-dependent receiver term is intentionally zero: Link-16-style
        entry latency is paid once by the transmitting source, after which every
        connected team member reads the same shared picture.

        A source replays a few seconds of its bearing-only strobe history so
        asynchronously dwelling radars can triangulate a jammer across ticks.
        ``exclude_irst_history`` keeps that jammer history but drops replayed IRST
        bearings: pooling several ticks of passive angle-only reports from every
        platform re-triangulates a temporally-mixed bearing set each tick, which for
        multiple IRST targets spawns combinatorial ghost intersections. The shared
        network picture sets it so IRST is fused only from the current tick (see
        ``network_picture``).
        """
        return self._shared_source_detections(
            src_radar,
            src_position,
            src_position,
            exclude_irst_history=exclude_irst_history,
        )

    def _resolve_jammer_strobes(self, strobes, own_position, cooperative=True):
        # Non-cooperative consumers (a classic Fox-3 seeker in the terminal phase)
        # cannot fuse other platforms' bearings, so only their own strobe is used —
        # yielding a bearing-only home-on-jam track. Cooperative consumers (fighters,
        # a full-datalink missile) keep every observer's bearing for triangulation.
        strobes = self._collapse_temporal_strobe_revisits(strobes)
        if not cooperative:
            strobes = [
                d
                for d in strobes
                if abs(float(d.get("obs_lat", 0.0)) - own_position.lat) < 1e-4
                and abs(float(d.get("obs_lon", 0.0)) - own_position.lon) < 1e-4
            ]
            if not strobes:
                return []

        out = []
        for group in self._associate_strobes_by_geometry(strobes, own_position, cooperative):
            # One bearing per distinct observer (dedupe co-located shares).
            per_obs: dict = {}
            for d in group:
                key = (
                    round(float(d.get("obs_lat", 0.0)), 4),
                    round(float(d.get("obs_lon", 0.0)), 4),
                    round(float(d.get("obs_alt", 0.0)), 0),
                )
                per_obs.setdefault(key, d)
            obs_dets = list(per_obs.values())
            hypothesis_id = "strobe-hypothesis:" + "|".join(
                sorted(str(d.get("strobe_id", "anonymous")) for d in obs_dets)
            )

            observers_enu = []
            directions_enu = []
            for d in obs_dets:
                obs = (float(d["obs_lat"]), float(d["obs_lon"]), float(d["obs_alt"]))
                p = np.array(
                    geodetic_to_enu(
                        obs[0], obs[1], obs[2], own_position.lat, own_position.lon, own_position.alt
                    ),
                    dtype=float,
                )
                u_local = np.array(
                    to_cart(float(d["strobe_az"]), float(d["strobe_el"]), 1.0), dtype=float
                )
                u_own = enu_rotation(obs[0], obs[1], own_position.lat, own_position.lon) @ u_local
                observers_enu.append(p)
                directions_enu.append(u_own)

            report_ids = tuple(
                report_id
                for detection in obs_dets
                for report_id in detection.get("report_ids", (detection.get("report_id"),))
                if report_id is not None
            )
            report_lineage = tuple(
                sorted(
                    {
                        lineage
                        for detection in obs_dets
                        for lineage in detection.get(
                            "report_lineage",
                            ((detection.get("source_id"), detection.get("report_id")),),
                        )
                        if lineage[0] is not None and lineage[1] is not None
                    },
                    key=lambda item: (str(item[0]), item[1]),
                )
            )
            source_ids = tuple(
                sorted(
                    {
                        source_id
                        for detection in obs_dets
                        for source_id in detection.get("source_ids", (detection.get("source_id"),))
                        if source_id is not None
                    },
                    key=str,
                )
            )
            point_enu, ok = triangulate(observers_enu, directions_enu)
            if ok:
                out.append(
                    self._jammer_cluster(
                        point_enu,
                        own_position,
                        hypothesis_id,
                        len(obs_dets),
                        triangulated=True,
                        report_ids=report_ids,
                        source_ids=source_ids,
                        report_lineage=report_lineage,
                    )
                )
            else:
                # Bearing-only: nominal range along the (own or first) bearing.
                p0, u0 = observers_enu[0], directions_enu[0]
                point = p0 + _BEARING_ONLY_NOMINAL_RANGE_M * (u0 / (np.linalg.norm(u0) + 1e-12))
                out.append(
                    self._jammer_cluster(
                        point,
                        own_position,
                        hypothesis_id,
                        len(obs_dets),
                        triangulated=False,
                        report_ids=report_ids,
                        source_ids=source_ids,
                        report_lineage=report_lineage,
                    )
                )
        return out

    @staticmethod
    def _collapse_temporal_strobe_revisits(strobes):
        """Keep the newest report in each source-local resolvable bearing cell.

        Recent history is retained to bridge asynchronous dwells, but revisits
        from one receiver are correlated and must not multiply the geometric
        association workload or classification evidence.
        """
        newest = {}
        for strobe in strobes:
            source_id = strobe.get("source_id")
            key = (
                source_id,
                strobe.get("frequency_band"),
                round(float(strobe.get("strobe_az", strobe.get("az", 0.0)))),
                round(float(strobe.get("strobe_el", strobe.get("el", 0.0)))),
            )
            acquisition_time = float(strobe.get("acquisition_time_s", 0.0) or 0.0)
            previous = newest.get(key)
            if previous is None or acquisition_time > previous[0]:
                newest[key] = (acquisition_time, strobe)
        return [item[1] for item in newest.values()]

    @staticmethod
    def _associate_strobes_by_geometry(strobes, own_position, cooperative=True):
        """Associate anonymous bearings using geometry and observable compatibility."""
        if not cooperative or len(strobes) < 2:
            return [[strobe] for strobe in strobes]

        observers = []
        directions = []
        observer_keys = []
        for detection in strobes:
            obs = (
                float(detection.get("obs_lat", own_position.lat)),
                float(detection.get("obs_lon", own_position.lon)),
                float(detection.get("obs_alt", own_position.alt)),
            )
            observers.append(
                np.asarray(
                    geodetic_to_enu(*obs, own_position.lat, own_position.lon, own_position.alt),
                    dtype=float,
                )
            )
            local = np.asarray(
                to_cart(
                    float(detection["strobe_az"]),
                    float(detection.get("strobe_el", 0.0)),
                    1.0,
                ),
                dtype=float,
            )
            direction = enu_rotation(obs[0], obs[1], own_position.lat, own_position.lon) @ local
            directions.append(direction / max(float(np.linalg.norm(direction)), 1e-12))
            observer_keys.append(tuple(round(value, 6) for value in obs))

        parent = list(range(len(strobes)))
        group_observers = [{observer_keys[index]} for index in range(len(strobes))]

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        # Geometric compatibility of every strobe pair in one vectorized pass instead
        # of an O(S^2) Python loop over triangulate_pair_normalized (the dominant EW
        # cost at scale). The result — which bearings intersect well enough to share
        # an emitter — is identical to the per-pair computation it replaces.
        residual_matrix, accept = pairwise_bearing_candidates(
            np.asarray(observers, dtype=float), np.asarray(directions, dtype=float)
        )
        # A pair may only merge across distinct observers and compatible RF bands;
        # apply those observable constraints as masks over the accepted grid.
        observer_codes = _codes([observer_keys[index] for index in range(len(strobes))])
        same_observer = observer_codes[:, None] == observer_codes[None, :]
        band_codes = _codes(
            [strobe.get("frequency_band") for strobe in strobes], missing_is_wildcard=True
        )
        incompatible_band = (
            (band_codes[:, None] != band_codes[None, :])
            & (band_codes[:, None] >= 0)
            & (band_codes[None, :] >= 0)
        )
        accept &= ~same_observer & ~incompatible_band

        left_indices, right_indices = np.nonzero(accept)
        candidates = sorted(
            (float(residual_matrix[left, right]), int(left), int(right))
            for left, right in zip(left_indices.tolist(), right_indices.tolist())
        )

        for _score, left, right in candidates:
            left_root, right_root = find(left), find(right)
            if left_root == right_root or group_observers[left_root] & group_observers[right_root]:
                continue
            parent[right_root] = left_root
            group_observers[left_root] |= group_observers[right_root]

        groups = {}
        for index, strobe in enumerate(strobes):
            groups.setdefault(find(index), []).append(strobe)
        return list(groups.values())

    def _jammer_cluster(
        self,
        point_enu,
        own_position,
        hypothesis_id,
        n_obs,
        triangulated,
        report_ids=(),
        source_ids=(),
        report_lineage=(),
    ):
        lat, lon, alt = enu_to_geodetic(
            point_enu, own_position.lat, own_position.lon, own_position.alt
        )
        e, n, u = float(point_enu[0]), float(point_enu[1]), float(point_enu[2])
        d = math.sqrt(e * e + n * n + u * u)
        az = math.degrees(math.atan2(e, n))
        el = math.degrees(math.atan2(u, math.hypot(e, n)))
        return {
            "T": None,
            "az": az,
            "el": el,
            "d": d,
            "dop": 0.0,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            # measurement_position overrides the ground-truth T position in the
            # tracker so range denial/triangulation actually drives the estimate.
            "measurement_position": Position(lat=lat, lon=lon, alt=alt),
            "n_obs": n_obs,
            "is_deception": False,
            "engagement_id": hypothesis_id,
            "jammer_id": hypothesis_id,
            "range_denied": not triangulated,
            "triangulated": triangulated,
            "report_ids": tuple(report_ids),
            "source_ids": tuple(source_ids),
            "report_lineage": tuple(report_lineage),
        }

    def _shared_source_detections(
        self, src_radar, src_pos, own_position, *, exclude_irst_history=False
    ):
        """Detections a group source contributes to the datalink, applying its
        configured staleness (delay grows with the range to that source). Sources
        with no delay (fighters) contribute their live detections."""
        base = float(getattr(src_radar, "dl_delay_base_s", 0.0))
        per_km = float(getattr(src_radar, "dl_delay_per_km_s", 0.0))
        if (base > 0.0 or per_km > 0.0) and hasattr(src_radar, "get_delayed_detections"):
            try:
                dist_m = _angles_dist(own_position, 0.0, 0.0, src_pos)[2]
            except Exception:
                dist_m = 0.0
            delay_s = base + per_km * (dist_m / 1000.0)
            return list(src_radar.get_delayed_detections(delay_s) or ())
        current = list(src_radar.cached_detections or ())
        recent_strobes = getattr(src_radar, "get_recent_strobes", None)
        if callable(recent_strobes):
            now_s = float(getattr(src_radar, "_current_time_s", 0.0) or 0.0)
            known = {(d.source_id, d.report_id) for d in current if isinstance(d, SensorReport)}
            current.extend(
                report
                for report in recent_strobes(now_s, exclude_irst=exclude_irst_history)
                if (report.source_id, report.report_id) not in known
            )
        return current

    def _gather_raw_detections(self, mode, own_radar, group_radars, own_position):
        """Detections visible to this datalink mode, all in the own-position frame.

        ``own`` sees only this radar; ``full`` adds peers also sharing in full mode;
        ``other`` takes peer contributions alone; ``msl_support`` is the in-flight
        missile case, which narrows peer reports to the missile's own targets.
        """
        if mode == "own":
            return self._own_source_detections(own_radar, own_position)
        if mode == "full":
            return self._own_source_detections(own_radar, own_position) + self._peer_detections(
                own_radar, group_radars, own_position, ("full",)
            )
        if mode == "other":
            return self._peer_detections(own_radar, group_radars, own_position, ("full", "own"))
        if mode == "msl_support":
            return self._missile_support_detections(own_radar, group_radars, own_position)
        return []

    def _own_source_detections(self, own_radar, own_position):
        """This radar's own detections; already in the local frame."""
        if not own_radar:
            return []
        return list(self._shared_source_detections(own_radar, own_position, own_position))

    def _peer_detections(self, own_radar, group_radars, own_position, sharing_modes):
        """Detections contributed by group members whose link shares in ``sharing_modes``."""
        detections = []
        for src_radar, src_pos in group_radars or []:
            if src_radar is own_radar or not src_radar.data_link:
                continue
            if src_radar.data_link.get_mode() not in sharing_modes:
                continue
            for detection in self._shared_source_detections(src_radar, src_pos, own_position):
                detections.append(
                    self._transform_detection_to_local_ref(detection, src_pos, own_position)
                )
        return detections

    def _missile_support_detections(self, own_radar, group_radars, own_position):
        """Peer reports about the targets this missile is actually guiding on.

        A supporting radar only contributes when it holds one of those targets *and*
        the missile itself lies inside that radar's field of regard, which is what
        makes the support geometrically credible.
        """
        detections = []
        if own_radar and own_radar.cached_detections:
            detections += own_radar.cached_detections

        own_target_ids = self._own_guidance_target_ids(own_radar)
        if not own_target_ids:
            return detections

        for src_radar, src_pos in group_radars or []:
            if not self._peer_supports_missile(
                src_radar, src_pos, own_radar, own_position, own_target_ids
            ):
                continue
            for detection in self._shared_source_detections(src_radar, src_pos, own_position):
                tid = getattr(detection.get("T", None), "id", None)
                if tid is not None and tid in own_target_ids:
                    detections.append(
                        self._transform_detection_to_local_ref(detection, src_pos, own_position)
                    )
        return detections

    def _peer_supports_missile(
        self, src_radar, src_pos, own_radar, own_position, own_target_ids
    ) -> bool:
        """Whether this peer can credibly support the missile's engagement."""
        if src_radar is own_radar or not getattr(src_radar, "owner", None):
            return False

        src_mode = getattr(getattr(src_radar, "data_link", None), "get_mode", lambda: "none")()
        if src_mode not in ("full", "own"):
            return False

        try:
            src_locked = set(src_radar.get_locked_targets() or [])
        except Exception:  # noqa: BLE001 - fusion loop: one malformed peer radar must
            # not abort the whole sweep; treat it as holding no locks.
            src_locked = set()
        if src_locked.isdisjoint(own_target_ids):
            return False

        return self._missile_within_field_of_regard(src_radar, src_pos, own_position)

    @staticmethod
    def _own_guidance_target_ids(own_radar) -> set:
        """Ids this weapon is guiding on: seeker locks plus any guidance-target object."""
        target_ids = set()
        try:
            target_ids |= set(own_radar.get_locked_targets() or [])
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass

        try:
            provider = getattr(own_radar, "target_provider", None)
            if provider is not None and hasattr(provider, "get_guidance_target"):
                target_obj = getattr(provider, "get_guidance_target_object", lambda: None)()
                if target_obj is not None and hasattr(target_obj, "id"):
                    target_ids.add(target_obj.id)
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
        return target_ids

    @staticmethod
    def _missile_within_field_of_regard(src_radar, src_pos, own_position) -> bool:
        """True when the missile sits inside the source radar's FOV and range."""
        try:
            az_off, el_off, dist = _angles_dist(
                src_pos,
                float(src_radar.yaw_deg),
                float(src_radar.pitch_deg),
                own_position,
            )
        except Exception:  # noqa: BLE001 - a peer with unreadable attitude is still
            # usable; fall back to a boresight-forward assumption rather than skipping it.
            az_off, el_off, dist = _angles_dist(src_pos, 0.0, 0.0, own_position)

        h_half = float(getattr(src_radar, "h_fov_deg", 120.0)) * 0.5
        v_half = float(getattr(src_radar, "v_fov_deg", 60.0)) * 0.5
        max_range_m = float(getattr(src_radar, "max_range_m", 1e9))
        return abs(az_off) <= h_half and abs(el_off) <= v_half and dist <= max_range_m

    def _transform_detection_to_local_ref(self, detection, src_ref_pos, tgt_ref_pos):
        """
        Transform detection from source reference frame to target reference frame.

        IMPORTANT: Geodetic coordinates (lat, lon, alt) are ABSOLUTE positions on Earth.
        They do not need to be transformed between reference frames - they are already
        in the correct global coordinate system.

        The detection is returned as-is. Each radar will convert the absolute geodetic
        position to its own local ENU frame when processing the detection.

        Args:
            detection: Detection dict with absolute geodetic coordinates
            src_ref_pos: Source radar position (not used, kept for API compatibility)
            tgt_ref_pos: Target radar position (not used, kept for API compatibility)

        Returns:
            Detection dict with absolute geodetic coordinates (unchanged)
        """
        # Geodetic coordinates are absolute - no transformation needed
        # Previous implementation had a bug that introduced 5km+ position errors
        # by incorrectly transforming through ENU frames
        return detection

    @staticmethod
    def update_group_radars(sim, owner=None):
        members = [
            (u.radar, u.position)
            for u in sim.active_units.values()
            if hasattr(u, "radar")
            and hasattr(u.radar, "data_link")
            and u.radar.data_link.get_mode() == "full"
            and (owner is None or u is not owner)
            and getattr(u, "group", None) == getattr(owner, "group", None)
        ]
        if owner is None or not hasattr(sim, "is_datalink_up"):
            return members
        receiver_id = getattr(owner, "id", None)
        return [
            member
            for member in members
            if sim.is_datalink_up(
                getattr(getattr(member[0], "owner", None), "id", None), receiver_id
            )
        ]

    @staticmethod
    def group_locked_target_ids(sim, owner):
        """Target ids held with a weapons-quality lock by a full-datalink friend.

        Enables AWACS/datalink-cued launches beyond the shooter's own radar range
        (the missile flies on datalink midcourse until its seeker goes active):

        - an AWACS contributes any enemy inside its locking cone
          (``can_lock_target`` — 360° detection, narrow ``lock_fov_deg`` lock), and
        - a wingman contributes its own radar locks.

        Only ``full`` datalink group members contribute.  The shooter's *own*
        radar lock is handled separately in ``fire_missile``.
        """
        if owner is None or sim is None:
            return set()
        own_group = getattr(owner, "group", None)
        units = list(sim.active_units.values())
        enemies = [
            u
            for u in units
            if getattr(u, "group", None) != own_group
            and not getattr(u, "is_missile", False)
            and not getattr(u, "is_non_engageable", False)
        ]
        locked: set = set()
        for u in units:
            if u is owner or getattr(u, "group", None) != own_group:
                continue
            radar = getattr(u, "radar", None)
            dl = getattr(radar, "data_link", None)
            if dl is None or dl.get_mode() != "full":
                continue
            if hasattr(sim, "is_datalink_up") and not sim.is_datalink_up(u.id, owner.id):
                continue
            if hasattr(u, "can_lock_target"):
                # AWACS (or similar): weapons-quality lock within its cone.
                for e in enemies:
                    try:
                        if u.can_lock_target(e):
                            locked.add(e.id)
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                        KeyError,
                        IndexError,
                        ZeroDivisionError,
                    ):
                        pass
            elif radar is not None:
                # Wingman: share its own radar locks.
                try:
                    locked |= set(radar.get_locked_targets() or [])
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    ZeroDivisionError,
                ):
                    pass
        return locked

    def _fuse_clusters_by_id_and_false_alarm(self, clusters):
        fused_measurements = []
        for c in clusters:
            # Handle T being list, None, or single object
            T_field = c.get("T", None)
            T_list = (
                T_field if isinstance(T_field, list) else ([T_field] if T_field is not None else [])
            )

            ids = [getattr(t, "id", None) for t in T_list]
            n_obs = len(T_list)
            id_set = set([i for i in ids if i is not None])

            if id_set:
                for tid in id_set:
                    group = [t for t in T_list if getattr(t, "id", None) == tid]
                    fused_measurements.append(
                        {
                            "T": group[0] if group else None,
                            "az": c["az"],
                            "el": c["el"],
                            "d": c["d"],
                            "dop": c["dop"],
                            "lat": c.get("lat"),
                            "lon": c.get("lon"),
                            "alt": c.get("alt"),
                            "n_obs": len(group),
                            "is_deception": c.get("is_deception", False),
                            "engagement_id": c.get("engagement_id", tid if group else None),
                            "jammer_id": c.get("jammer_id", None),
                            "report_ids": tuple(c.get("report_ids", ())),
                            "source_ids": tuple(c.get("source_ids", ())),
                            "report_lineage": tuple(c.get("report_lineage", ())),
                            "acquisition_time_s": c.get("acquisition_time_s"),
                        }
                    )
            else:
                fused_measurements.append(
                    {
                        "T": None,
                        "az": c["az"],
                        "el": c["el"],
                        "d": c["d"],
                        "dop": c["dop"],
                        "lat": c.get("lat"),
                        "lon": c.get("lon"),
                        "alt": c.get("alt"),
                        "n_obs": n_obs,
                        "is_deception": c.get("is_deception", False),
                        "engagement_id": c.get("engagement_id", None),
                        "jammer_id": c.get("jammer_id", None),
                        "report_ids": tuple(c.get("report_ids", ())),
                        "source_ids": tuple(c.get("source_ids", ())),
                        "report_lineage": tuple(c.get("report_lineage", ())),
                        "acquisition_time_s": c.get("acquisition_time_s"),
                    }
                )
        return fused_measurements
