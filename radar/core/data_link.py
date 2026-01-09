import numpy as np
from collections import defaultdict
from radar.core.utils import geodetic_to_enu, enu_to_geodetic, to_cart, _angles_dist
import math

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

    def get_fused_clusters(self, own_radar, group_radars, own_position, clusterer):
        raw_detections = self._gather_raw_detections(self.get_mode(), own_radar, group_radars, own_position)
        src_ids = []
        src_counts = []
        for (src_radar, _src_pos) in (group_radars or []):
            src_ids.append(getattr(getattr(src_radar.owner, "id", None), "__int__", lambda: getattr(src_radar.owner, "id", None))())
            src_counts.append(len(getattr(src_radar, "cached_detections", []) or []))
        clusters = clusterer.cluster(raw_detections)
        fused = self._fuse_clusters_by_id_and_false_alarm(clusters)

        # Apply ECCM parallax filtering if multi-radar datalink
        if group_radars and len(group_radars) > 0:
            fused = self._eccm_parallax_filter(fused, group_radars)

        return fused

    def _gather_raw_detections(self, mode, own_radar, group_radars, own_position):
        detections = []

        if mode == "own":
            if own_radar and own_radar.cached_detections:
                detections += own_radar.cached_detections

        elif mode == "full":
            if own_radar and own_radar.cached_detections:
                detections += own_radar.cached_detections
            for src_radar, src_pos in group_radars or []:
                if src_radar is not own_radar and src_radar.data_link and src_radar.data_link.get_mode() == "full":
                    for d in src_radar.cached_detections or []:
                        detections.append(self._transform_detection_to_local_ref(d, src_pos, own_position))

        elif mode == "other":
            for src_radar, src_pos in group_radars or []:
                if src_radar is not own_radar and src_radar.data_link and src_radar.data_link.get_mode() in ("full", "own"):
                    for d in src_radar.cached_detections or []:
                        detections.append(self._transform_detection_to_local_ref(d, src_pos, own_position))

        elif mode == "msl_support":
            # (i) Always include own seeker detections, if any
            if own_radar and own_radar.cached_detections:
                detections += own_radar.cached_detections

            # Determine missile's current target IDs (locked or designated)
            own_target_ids = set()
            try:
                own_target_ids |= set(own_radar.get_locked_targets() or [])
            except Exception:
                pass

            # If missile has a target provider, try to include that, too
            try:
                tp = getattr(own_radar, "target_provider", None)
                if tp is not None and hasattr(tp, "get_guidance_target"):
                    tgt_pos = tp.get_guidance_target()
                    tgt_obj = getattr(tp, "get_guidance_target_object", lambda: None)()
                    if tgt_obj is not None and hasattr(tgt_obj, "id"):
                        own_target_ids.add(tgt_obj.id)
            except Exception:
                pass

            # (ii) Include ONLY detections from friendlies that:
            #   - are tracking at least one of own_target_ids, AND
            #   - currently have the missile in their FOV (az/el within half-FOV), AND
            #   - (optionally) are within range
            if len(own_target_ids) > 0:
                for src_radar, src_pos in group_radars or []:
                    if src_radar is own_radar or not getattr(src_radar, "owner", None):
                        continue

                    # Only consume from sources willing to share (mirror your 'other' logic leniently)
                    src_mode = getattr(getattr(src_radar, "data_link", None), "get_mode", lambda: "none")()
                    if src_mode not in ("full", "own"):
                        continue

                    # Must be tracking the same target
                    try:
                        src_locked = set(src_radar.get_locked_targets() or [])
                    except Exception:
                        src_locked = set()
                    if src_locked.isdisjoint(own_target_ids):
                        continue

                    # Missile must be in the source radar's FOV
                    # Use off-boresight differences returned by _angles_dist()
                    try:
                        az_off, el_off, dist = _angles_dist(src_pos, float(src_radar.yaw_deg), float(src_radar.pitch_deg), own_position)
                    except Exception:
                        # Fallback if yaw/pitch properties not present
                        az_off, el_off, dist = _angles_dist(src_pos, 0.0, 0.0, own_position)

                    h_half = float(getattr(src_radar, "h_fov_deg", 120.0)) * 0.5
                    v_half = float(getattr(src_radar, "v_fov_deg", 60.0)) * 0.5
                    max_range_m = float(getattr(src_radar, "max_range_m", 1e9))

                    if (abs(az_off) <= h_half) and (abs(el_off) <= v_half) and (dist <= max_range_m):
                        # Bring only relevant detections (those about our target IDs)
                        src_dets = getattr(src_radar, "cached_detections", []) or []
                        for d in src_dets:
                            tid = getattr(d.get("T", None), "id", None)
                            if (tid is not None) and (tid in own_target_ids):
                                detections.append(self._transform_detection_to_local_ref(d, src_pos, own_position))

        return detections

    def _transform_detection_to_local_ref(self, detection, src_ref_pos, tgt_ref_pos):
        lat, lon, alt = detection["lat"], detection["lon"], detection["alt"]
        enu = geodetic_to_enu(lat, lon, alt, src_ref_pos.lat, src_ref_pos.lon, src_ref_pos.alt)
        lat2, lon2, alt2 = enu_to_geodetic(enu, tgt_ref_pos.lat, tgt_ref_pos.lon, tgt_ref_pos.alt)
        detection_new = dict(detection)
        detection_new["lat"], detection_new["lon"], detection_new["alt"] = lat2, lon2, alt2
        return detection_new

    @staticmethod
    def update_group_radars(sim, owner=None):
        return [
            (u.radar, u.position)
            for u in sim.active_units.values()
            if hasattr(u, "radar")
            and hasattr(u.radar, "data_link")
            and u.radar.data_link.get_mode() == "full"
            and (owner is None or u is not owner)
            and getattr(u, "group", None) == getattr(owner, "group", None)
        ]

    def _fuse_clusters_by_id_and_false_alarm(self, clusters):
        fused_measurements = []
        for c in clusters:
            # Handle T being list, None, or single object
            T_field = c.get("T", None)
            T_list = T_field if isinstance(T_field, list) else ([T_field] if T_field is not None else [])

            ids = [getattr(t, "id", None) for t in T_list]
            n_obs = len(T_list)
            id_set = set([i for i in ids if i is not None])

            if id_set:
                for tid in id_set:
                    group = [t for t in T_list if getattr(t, "id", None) == tid]
                    fused_measurements.append({
                        'T': group[0] if group else None,
                        'az': c["az"], 'el': c["el"], 'd': c["d"], 'dop': c["dop"],
                        'lat': c.get('lat'), 'lon': c.get('lon'), 'alt': c.get('alt'),
                        'n_obs': len(group),
                        # Propagate ECM fields from cluster
                        'is_deception': c.get('is_deception', False),
                        'engagement_id': c.get('engagement_id', tid if group else None),  # Use real target ID if no engagement_id
                        'jammer_id': c.get('jammer_id', None),
                    })
            else:
                # No valid targets found - could be noise, ghosts, or edge case
                fused_measurements.append({
                    'T': None,
                    'az': c["az"], 'el': c["el"], 'd': c["d"], 'dop': c["dop"],
                    'lat': c.get('lat'), 'lon': c.get('lon'), 'alt': c.get('alt'),
                    'n_obs': n_obs,
                    # Propagate ECM fields
                    'is_deception': c.get('is_deception', False),
                    'engagement_id': c.get('engagement_id', None),
                    'jammer_id': c.get('jammer_id', None),
                })
        return fused_measurements

    def _eccm_parallax_filter(self, clusters, group_radars, parallax_threshold_km=2.5):
        """
        ECCM parallax consistency check: cross-radar triangulation.

        Ghost detections from enemy jammers will appear at different 3D positions
        when viewed from different friendly radars. This inconsistency reveals deception.

        Args:
            clusters: List of fused cluster dicts
            group_radars: List of (radar, position) tuples for friendly radars
            parallax_threshold_km: Maximum position dispersion (km) for consistent tracks

        Returns:
            Filtered clusters with suspected ghosts removed
        """
        # Group clusters by engagement_id
        grouped = defaultdict(list)
        for c in clusters:
            eid = c.get('engagement_id', None)
            grouped[eid].append(c)

        out = []
        for eid, group in grouped.items():
            # Pass through clusters without engagement_id
            if eid is None:
                out.extend(group)
                continue

            # Need at least 2 observations for parallax check
            if len(group) < 2:
                out.extend(group)
                continue

            # Convert cluster positions to ENU (use only geodetic path for consistency)
            ref = group_radars[0][1] if group_radars else None
            if not ref:
                out.extend(group)
                continue

            positions = []
            for c in group:
                # Only use geodetic coordinates for consistent frame comparison
                if all(k in c for k in ('lat', 'lon', 'alt')):
                    enu = geodetic_to_enu(c['lat'], c['lon'], c['alt'],
                                        ref.lat, ref.lon, ref.alt)
                    positions.append(enu)

            # Check position consistency
            if len(positions) >= 2:
                positions_array = np.array(positions)
                mean_pos = positions_array.mean(axis=0)
                deviations = np.linalg.norm(positions_array - mean_pos, axis=1)
                max_deviation_m = deviations.max()

                # If position dispersion exceeds threshold, drop deception clusters
                if max_deviation_m > parallax_threshold_km * 1000.0:
                    # Inconsistent across radars → likely ghosts
                    group = [c for c in group if not c.get('is_deception', False)]

            out.extend(group)

        return out