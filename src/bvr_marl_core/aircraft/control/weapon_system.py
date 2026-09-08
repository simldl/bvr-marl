import logging

import numpy as np

from bvr_marl_core.aircraft.control.gun_projectile import GunSystem
from bvr_marl_core.domain.information import TrackLifecycle, TrackSnapshot, WeaponTrack
from bvr_marl_core.domain.tactical_contact import TacticalContact
from bvr_marl_core.radar.core.utils import enu_to_geodetic
from bvr_marl_core.simulator.core.helpers import units_distance_km
from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff
from bvr_marl_core.simulator.utils.geodesics import geodetic_bearing_deg

logger = logging.getLogger(__name__)


class AircraftWeaponSystem:
    def __init__(self, parent):
        self.parent = parent
        self.missile_types = getattr(parent, "missile_types", [])
        self.max_missiles = getattr(parent, "max_missiles", 8)
        self.remaining_missiles = getattr(parent, "max_missiles", 8)
        self.missiles = getattr(parent, "missiles", [])

        # Gun system
        gun_config = getattr(parent, "gun_config", {})
        self.gun = GunSystem(
            parent_aircraft=parent,
            max_ammo=gun_config.get("max_ammo", 500),
            muzzle_velocity_mps=gun_config.get("muzzle_velocity_mps", 1000.0),
            max_range_m=gun_config.get("max_range_m", 800.0),
            burst_size=gun_config.get("burst_size", 10),
            burst_duration_s=gun_config.get("burst_duration_s", 0.1),
        )

    def should_fire(self, fire=1):
        return fire > 0.5

    def is_target_in_fov(self, target):
        """
        Check if target lies within the lock FOV for weapons engagement.

        For most aircraft, this is the radar's horizontal FOV.
        For AWACS (and similar support aircraft), this uses a separate lock_fov_deg
        that is narrower than the detection FOV. AWACS can detect in 360° but
        can only lock/engage targets within lock_fov_deg of their heading.

        Args:
            target: The target unit to check.

        Returns:
            True if target is within the lock FOV.
        """

        target_bearing = geodetic_bearing_deg(
            self.parent.position.lat,
            self.parent.position.lon,
            target.position.lat,
            target.position.lon,
        )
        angle_diff = abs(signed_yaw_deg_diff(self.parent.yaw_deg, target_bearing))

        # AWACS uses a narrower lock_fov_deg than its 360° detection FOV
        if hasattr(self.parent, "lock_fov_deg") and self.parent.lock_fov_deg is not None:
            return angle_diff < (float(self.parent.lock_fov_deg) * 0.5)

        hfov = getattr(getattr(self.parent, "radar", None), "h_fov_deg", 90.0)
        return angle_diff < (float(hfov) * 0.5)

    def is_contact_in_fov(self, contact: TacticalContact) -> bool:
        """Check weapon FOV using only the contact's estimated ENU position."""
        east_m, north_m = contact.state[:2]
        bearing_deg = np.degrees(np.arctan2(east_m, north_m))
        angle_diff = abs(signed_yaw_deg_diff(self.parent.yaw_deg, bearing_deg))
        lock_fov = getattr(self.parent, "lock_fov_deg", None)
        if lock_fov is not None:
            return angle_diff < float(lock_fov) * 0.5
        hfov = getattr(getattr(self.parent, "radar", None), "h_fov_deg", 90.0)
        return angle_diff < float(hfov) * 0.5

    def check_fire_feasibility(self, target):
        """
        Check if firing at target is feasible (for observations).
        Returns dict with all gate statuses.
        """
        feasibility = {
            "inventory_ok": False,
            "radar_lock_ok": False,
            "fov_ok": False,
            "can_fire": False,
            "remaining_missiles": 0,
            "veto_reason": None,
        }

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        feasibility["remaining_missiles"] = remaining
        if remaining <= 0:
            feasibility["veto_reason"] = "no_inventory"
            return feasibility
        feasibility["inventory_ok"] = True

        if not hasattr(self.parent, "sensor") or not self.parent.sensor.has_radar_lock(target):
            feasibility["veto_reason"] = "no_radar_lock"
            return feasibility
        feasibility["radar_lock_ok"] = True

        if not self.is_target_in_fov(target):
            feasibility["veto_reason"] = "not_in_fov"
            return feasibility
        feasibility["fov_ok"] = True

        feasibility["can_fire"] = True
        return feasibility

    def select_and_engage_target(
        self, target_candidates, target_selector, fire_action, gun_fire_action, simulator
    ):
        if not target_candidates:
            return None

        selected = self.parent.sensor.select_target(target_candidates, target_selector)
        self.parent.target = selected

        if selected is not None:
            if fire_action > 0.5:
                self.fire_missile(simulator, selected, self.missile_types[0])
            if gun_fire_action > 0.5:
                self.fire_gun(simulator, selected)

        return selected

    def fire_gun(self, sim, target=None):
        """Fire the gun at a target or straight ahead."""
        if getattr(self.parent, "is_mortally_hit", False):
            return []
        if not self.gun.can_fire(sim.utc_time):
            return []
        target_position = (
            self._calculate_gun_intercept(target)
            if target is not None and hasattr(target, "position")
            else None
        )
        return self.gun.start_firing(sim, target_position)

    def fire_gun_at_contact(self, sim, contact: TacticalContact):
        """Fire a predicted burst using track geometry without a target object."""
        if getattr(self.parent, "is_mortally_hit", False) or not contact.engageable:
            return []
        if contact.suspect_deception or contact.is_missile or not self.is_contact_in_fov(contact):
            return []
        position = np.asarray(contact.state[:3], dtype=float)
        target_velocity = np.asarray(contact.state[3:6], dtype=float)
        parent_velocity = np.asarray(
            [
                float(getattr(getattr(self.parent, "velocity", None), axis, 0.0))
                for axis in ("vx", "vy", "vz")
            ]
        )
        distance_m = float(np.linalg.norm(position))
        time_to_intercept_s = distance_m / max(float(self.gun.muzzle_velocity_mps), 1.0)
        predicted_enu = position + (target_velocity - parent_velocity) * time_to_intercept_s
        latitude, longitude, altitude = enu_to_geodetic(
            predicted_enu,
            self.parent.position.lat,
            self.parent.position.lon,
            self.parent.position.alt,
        )
        return self.gun.start_firing(sim, (longitude, latitude, altitude))

    def _calculate_gun_intercept(self, target):
        """Calculate lead-angle intercept point for gun firing."""
        rel_pos = np.array(
            [
                target.position.lon - self.parent.position.lon,
                target.position.lat - self.parent.position.lat,
                target.position.alt - self.parent.position.alt,
            ]
        )
        rel_pos_m = rel_pos * np.array(
            [111000 * np.cos(np.radians(self.parent.position.lat)), 111000, 1]
        )
        rel_vel = np.array(
            [
                target.velocity.vx - self.parent.velocity.vx,
                target.velocity.vy - self.parent.velocity.vy,
                target.velocity.vz - self.parent.velocity.vz,
            ]
        )
        range_to_target = np.linalg.norm(rel_pos_m)
        if range_to_target > 0:
            time_to_intercept = range_to_target / self.gun.muzzle_velocity_mps
            predicted_pos = rel_pos_m + rel_vel * time_to_intercept
            predicted_lat = self.parent.position.lat + predicted_pos[1] / 111000
            predicted_lon = self.parent.position.lon + predicted_pos[0] / (
                111000 * np.cos(np.radians(self.parent.position.lat))
            )
            predicted_alt = self.parent.position.alt + predicted_pos[2]
            return (predicted_lon, predicted_lat, predicted_alt)
        return (target.position.lon, target.position.lat, target.position.alt)

    def _record_launch_context(self, sim, missile, target):
        """Cache cheap launch-context fields on the missile for the terminal-event
        record. Computed once at fire time and never recomputed in the sim loop;
        fully guarded so it can never block a launch.
        """
        parent = self.parent
        p_pos = getattr(parent, "position", None)
        t_pos = getattr(target, "position", None)
        ctx = {
            "launch_time_s": getattr(sim, "elapsed_time_s", None),
            "shooter_alt_m": getattr(p_pos, "alt", None),
            "shooter_speed_mps": getattr(parent, "speed", None),
            "target_alt_m": getattr(t_pos, "alt", None),
            "target_speed_mps": getattr(target, "speed", None),
        }
        try:
            ctx["range_m"] = units_distance_km(parent, target) * 1000.0
        except Exception:
            ctx["range_m"] = None
        try:
            tgt_to_shooter = geodetic_bearing_deg(t_pos.lat, t_pos.lon, p_pos.lat, p_pos.lon)
            ctx["aspect_deg"] = abs(signed_yaw_deg_diff(target.yaw_deg, tgt_to_shooter))
        except Exception:
            ctx["aspect_deg"] = None
        # Launch DLZ/NEZ from the shooter's (per-tick cached) zone calculator.
        try:
            wez = getattr(parent, "wez", None)
            if wez is not None:
                dlz = wez.compute_dlz(target)
                rng = wez._slant_range_m(parent, target)
                ctx["zone"] = wez.zone_for_range(rng, dlz)
                ctx["nez_out_m"] = dlz.r_nez_out_m
                ctx["r_aero_m"] = dlz.r_aero_m
                ctx["r_min_m"] = dlz.r_min_m
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
            pass
        missile._launch_context = ctx

    def fire_missile(self, sim, target, missile_cls):
        """
        Fire a missile with comprehensive gating logic.

        Returns:
            - On success: (missile, None, diagnostics_dict)
            - On failure: (None, veto_reason_str, diagnostics_dict)
        """
        diagnostics = {
            "has_inventory": False,
            "has_lock": False,
            "in_locked_tracks": False,
            "in_fov": False,
            "target_id": getattr(target, "id", None),
            "is_engageable": True,
        }

        # Gate -1: a mortally-hit (dying) aircraft is out of the fight.
        if getattr(self.parent, "is_mortally_hit", False):
            return None, "shooter_mortally_hit", diagnostics

        # Gate 0: Filter out support assets (e.g. AWACS)
        if getattr(target, "is_non_engageable", False):
            diagnostics["is_engageable"] = False
            veto = f"target_non_engageable(target={getattr(target, 'id', 'unknown')}, is_support={getattr(target, 'is_support_asset', False)})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics["has_inventory"] = remaining > 0
        diagnostics["remaining"] = remaining

        if remaining <= 0:
            veto = f"no_inventory(remaining={remaining})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        has_sensor = hasattr(self.parent, "sensor") and self.parent.sensor is not None
        target_id = getattr(target, "id", None)
        resolve_contacts = getattr(sim, "evaluator_contact_ids_for_truth", None)
        own_contact_ids = (
            resolve_contacts(getattr(self.parent, "id", None), target_id)
            if callable(resolve_contacts) and target_id is not None
            else {target_id}
        )
        locked_contact_ids = (
            set(self.parent.sensor.get_locked_targets() or ()) if has_sensor else set()
        )
        own_locked_contacts = own_contact_ids & locked_contact_ids
        own_lock = (
            bool(own_locked_contacts)
            if callable(resolve_contacts)
            else bool(has_sensor and target and self.parent.sensor.has_radar_lock(target))
        )
        if not own_lock:
            own_locked_contacts = set()

        # AWACS / datalink-cued lock: a full-datalink friendly (AWACS within its
        # locking cone, or a wingman) holding the target lets us launch beyond our
        # own radar range and fly the missile on datalink midcourse guidance.
        datalink_lock = False
        if target is not None and not own_lock:
            from bvr_marl_core.radar.core.data_link import DataLink

            group_locked = DataLink.group_locked_target_ids(sim, self.parent)
            friendly_ids = {
                getattr(unit, "id", None)
                for unit in getattr(sim, "active_units", {}).values()
                if getattr(unit, "group", None) == getattr(self.parent, "group", None)
            }
            datalink_contact_ids = (
                set().union(*(resolve_contacts(sensor_id, target_id) for sensor_id in friendly_ids))
                if callable(resolve_contacts)
                else {target_id}
            )
            datalink_lock = bool(datalink_contact_ids & set(group_locked))

        has_lock = own_lock or datalink_lock
        diagnostics["has_lock"] = has_lock
        diagnostics["datalink_lock"] = datalink_lock

        if not has_lock:
            veto = f"no_radar_lock(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        tracker_state = None
        # A datalink/AWACS-cued shot need not be in our own locked-tracks set; the
        # missile seeds from the target directly and updates via midcourse.  Only
        # verify local locked-tracks membership for an own-radar lock.
        locked_tracks = locked_contact_ids if own_lock else set()

        if own_lock:
            if isinstance(locked_tracks, dict):
                locked_contact_id = next(iter(own_locked_contacts), None)
                if locked_contact_id not in locked_tracks:
                    veto = f"not_in_locked_tracks_dict(target={target.id},locked={list(locked_tracks.keys())})"
                    logger.debug(f"Fire veto: {veto}")
                    return None, veto, diagnostics
                diagnostics["in_locked_tracks"] = True
                entry = locked_tracks[locked_contact_id]
                tracker_state = entry[0] if isinstance(entry, tuple) else entry
            elif isinstance(locked_tracks, set):
                locked_contact_id = next(iter(own_locked_contacts), None)
                if locked_contact_id not in locked_tracks:
                    veto = f"not_in_locked_tracks_set(target={target.id},locked={locked_tracks})"
                    logger.debug(f"Fire veto: {veto}")
                    return None, veto, diagnostics
                diagnostics["in_locked_tracks"] = True
                # Retrieve the 6-DOF track state from the radar's cached tracks.
                # state[:3] is ENU offset from the shooter position.
                tracker_state = None
                radar = getattr(self.parent, "radar", None)
                if radar is not None:
                    for track in getattr(radar, "cached_tracks", None) or []:
                        if track.track_id == locked_contact_id:
                            tracker_state = track.state
                            break
            else:
                veto = f"invalid_locked_tracks_type(type={type(locked_tracks)})"
                logger.debug(f"Fire veto: {veto}")
                return None, veto, diagnostics
        else:
            # Datalink/AWACS-cued lock: seed the missile from the target directly.
            diagnostics["in_locked_tracks"] = True

        in_fov = self.is_target_in_fov(target)
        diagnostics["in_fov"] = in_fov

        if not in_fov:
            veto = f"not_in_fov(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto: {veto}")
            return None, veto, diagnostics

        tracker_vel = (
            tracker_state[3:6] if tracker_state is not None and len(tracker_state) >= 6 else None
        )
        tracker_ref = self.parent.position.copy() if tracker_state is not None else None

        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=target,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group,
        )
        target_id = getattr(target, "id", None)
        missile.designated_target_id = target_id
        missile.retarget_policy = "locked_override"
        if hasattr(missile, "target_provider"):
            missile.target_provider.current_target_id = target_id

        if tracker_state is not None:
            missile.initial_tracked_position_enu = tracker_state[:3]
            missile.initial_tracked_velocity_enu = tracker_vel
            missile.tracker_reference_pos = tracker_ref
        missile.tracking_lock = True

        if hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", target_id)
        if hasattr(missile, "radar"):
            missile.radar.locked_target = target_id

        sim.add_unit(missile)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1
        self._record_launch_context(sim, missile, target)

        logger.info(
            f"Missile fired successfully: target={target.id}, remaining={self.remaining_missiles}"
        )
        diagnostics["fired"] = True
        return missile, None, diagnostics

    def _contact_is_protected(self, sim, contact: TacticalContact) -> bool:
        """True when this contact resolves to a unit the scenario forbids engaging."""
        resolve_truth = getattr(sim, "evaluator_truth_id_for_contact", None)
        if not callable(resolve_truth):
            return False
        truth_id = resolve_truth(getattr(self.parent, "id", None), contact.track_id)
        if truth_id is None:
            return False
        target = getattr(sim, "active_units", {}).get(truth_id)
        return bool(getattr(target, "is_non_engageable", False))

    def fire_missile_at_contact(self, sim, contact: TacticalContact, missile_cls):
        """Launch from an immutable operational track without resolving a Unit."""
        diagnostics = {
            "has_inventory": False,
            "has_lock": False,
            "in_locked_tracks": False,
            "in_fov": False,
            "target_id": contact.track_id,
            "selected_track_id": contact.track_id,
            "is_engageable": contact.engageable,
        }
        if not contact.engageable or contact.suspect_deception or contact.is_missile:
            return None, "contact_not_engageable", diagnostics
        if getattr(self.parent, "is_mortally_hit", False):
            return None, "shooter_mortally_hit", diagnostics

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics["has_inventory"] = remaining > 0
        diagnostics["remaining"] = remaining
        if remaining <= 0:
            return None, "no_inventory", diagnostics

        tracks = getattr(getattr(self.parent, "sensor", None), "sensor_tracks", ()) or ()
        matching_track = next(
            (track for track in tracks if track.track_id == contact.track_id),
            None,
        )
        if matching_track is None:
            return None, "contact_track_not_current", diagnostics
        if matching_track.suspect_deception or not matching_track.engageable:
            return None, "contact_track_not_engageable", diagnostics
        # A track is "engageable" only in the tracker's sense -- confirmed and not
        # deception. It carries no rules-of-engagement, so without this the protection
        # on units flagged non-engageable (AWACS by default) applied on the oracle fire
        # paths but not here, and a sensor-limited agent could spend its whole load on
        # a support aircraft. Resolved at the evaluator boundary, like the truth
        # association below; the weapon and its guidance still receive nothing.
        if self._contact_is_protected(sim, contact):
            diagnostics["is_engageable"] = False
            return None, "contact_non_engageable_roe", diagnostics

        diagnostics["has_lock"] = True
        diagnostics["in_locked_tracks"] = True
        diagnostics["in_fov"] = self.is_contact_in_fov(contact)
        if not diagnostics["in_fov"]:
            return None, "contact_not_in_fov", diagnostics

        now_s = float(getattr(sim, "elapsed_time_s", 0.0))
        snapshot = TrackSnapshot(
            track_id=contact.track_id,
            state_time_s=now_s,
            state=contact.state,
            covariance=contact.covariance,
            confidence=contact.confidence,
            lifecycle=TrackLifecycle.CONFIRMED,
            classification_probabilities=contact.classification_probabilities,
            classification_entropy_nats=contact.classification_entropy_nats,
            effective_classification_evidence=contact.effective_classification_evidence,
            source_ids=contact.source_ids,
            report_lineage=contact.report_lineage,
            age_s=contact.age_s,
        )
        weapon_track = WeaponTrack(snapshot=snapshot, launch_time_s=now_s)
        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=weapon_track,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group,
        )
        missile.weapon_track = weapon_track
        # Three distinct identities, deliberately not conflated:
        #   launch_contact_id  - immutable, the contact this weapon was committed to.
        #   designated_track_id - the track guidance currently prosecutes; the tracker
        #                         may retire and re-issue it during the flight.
        #   the physical unit   - evaluator-only, and never stored on the weapon.
        missile.launch_sensor_id = getattr(self.parent, "id", None)
        missile.launch_contact_id = contact.track_id
        missile.launch_report_lineage = weapon_track.snapshot.report_lineage
        missile.designated_target_id = contact.track_id
        missile.designated_track_id = contact.track_id
        missile.retarget_policy = "track_only"
        missile.initial_tracked_position_enu = np.asarray(contact.state[:3], dtype=float)
        missile.initial_tracked_velocity_enu = np.asarray(contact.state[3:6], dtype=float)
        missile.tracker_reference_pos = self.parent.position.copy()
        missile.tracking_lock = True
        missile.target_provider.current_target_id = contact.track_id
        if hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", contact.track_id)
        missile.radar.locked_target = contact.track_id

        sim.add_unit(missile)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1

        # Keep the contact identity and its evidence in the evaluator registry, but
        # do not commit this sensor-limited shot to the contact's current truth vote.
        # A fused formation track can have a slight (and transient) report majority
        # for the wrong aircraft. Committing that vote here makes guidance follow one
        # member while collision detection checks another, so the missile can pass
        # through the aircraft it actually intercepted. The evaluator resolves the
        # physical member from the weapon's continuing guidance hypothesis and only
        # commits that choice in the terminal region.
        register_contact = getattr(sim, "register_weapon_contact_association", None)
        if callable(register_contact):
            register_contact(
                missile.id,
                getattr(self.parent, "id", None),
                contact.track_id,
                weapon_track.snapshot.report_lineage,
            )

        diagnostics["fired"] = True
        return missile, None, diagnostics

    def fire_missile_direct(self, sim, target, missile_cls):
        """
        Fire from an evaluator-selected target without retaining the target object.

        This explicit oracle/test helper bypasses radar lock and FOV gates, but the
        privilege ends at launch: truth is sampled once into an immutable WeaponTrack.
        The live target is registered only in the evaluator collision registry.

        Returns:
            - On success: (missile, None, diagnostics_dict)
            - On failure: (None, veto_reason_str, diagnostics_dict)
        """
        diagnostics = {
            "has_inventory": False,
            "has_lock": False,
            "in_locked_tracks": False,
            "in_fov": True,
            "target_id": getattr(target, "id", None),
            "is_engageable": True,
        }

        if getattr(self.parent, "is_mortally_hit", False):
            return None, "shooter_mortally_hit", diagnostics

        if getattr(target, "is_non_engageable", False):
            diagnostics["is_engageable"] = False
            veto = f"target_non_engageable(target={getattr(target, 'id', 'unknown')})"
            logger.debug(f"Fire veto (direct): {veto}")
            return None, veto, diagnostics

        remaining = getattr(self.parent, "remaining_missiles", None)
        if remaining is None:
            remaining = self.remaining_missiles
        diagnostics["has_inventory"] = remaining > 0
        diagnostics["remaining"] = remaining

        if remaining <= 0:
            veto = f"no_inventory(remaining={remaining})"
            logger.debug(f"Fire veto (direct): {veto}")
            return None, veto, diagnostics

        # Seed missile with truth ENU offset so PN guidance bootstraps correctly
        try:
            from bvr_marl_core.radar.core.utils import geodetic_to_enu

            shooter_pos = self.parent.position
            tgt_pos = target.position
            enu_offset = np.asarray(
                geodetic_to_enu(
                    tgt_pos.lat,
                    tgt_pos.lon,
                    tgt_pos.alt,
                    shooter_pos.lat,
                    shooter_pos.lon,
                    shooter_pos.alt,
                ),
                dtype=np.float32,
            )
        except Exception:
            enu_offset = None

        tgt_vel_enu = None
        if hasattr(target, "velocity_enu"):
            tgt_vel_enu = np.asarray(target.velocity_enu, dtype=np.float32)

        target_id = getattr(target, "id", None)
        now_s = float(getattr(sim, "elapsed_time_s", 0.0))
        state = np.zeros(6, dtype=float)
        if enu_offset is not None:
            state[:3] = enu_offset
        if tgt_vel_enu is not None:
            state[3:6] = tgt_vel_enu
        snapshot = TrackSnapshot(
            track_id=int(target_id),
            state_time_s=now_s,
            state=tuple(state),
            covariance=tuple(tuple(row) for row in np.eye(6)),
            confidence=1.0,
            lifecycle=TrackLifecycle.CONFIRMED,
            age_s=0.0,
        )
        weapon_track = WeaponTrack(snapshot=snapshot, launch_time_s=now_s)
        missile = missile_cls(
            firing_time_s=sim.utc_time,
            target=weapon_track,
            source=self.parent,
            map_limits=self.parent.map_limits,
            group=self.parent.group,
        )
        missile.weapon_track = weapon_track
        missile.designated_target_id = target_id
        missile.designated_track_id = target_id
        missile.retarget_policy = "track_only"
        missile.oracle_direct_launch = True
        if hasattr(missile, "target_provider"):
            missile.target_provider.current_target_id = target_id

        if enu_offset is not None:
            tracker_state = np.zeros(6, dtype=np.float32)
            tracker_state[:3] = enu_offset
            if tgt_vel_enu is not None:
                tracker_state[3:6] = tgt_vel_enu
            missile.initial_tracked_position_enu = tracker_state[:3]
            missile.initial_tracked_velocity_enu = tracker_state[3:6]
            missile.tracker_reference_pos = self.parent.position.copy()

        missile.tracking_lock = True

        if hasattr(missile, "radar") and hasattr(missile.radar, "lock_ctrl"):
            missile.radar.lock_ctrl.set_mode("track", target_id)
        if hasattr(missile, "radar"):
            missile.radar.locked_target = target_id

        sim.add_unit(missile)
        register = getattr(sim, "register_weapon_truth_association", None)
        if target_id is not None and callable(register):
            register(missile.id, target_id)
        self.parent.missiles.append(missile)
        self.remaining_missiles -= 1
        self._record_launch_context(sim, missile, target)

        logger.info(
            f"Missile fired (direct): target={getattr(target, 'id', 'none')}, remaining={self.remaining_missiles}"
        )
        diagnostics["fired"] = True
        return missile, None, diagnostics
