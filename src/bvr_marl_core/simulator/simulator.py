from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta

from bvr_marl_core.simulator.core.events import Event, UnitRegisteredEvent, UnitRemovedEvent
from bvr_marl_core.simulator.core.experiment_metadata import build_experiment_metadata
from bvr_marl_core.simulator.core.helpers import units_distance_km
from bvr_marl_core.simulator.core.hit_event_helpers import (
    CCDConfig,
    MissileCCDManager,
    stochastic_on_hit,
)
from bvr_marl_core.simulator.core.random_streams import EpisodeRandomStreams
from bvr_marl_core.simulator.core.substepping import SubstepConfig, TerminalPathResolver
from bvr_marl_core.simulator.core.tick_buffer import TickStateBuffer
from bvr_marl_core.simulator.core.units import Unit
from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km

logger = logging.getLogger(__name__)

# How many of each source's most recent reports vote on what a track is about.
# Wide enough that one mis-associated report cannot flip the attribution, short
# enough that the vote follows a track which re-associates onto another aircraft.
_ATTRIBUTION_WINDOW_PER_SOURCE = 8

# Below this weapon→intended-target range the geometric attribution fallback commits
# (latches) to the chosen aircraft. Beyond it the choice is left transient and
# re-evaluated every tick, so an early guess made while a formation is still fused
# cannot freeze the shot onto the wrong member before guidance has separated them.
_GEOMETRIC_ATTRIBUTION_COMMIT_KM = 3.0


class Simulator:
    def __init__(
        self,
        utc_time=None,
        tick_secs=1,
        random_seed=None,
        num_units=0,
        num_opp_units=0,
        weapon_config=None,
    ):
        self.active_units: dict[int, Unit] = {}
        self.trace_record_units = {}
        # Radar-emission duty per unit id, updated every step by the action processor
        # and PERSISTING across a unit's death, so episode-end active-sensing metrics
        # (paper Phase 4b) cover dead agents too. {unit_id: {"group", "duty"}}.
        self.emission_duty_records: dict[int, dict] = {}
        # Per-tick position/velocity trace, consumed ONLY by the visualization /
        # tacview export path. Disable during headless training to avoid the
        # unbounded per-episode growth and the per-unit copy each tick.
        self.record_traces = True
        self.events: list[Event] = []
        self.diagnostic_counters: Counter[str] = Counter()
        self._missile_target_counts: dict[str, dict[int, int]] = {}
        # Evaluator-only association. Operational missile/guidance objects never
        # receive the referenced Unit; collision and attribution code resolve it.
        self._weapon_truth_associations: dict[int, int] = {}
        # Which operational contact a weapon was launched against, so a shot taken
        # while the picture was still ambiguous can be attributed later.
        self._weapon_contact_associations: dict[int, tuple[object, object]] = {}
        self._weapon_launch_lineage: dict[int, tuple[tuple[object, int], ...]] = {}
        self._sensor_report_truth_associations: dict[tuple[object, int], int] = {}
        self._contact_truth_associations: dict[tuple[object, object], int] = {}
        self.network_pictures: dict[object, object] = {}
        resolved_utc_time = utc_time if utc_time is not None else datetime.now()
        self.utc_time = resolved_utc_time
        self.utc_time_initial = resolved_utc_time
        self.tick_secs = tick_secs
        self._tick_callbacks: list[Callable[[datetime], None]] = []
        self.random_seed = random_seed
        self.rnd_gen = random.Random(random_seed)
        self.random_streams = EpisodeRandomStreams(random_seed)
        self.replay_metadata = build_experiment_metadata()
        self.replay_metadata.update(self.random_streams.metadata())
        self._next_unit_id = 1
        self._next_event_id = 1
        self.status_text = None

        self.num_units = num_units
        self.num_opp_units = num_opp_units

        # Datalink 0/1 dropout: per-link, per-tick probability the link is down.
        # 0.0 = links always up (default). Set by the environment from config.
        self.datalink_drop_prob = 0.0
        self._datalink_link_states: dict[tuple[int, int], bool] = {}
        self.datalink_link_history: list[dict] = []
        self._tick_index = 0
        self._tick_in_progress = False
        self._pending_units: dict[int, Unit] = {}

        weapon_config = weapon_config or {}
        missile_hit_radius = weapon_config.get("missile_hit_radius_m", 500.0)
        gun_hit_radius = weapon_config.get("gun_hit_radius_m", 5.0)
        self.gun_hit_radius_m = gun_hit_radius

        self.ccd = MissileCCDManager(
            CCDConfig(
                fuse_radius_m=missile_hit_radius,
                target_radius_m=0.0,
                max_consider_range_m=50_000.0,
                within_lock_gate_m=10_000.0,
            ),
            on_hit=stochastic_on_hit,
        )

        # One authoritative hit calculator.  ``hit_calc`` remains an alias for
        # compatibility; the CCD manager owns the sole production instance.
        self.hit_calc = self.ccd.hit_calc

        # Electronic-warfare coordinator (noise jamming = range denial). Radars
        # query it each update to learn which enemy jammers are denying their range.
        from bvr_marl_core.radar.ew import EWWorld

        self.ew_world = EWWorld(self)

        self.terminal_resolver = TerminalPathResolver(
            config=SubstepConfig(
                # Gate must exceed the largest single-tick relative travel
                # (~1.7 km at BVR closing speeds) plus the fuse radius so a fast
                # fly-through is never skipped before sub-tick CCD runs.
                engage_distance_km=3,
                min_substeps=4,
                max_substeps=24,
                safety_travel_per_substep_m=150,
                physics_only=False,
            )
        )

    @property
    def elapsed_time_s(self):
        """Returns the elapsed simulation time in seconds."""
        return (self.utc_time - self.utc_time_initial).total_seconds()

    def seed(self, seed: int | None = None) -> None:
        """Reset episode RNG ownership to a new deterministic root seed."""
        self.random_seed = int(seed) if seed is not None else 0
        self.rnd_gen = random.Random(self.random_seed)
        self.random_streams = EpisodeRandomStreams(self.random_seed)
        self.replay_metadata.update(self.random_streams.metadata())

    def reset_sim(self, units, utc_time=None):
        """
        Reset the simulator to its initial state with a new set of units.
        :param units: A dictionary of units to initialize in the simulator.
        """
        self.utc_time = utc_time if utc_time is not None else datetime.now()
        self.utc_time_initial = self.utc_time
        self._tick_callbacks = []
        self.status_text = None
        validated_units = {}
        for unit_id, unit in units.items():
            if getattr(unit, "id", None) != unit_id:
                raise ValueError(
                    f"Unit dictionary key {unit_id!r} does not match object id "
                    f"{getattr(unit, 'id', None)!r}."
                )
            if unit_id in validated_units:
                raise ValueError(f"Duplicate unit id {unit_id!r} during reset.")
            validated_units[unit_id] = unit
        self.active_units = validated_units
        self.trace_record_units = {}
        self.emission_duty_records = {}
        self._next_unit_id = max(validated_units, default=0) + 1
        self._next_event_id = 1
        self._missile_target_counts = {}
        self._weapon_truth_associations = {}
        self._weapon_contact_associations = {}
        self._weapon_launch_lineage = {}
        self._sensor_report_truth_associations = {}
        self._contact_truth_associations = {}
        self.network_pictures = {}
        self.events = []
        self.diagnostic_counters.clear()
        self.rnd_gen = random.Random(self.random_seed)
        self.random_streams = EpisodeRandomStreams(self.random_seed)
        self.replay_metadata = build_experiment_metadata()
        self.replay_metadata.update(self.random_streams.metadata())
        self._datalink_link_states = {}
        self.datalink_link_history = []
        self._tick_index = 0
        self._tick_in_progress = False
        self._pending_units = {}
        if hasattr(self, "ccd") and hasattr(self.ccd, "reset_engagement_tracking"):
            self.ccd.reset_engagement_tracking()
        for unit_id, unit in units.items():
            self.record_unit_trace(unit_id)
        logger.debug("Simulator reset.")

    def add_unit(self, unit: Unit) -> int:
        if unit.id is not None and (unit.id in self.active_units or unit.id in self._pending_units):
            return unit.id

        if hasattr(unit, "target") and unit.target is not None:
            if unit.target.id is None or unit.target.id not in self.active_units:
                raise ValueError(
                    f"Target unit {unit.target} must be registered before adding unit {unit}."
                )
        unit.id = self._next_unit_id
        destination = self._pending_units if self._tick_in_progress else self.active_units
        destination[unit.id] = unit
        if self.record_traces and not self._tick_in_progress:
            self.trace_record_units[unit.id] = []
            self._store_unit_state(unit.id)
        self.log_event(UnitRegisteredEvent(self, unit))
        # Reseed any attached radar's numpy RNG now that the unit has a stable ID.
        # Aircraft construct their radar before add_unit assigns an ID, so the radar's
        # np_rng was seeded with None (OS entropy).  Reseeding here makes detection
        # outcomes reproducible across Python processes for a given random_seed.
        radar = getattr(unit, "radar", None)
        if radar is not None and hasattr(radar, "np_rng"):
            rng = self.random_streams.generator("radar", unit.id)
            radar.np_rng = rng
            if hasattr(radar, "obsgen") and radar.obsgen is not None:
                radar.obsgen.np_rng = rng
        self._next_unit_id += 1
        if getattr(unit, "is_missile", False):
            group = getattr(unit, "group", None)
            target_id = self._missile_target_id(unit)
            if group is not None and target_id is not None:
                grp_map = self._missile_target_counts.setdefault(group, {})
                grp_map[target_id] = grp_map.get(target_id, 0) + 1
        return unit.id

    def _publish_pending_units(self) -> None:
        """Expose mid-tick creations after all start-of-tick readers have run."""
        for unit_id, unit in self._pending_units.items():
            self.active_units[unit_id] = unit
            if self.record_traces:
                self.trace_record_units[unit_id] = []
                self._store_unit_state(unit_id)
        self._pending_units.clear()

    def allocate_event_id(self) -> int:
        """Allocate a replay-local event identity that is never reused."""
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def remove_unit(self, unit_id: int):
        if unit_id in self.active_units:
            unit = self.active_units[unit_id]
            if getattr(unit, "is_missile", False):
                group = getattr(unit, "group", None)
                target_id = self._missile_target_id(unit)
                if group is not None and target_id is not None:
                    grp_map = self._missile_target_counts.get(group, {})
                    if grp_map.get(target_id, 0) > 1:
                        grp_map[target_id] -= 1
                    elif target_id in grp_map:
                        del grp_map[target_id]
            self._weapon_truth_associations.pop(unit_id, None)
            self._weapon_contact_associations.pop(unit_id, None)
            self._weapon_launch_lineage.pop(unit_id, None)
            del self.active_units[unit_id]
            return unit
        return None

    def count_missiles_at_target(self, group: str, target_id: int) -> int:
        """Active missiles from *group* against a physical unit (oracle/legacy namespace)."""
        return self._missile_target_counts.get(group, {}).get(("unit", target_id), 0)

    def resync_missile_target(self, group: str, old_target_id, new_target_id) -> None:
        """Move a weapon's saturation count when a legacy/oracle missile retargets.

        Unit-space only: retargeting exists solely on the path that holds a Unit. The
        namespacing lives here so callers cannot drift from the key format the
        add_unit/remove_unit accounting uses.
        """
        if group is None:
            return
        if old_target_id is not None:
            grp_map = self._missile_target_counts.get(group, {})
            old_key = ("unit", old_target_id)
            if grp_map.get(old_key, 0) > 1:
                grp_map[old_key] -= 1
            elif old_key in grp_map:
                del grp_map[old_key]
        if new_target_id is not None:
            grp_map = self._missile_target_counts.setdefault(group, {})
            new_key = ("unit", new_target_id)
            grp_map[new_key] = grp_map.get(new_key, 0) + 1

    def count_missiles_at_contact(
        self,
        group: str,
        sensor_id: object,
        contact_id: object,
        report_lineage=(),
    ) -> int:
        """Active missiles from *group* against an operational contact.

        Saturation has to be counted in the namespace the shooter actually selects in.
        A weapon-track missile is launched against a contact id, which shares no
        namespace with unit ids -- counting the two together silently both misses caps
        and invents them when a track number happens to equal a live unit id. Shared
        report identities let teammates recognize the same network contact even when
        their local track IDs differ, without consulting evaluator truth.
        """
        lineage = set(report_lineage or ())
        if lineage:
            missiles = (*self.active_units.values(), *self._pending_units.values())
            return sum(
                1
                for missile in missiles
                if getattr(missile, "is_missile", False)
                and getattr(missile, "group", None) == group
                and lineage.intersection(getattr(missile, "launch_report_lineage", ()) or ())
            )
        return self._missile_target_counts.get(group, {}).get(("contact", sensor_id, contact_id), 0)

    @staticmethod
    def _missile_target_id(missile):
        """Namespaced saturation key for a weapon: contact-space or unit-space."""
        from bvr_marl_core.domain.information import WeaponTrack

        if isinstance(getattr(missile, "weapon_track", None), WeaponTrack):
            contact_id = getattr(missile, "launch_contact_id", None)
            if contact_id is None:
                return None
            return ("contact", getattr(missile, "launch_sensor_id", None), contact_id)
        designated = getattr(missile, "designated_target_id", None)
        if isinstance(designated, (int, str)):
            return ("unit", designated)
        target_id = getattr(getattr(missile, "target", None), "id", None)
        return ("unit", target_id) if isinstance(target_id, (int, str)) else None

    def register_weapon_truth_association(self, missile_id: int, target_id: int) -> None:
        """Register evaluator-only collision/attribution identity for a weapon."""
        if missile_id not in self.active_units or target_id not in self.active_units:
            raise ValueError("Weapon truth associations require two registered units.")
        self._weapon_truth_associations[int(missile_id)] = int(target_id)

    def register_sensor_report_truth_association(
        self, sensor_id: object, report_id: int, target_id: int
    ) -> None:
        """Record truth lineage exclusively for evaluator attribution."""
        self._sensor_report_truth_associations[(sensor_id, int(report_id))] = int(target_id)

    def refresh_contact_truth_associations(
        self,
        sensor_id: object,
        report_lineage_by_contact: dict[object, tuple[tuple[object, int], ...]],
    ) -> None:
        """Resolve anonymous contact lineage inside the evaluator boundary."""
        active_keys = {(sensor_id, contact_id) for contact_id in report_lineage_by_contact}
        self._contact_truth_associations = {
            key: value
            for key, value in self._contact_truth_associations.items()
            if key[0] != sensor_id or key in active_keys
        }
        for contact_id, report_lineage in report_lineage_by_contact.items():
            attribution = self._dominant_truth_id(report_lineage)
            if attribution is not None:
                self._contact_truth_associations[(sensor_id, contact_id)] = attribution

    def _dominant_truth_id(self, report_lineage) -> int | None:
        """Resolve which physical unit a track is actually about.

        Requiring every report in the lineage to agree is not usable here: a track's
        lineage only ever grows, so one report that association placed on the wrong
        aircraft -- unavoidable for a close formation, where the reports genuinely are
        statistically compatible -- would suppress the track's attribution for the rest
        of the episode, and with it every hit check for weapons fired at it.

        Instead take the aircraft that contributed most of the *recent* evidence.
        Report ids increase monotonically per source, so the newest reports per source
        are the tail of that source's lineage; older reports are dropped so a track
        that has genuinely re-associated onto another aircraft follows it. A tie is
        real ambiguity and yields no attribution.
        """
        recent_by_source: dict[object, list[int]] = {}
        for source_id, report_id in report_lineage:
            recent_by_source.setdefault(source_id, []).append(int(report_id))
        votes: Counter[int] = Counter()
        for source_id, report_ids in recent_by_source.items():
            for report_id in sorted(report_ids)[-_ATTRIBUTION_WINDOW_PER_SOURCE:]:
                truth_id = self._sensor_report_truth_associations.get((source_id, report_id))
                if truth_id is not None:
                    votes[truth_id] += 1
        if not votes:
            return None
        ranked = votes.most_common(2)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def evaluator_truth_id_for_contact(self, sensor_id: object, contact_id: object) -> int | None:
        """Return evaluator attribution without exposing it through a radar object."""
        return self._contact_truth_associations.get((sensor_id, contact_id))

    def evaluator_contact_ids_for_truth(self, sensor_id: object, target_id: object) -> set[object]:
        """Reverse evaluator attribution for legacy target-object adapters only."""
        return {
            contact_id
            for (source_id, contact_id), truth_id in self._contact_truth_associations.items()
            if source_id == sensor_id and truth_id == target_id
        }

    def register_weapon_contact_association(
        self,
        missile_id: int,
        sensor_id: object,
        contact_id: object,
        report_lineage: object = (),
    ) -> None:
        """Record what a weapon was launched against, by lineage and by contact.

        Truth attribution can be unavailable at launch and become available later,
        once the aircraft separate enough for the picture to tell them apart. Two
        identities are kept because they fail differently:

        ``report_lineage`` is the immutable evidence the launch track was built from.
        It never changes and stays resolvable for the whole episode, so it still
        identifies the engaged aircraft after the contact that carried it has been
        retired and re-issued under a new number.

        ``(sensor_id, contact_id)`` is the live contact, which resolves cases where
        the launch lineage names no report the evaluator can attribute -- but it is
        ephemeral, and a re-issued contact id silently points somewhere else.
        """
        key = int(missile_id)
        self._weapon_contact_associations[key] = (sensor_id, contact_id)
        self._weapon_launch_lineage[key] = tuple(
            (source_id, int(report_id)) for source_id, report_id in report_lineage or ()
        )

    def evaluator_target_for_weapon(self, missile):
        """Resolve a weapon target for evaluator physics, never operational guidance."""
        missile_id = getattr(missile, "id", None)
        target_id = self._weapon_truth_associations.get(missile_id)
        if target_id is None:
            weapon_track = getattr(missile, "weapon_track", None)
            snapshot = getattr(weapon_track, "snapshot", None)
            current_lineage = getattr(snapshot, "report_lineage", ())
            if weapon_track is not None:
                # A report majority is not a safe terminal identity for a fused
                # formation track: it can name a different aircraft from the
                # operational hypothesis the missile is actually following. Resolve
                # every sensor-limited weapon geometrically until it is close enough
                # to commit, not only the exact-tie cases.
                target_id = self._resolve_weapon_truth_by_geometry(missile, current_lineage)
            else:
                target_id = self._resolve_weapon_truth(missile_id, current_lineage)
        return self.active_units.get(target_id)

    def _resolve_weapon_truth_by_geometry(self, missile, current_lineage=()) -> int | None:
        """Attribute an ambiguous-lineage weapon to one aircraft by geometry.

        Last resort after ``_resolve_weapon_truth`` cannot disambiguate the launch
        track. Candidates are the aircraft that actually contributed to the launch /
        current lineage (so the shot is only ever scored against an aircraft it was
        genuinely built from); when no lineage attribution is available at all we
        widen to every engageable opponent. Among those we pick the one nearest the
        weapon's own guidance estimate — i.e. where the shooter aimed — falling back
        to the missile body when no estimate exists.

        The result remains provisional outside terminal range. It is latched into
        ``_weapon_truth_associations`` only when the missile body is close to the
        selected aircraft, so a noisy midcourse estimate cannot permanently bind the
        shot to the wrong member. Evaluator only: guidance never receives the selected
        Unit.
        """
        missile_id = getattr(missile, "id", None)
        if missile_id is None:
            return None

        group = getattr(missile, "group", None)

        def _selectable(unit) -> bool:
            if unit is missile or getattr(unit, "is_missile", False):
                return False
            if getattr(unit, "is_non_engageable", False) or getattr(unit, "is_destroyed", False):
                return False
            # Never resolve a weapon onto a friendly aircraft.
            return group is None or getattr(unit, "group", None) != group

        candidate_ids = self._lineage_truth_candidates(missile_id, current_lineage)
        if candidate_ids:
            candidates = [
                self.active_units[cid] for cid in candidate_ids if cid in self.active_units
            ]
            candidates = [unit for unit in candidates if _selectable(unit)]
        else:
            candidates = [unit for unit in self.active_units.values() if _selectable(unit)]
        if not candidates:
            return None

        reference = self._weapon_guidance_estimate(missile)

        def _distance_km(unit) -> float:
            try:
                if reference is not None:
                    return geodetic_distance_km(
                        reference.lat,
                        reference.lon,
                        reference.alt,
                        unit.position.lat,
                        unit.position.lon,
                        unit.position.alt,
                    )
                return units_distance_km(missile, unit)
            except Exception:  # noqa: BLE001 - a malformed unit must not abort resolution
                return float("inf")

        ranked = min(
            ((_distance_km(unit), getattr(unit, "id", 0), unit) for unit in candidates),
            key=lambda item: (item[0], item[1]),
        )
        _guidance_to_chosen_km, _, chosen = ranked
        chosen_id = getattr(chosen, "id", None)
        if chosen_id is None:
            return None
        # Commit (latch) only once the weapon is close to the aircraft it is prosecuting.
        # Latching earlier could freeze the shot onto the wrong member of a formation
        # that guidance has not yet separated; leaving it transient lets the choice
        # follow the weapon's converging guidance estimate until terminal. Once latched
        # the weapon is scored against the same aircraft for the rest of its flight.
        try:
            missile_to_chosen_km = units_distance_km(missile, chosen)
        except Exception:  # noqa: BLE001
            missile_to_chosen_km = float("inf")
        if missile_to_chosen_km <= _GEOMETRIC_ATTRIBUTION_COMMIT_KM:
            self._weapon_truth_associations[int(missile_id)] = int(chosen_id)
        return int(chosen_id)

    def _lineage_truth_candidates(self, missile_id, current_lineage=()) -> set[int]:
        """Truth ids of every aircraft that contributed to a weapon's lineage.

        Unlike ``_dominant_truth_id`` this does not vote for a single winner; it
        returns the whole candidate set (launch lineage ∪ current track lineage) so
        the geometric tie-break can choose among exactly the aircraft the track was
        actually built from.
        """
        ids: set[int] = set()
        lineages = (
            self._weapon_launch_lineage.get(missile_id, ()),
            tuple(current_lineage or ()),
        )
        for lineage in lineages:
            for source_id, report_id in lineage:
                truth_id = self._sensor_report_truth_associations.get((source_id, int(report_id)))
                if truth_id is not None:
                    ids.add(int(truth_id))
        return ids

    @staticmethod
    def _weapon_guidance_estimate(missile):
        """The weapon's own estimated target position (operational state), or None.

        Reading the missile's guidance estimate does not breach the information
        firewall — it is the evaluator inspecting operational state, not guidance
        receiving truth. Used only to rank truth candidates the evaluator already
        holds.
        """
        provider = getattr(missile, "target_provider", None)
        getter = getattr(provider, "get_guidance_target", None)
        if callable(getter):
            try:
                estimate = getter()
            except Exception:  # noqa: BLE001 - guidance estimate is best-effort
                estimate = None
            if estimate is not None:
                return estimate
        return getattr(missile, "position", None)

    def evaluator_weapon_target_departed(self, missile) -> bool:
        """Return whether a latched weapon target has left the active roster."""
        target_id = self._weapon_truth_associations.get(getattr(missile, "id", None))
        return target_id is not None and target_id not in self.active_units

    def _resolve_weapon_truth(self, missile_id, current_lineage=()) -> int | None:
        """Late-resolve a weapon that could not be attributed at launch."""
        target_id = self._dominant_truth_id(self._weapon_launch_lineage.get(missile_id, ()))
        if target_id is None:
            # A launch track can be genuinely ambiguous and later be retired. Its
            # immutable lineage remains tied forever, while the missile's continuing
            # hypothesis acquires fresh reports under a replacement track ID. The
            # current WeaponTrack lineage is therefore the only evidence that can
            # resolve that shot without coupling operational guidance to truth.
            target_id = self._dominant_truth_id(current_lineage)
        if target_id is None:
            contact_key = self._weapon_contact_associations.get(missile_id)
            if contact_key is not None:
                target_id = self._contact_truth_associations.get(contact_key)
        if target_id is None or target_id not in self.active_units:
            return None
        # Attribution is settled once resolved: a weapon already in flight must not
        # change which aircraft it is scored against if the picture drifts later.
        self._weapon_truth_associations[int(missile_id)] = int(target_id)
        return int(target_id)

    def record_diagnostic(self, name: str) -> None:
        """Increment a diagnostic without exposing evaluator state operationally."""
        self.diagnostic_counters[str(name)] += 1

    def _remove_countermeasures_for_parent(self, parent_unit):
        """Remove all countermeasures deployed by the given parent unit."""
        if not parent_unit:
            return
        countermeasures_to_remove = [
            uid
            for uid, unit in self.active_units.items()
            if getattr(unit, "is_countermeasure", False)
            and hasattr(unit, "parent_aircraft")
            and unit.parent_aircraft == parent_unit
        ]
        for cm_id in countermeasures_to_remove:
            self.remove_unit(cm_id)

    def get_unit(self, unit_id: int) -> Unit:
        return self.active_units.get(unit_id)

    def unit_exists(self, unit_id: int) -> bool:
        return unit_id in self.active_units

    def record_unit_trace(self, unit_id: int):
        # No-op when trace recording is disabled (headless training): the trace is
        # only consumed by visualization/tacview, so nothing is stored and the
        # per-tick storage loop stays empty.
        if not self.record_traces:
            return
        if unit_id not in self.active_units:
            raise Exception(f"Unit.record_unit_trace(): unknown unit {unit_id}")
        if unit_id not in self.trace_record_units:
            self.trace_record_units[unit_id] = []
            self._store_unit_state(unit_id)

    def set_status_text(self, text: str):
        self.status_text = text

    def add_tick_callback(self, cb_fn: Callable[[datetime], None]):
        self._tick_callbacks.append(cb_fn)

    def do_tick(self) -> list[Event]:
        from bvr_marl_core.aircraft.core.nez import clear_dlz_cache
        from bvr_marl_core.simulator.utils.geodesics import clear_bearing_cache

        clear_bearing_cache()
        clear_dlz_cache()

        tick_secs = self.tick_secs
        events = []

        self._sample_datalink_links()

        # Freeze roster membership for this tick. New entities are visible from
        # the next tick, and stable ID order prevents dictionary insertion order
        # from changing controller/update order.
        tick_roster = tuple(
            sorted(
                self.active_units.values(),
                key=lambda unit: (bool(getattr(unit, "is_missile", False)), str(unit.id)),
            )
        )
        self.substepper = self.terminal_resolver  # compatibility alias
        self.ccd.begin_tick_snapshot(tick_roster)

        state_buffer = TickStateBuffer(tick_roster)
        self._tick_state_buffer = state_buffer
        self._tick_in_progress = True
        try:
            # Global sensor stage 1: every aircraft generates immutable raw
            # reports from the same authoritative state-t roster.
            for unit in tick_roster:
                if unit.id not in self.active_units:
                    continue
                stage_reports = getattr(unit, "stage_sensor_reports", None)
                if callable(stage_reports):
                    state_buffer.restore_start()
                    stage_reports(tick_secs, self)

            # Global sensor stage 1.5: fuse+track each datalink team's net-entered
            # reports ONCE into a shared Link-16 network picture, before any platform
            # consumes it. This replaces N independent cross-platform fusions with one
            # per team (see radar.core.network_picture).
            from bvr_marl_core.radar.core.network_picture import update_network_pictures

            update_network_pictures(self, tick_secs)

            # Global sensor stage 2: datalink/fusion/tracking may now consume only
            # the complete frozen report set, never a peer's partially updated
            # tracker object.
            for unit in tick_roster:
                if unit.id not in self.active_units:
                    continue
                stage_products = getattr(unit, "update_staged_sensor_products", None)
                if callable(stage_products):
                    state_buffer.restore_start()
                    stage_products(tick_secs, self)

            # Global dynamics stage: integrate each next state independently,
            # then publish all next states together below.
            for unit in tick_roster:
                if unit.id not in self.active_units:
                    continue
                state_buffer.restore_start()
                staged_update = getattr(unit, "update_after_staged_sensors", None)
                unit_events = (
                    staged_update(tick_secs, self)
                    if callable(staged_update)
                    else unit.update(tick_secs, self)
                )
                events.extend(unit_events)
                state_buffer.capture_next(unit)
        finally:
            state_buffer.publish()
            self._tick_in_progress = False
            self._tick_state_buffer = None
            self._publish_pending_units()

        substep_events = self.terminal_resolver.run_tick_with_substeps(self, tick_secs)
        events.extend(substep_events)

        units_to_remove = [
            uid
            for uid, unit in self.active_units.items()
            if getattr(unit, "should_be_removed", False)
        ]
        for unit_id in units_to_remove:
            unit = self.remove_unit(unit_id)
            if unit:
                removal_event = UnitRemovedEvent(
                    self, unit, getattr(unit, "removal_reason", "unknown")
                )
                events.append(removal_event)
                self._remove_countermeasures_for_parent(unit)

        self.utc_time += timedelta(seconds=tick_secs)

        for unit_id in self.trace_record_units.keys():
            self._store_unit_state(unit_id)

        for fn in self._tick_callbacks:
            fn(self.utc_time)

        self.events.extend(events)
        self._tick_index += 1
        return events

    def staged_next_position(self, unit):
        """Return an already-integrated same-tick pose without publishing it."""
        buffer = getattr(self, "_tick_state_buffer", None)
        return buffer.next_position(unit) if buffer is not None else None

    def _sample_datalink_links(self) -> None:
        """Sample one sender-receiver link state matrix for the entire tick."""
        radar_units = sorted(
            (
                unit
                for unit in self.active_units.values()
                if getattr(unit, "radar", None) is not None
                and getattr(getattr(unit.radar, "data_link", None), "get_mode", lambda: "none")()
                != "none"
            ),
            key=lambda unit: str(unit.id),
        )
        probability = min(1.0, max(0.0, float(self.datalink_drop_prob)))
        states: dict[tuple[int, int], bool] = {}
        for sender in radar_units:
            for receiver in radar_units:
                if sender.id == receiver.id or sender.group != receiver.group:
                    continue
                rng = self.random_streams.generator("datalink_link", f"{sender.id}->{receiver.id}")
                states[(sender.id, receiver.id)] = bool(rng.random() >= probability)
        self._datalink_link_states = states
        self.datalink_link_history.append(
            {
                "tick": self._tick_index,
                "states": dict(states),
            }
        )

    def is_datalink_up(self, sender_id: int, receiver_id: int) -> bool:
        """Return the already-sampled link state for the current tick."""
        if sender_id == receiver_id:
            return True
        if (sender_id, receiver_id) not in self._datalink_link_states:
            return float(self.datalink_drop_prob) <= 0.0
        return self._datalink_link_states[(sender_id, receiver_id)]

    def log_event(self, event: Event):
        self.events.append(event)

    def _store_unit_state(self, unit_id):
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append(
                (self.utc_time, unit.position.copy(), unit.yaw_deg, unit.speed)
            )
