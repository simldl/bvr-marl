from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np

from bvr_marl_core.simulator.core.events import Event, UnitRegisteredEvent, UnitRemovedEvent
from bvr_marl_core.simulator.core.hit_calculation import HitParams, MissileHitCalculator
from bvr_marl_core.simulator.core.hit_event_helpers import (
    CCDConfig,
    MissileCCDManager,
    stochastic_on_hit,
)
from bvr_marl_core.simulator.core.substepping import SubstepConfig, Substepper
from bvr_marl_core.simulator.core.units import Unit
from bvr_marl_core.simulator.utils.angles import deg2rad


class Simulator:
    def __init__(
        self,
        utc_time=datetime.now(),
        tick_secs=1,
        random_seed=None,
        num_units=0,
        num_opp_units=0,
        weapon_config=None,
    ):
        self.active_units: dict[int, Unit] = {}
        self.trace_record_units = {}
        self.events: list[Event] = []
        self._missile_target_counts: dict[str, dict[int, int]] = {}
        self.utc_time = utc_time
        self.utc_time_initial = utc_time
        self.tick_secs = tick_secs
        self._tick_callbacks: list[Callable[[datetime], None]] = []
        self.random_seed = random_seed
        self.rnd_gen = random.Random(random_seed)
        self._next_unit_id = 1
        self.status_text = None

        self.num_units = num_units
        self.num_opp_units = num_opp_units

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

        self.hit_calc = MissileHitCalculator(
            HitParams(
                fuse_radius_m=missile_hit_radius,
                target_radius_m=0.0,
                max_consider_range_m=50_000.0,
            )
        )

        self.substepper = Substepper(
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

    def reset_sim(self, units):
        """
        Reset the simulator to its initial state with a new set of units.
        :param units: A dictionary of units to initialize in the simulator.
        """
        self.utc_time = datetime.now()
        self.utc_time_initial = self.utc_time
        self._tick_callbacks = []
        self.status_text = None
        self.active_units = units
        self.trace_record_units = {}
        self._next_unit_id = 1
        self._missile_target_counts = {}
        self.events = []
        if hasattr(self, "ccd") and hasattr(self.ccd, "reset_engagement_tracking"):
            self.ccd.reset_engagement_tracking()
        for unit_id, unit in units.items():
            self.record_unit_trace(unit_id)
        print("Simulator reset.")

    def add_unit(self, unit: Unit) -> int:
        if unit.id is not None and unit.id in self.active_units:
            return unit.id

        if hasattr(unit, "target") and unit.target is not None:
            if unit.target.id is None or unit.target.id not in self.active_units:
                raise ValueError(
                    f"Target unit {unit.target} must be registered before adding unit {unit}."
                )
        unit.id = self._next_unit_id
        self.active_units[unit.id] = unit
        self.trace_record_units[unit.id] = []
        self._store_unit_state(unit.id)
        self.log_event(UnitRegisteredEvent(self, unit))
        # Reseed any attached radar's numpy RNG now that the unit has a stable ID.
        # Aircraft construct their radar before add_unit assigns an ID, so the radar's
        # np_rng was seeded with None (OS entropy).  Reseeding here makes detection
        # outcomes reproducible across Python processes for a given random_seed.
        radar = getattr(unit, "radar", None)
        if radar is not None and hasattr(radar, "np_rng"):
            rng = np.random.default_rng(unit.id)
            radar.np_rng = rng
            if hasattr(radar, "obsgen") and radar.obsgen is not None:
                radar.obsgen.np_rng = rng
        self._next_unit_id += 1
        if getattr(unit, "is_missile", False):
            group = getattr(unit, "group", None)
            target_id = getattr(getattr(unit, "target", None), "id", None)
            if group is not None and target_id is not None:
                grp_map = self._missile_target_counts.setdefault(group, {})
                grp_map[target_id] = grp_map.get(target_id, 0) + 1
        return unit.id

    def remove_unit(self, unit_id: int):
        if unit_id in self.active_units:
            unit = self.active_units[unit_id]
            if getattr(unit, "is_missile", False):
                group = getattr(unit, "group", None)
                target_id = getattr(getattr(unit, "target", None), "id", None)
                if group is not None and target_id is not None:
                    grp_map = self._missile_target_counts.get(group, {})
                    if grp_map.get(target_id, 0) > 1:
                        grp_map[target_id] -= 1
                    elif target_id in grp_map:
                        del grp_map[target_id]
            del self.active_units[unit_id]
            return unit
        return None

    def count_missiles_at_target(self, group: str, target_id: int) -> int:
        """Return number of active missiles from *group* currently targeting *target_id*."""
        return self._missile_target_counts.get(group, {}).get(target_id, 0)

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

        self.ccd.begin_tick_snapshot(self.active_units.values())

        for unit in list(self.active_units.values()):
            unit_events = unit.update(self.tick_secs, self)
            events.extend(unit_events)

        substep_events = self.substepper.run_tick_with_substeps(self, tick_secs)
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
        return events

    def log_event(self, event: Event):
        self.events.append(event)

    def _store_unit_state(self, unit_id):
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append(
                (self.utc_time, unit.position.copy(), unit.yaw_deg, unit.speed)
            )
