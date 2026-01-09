from __future__ import annotations
from datetime import datetime, timedelta
from typing import Callable, Dict, List
import random
import math
from simulator.core.events import Event, UnitRegisteredEvent, UnitRemovedEvent
from simulator.core.units import Unit
from simulator.core.helpers import units_distance_km
from simulator.utils.angles import deg2rad
from simulator.core.hit_event_helpers import MissileCCDManager, CCDConfig, stochastic_on_hit
from simulator.core.hit_calculation import MissileHitCalculator, HitParams
from simulator.core.substepping import Substepper, SubstepConfig

class Simulator:
    def __init__(self, utc_time=datetime.now(), tick_secs=1, random_seed=None, num_units=0, num_opp_units=0, weapon_config=None):
        self.active_units: Dict[int, Unit] = {}
        self.trace_record_units = {}
        self.events: List[Event] = []  # Event log storage
        self.utc_time = utc_time
        self.utc_time_initial = utc_time
        self.tick_secs = tick_secs
        self._tick_callbacks: List[Callable[[datetime], None]] = []
        self.random_seed = random_seed
        self.rnd_gen = random.Random(random_seed)
        self._next_unit_id = 1
        self.status_text = None

        self.num_units = num_units
        self.num_opp_units = num_opp_units

        # Extract weapon configuration
        weapon_config = weapon_config or {}
        missile_hit_radius = weapon_config.get("missile_hit_radius_m", 500.0)
        gun_hit_radius = weapon_config.get("gun_hit_radius_m", 5.0)
        
        # Store for gun projectiles
        self.gun_hit_radius_m = gun_hit_radius

        self.ccd = MissileCCDManager(
            CCDConfig(
                fuse_radius_m=missile_hit_radius,
                target_radius_m=0.0,
                max_consider_range_m=50_000.0,
                within_lock_gate_m=10_000.0,  # Keep existing lock gate
            ),
            on_hit=stochastic_on_hit  # Use stochastic hit probability
        )

        self.hit_calc = MissileHitCalculator(HitParams(
            fuse_radius_m=missile_hit_radius,
            target_radius_m=0.0,
            max_consider_range_m=50_000.0,
        ))

        self.substepper = Substepper(
            config=SubstepConfig(
                engage_distance_km=2,
                min_substeps=1,
                max_substeps=16,
                safety_travel_per_substep_m=250,
                physics_only=False,
            )
        )

        self.events: List[Event] = []

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
        self.events = []  # Clear events list to prevent memory leak
        # also reset CCD 'once per missile' engagement tracking
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
        self._next_unit_id += 1
        return unit.id

    def remove_unit(self, unit_id: int):
        if unit_id in self.active_units:
            unit = self.active_units[unit_id]
            del self.active_units[unit_id]
            return unit
        return None

    def _remove_countermeasures_for_parent(self, parent_unit):
        """
        Remove all countermeasures deployed by the given parent unit.
        
        Args:
            parent_unit: The unit (typically an aircraft) whose countermeasures should be removed
        """
        if not parent_unit:
            return
            
        # Find all countermeasures belonging to this parent
        countermeasures_to_remove = []
        for unit_id, unit in self.active_units.items():
            # Check if this is a countermeasure and if it belongs to the parent
            if getattr(unit, 'is_countermeasure', False):
                # Check if the parent_aircraft matches
                if hasattr(unit, 'parent_aircraft') and unit.parent_aircraft == parent_unit:
                    countermeasures_to_remove.append(unit_id)
        
        # Remove the countermeasures
        for cm_id in countermeasures_to_remove:
            self.remove_unit(cm_id)

    def get_unit(self, unit_id: int) -> Unit:
        return self.active_units.get(unit_id, None)

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

    def do_tick(self) -> List[Event]:
        # OPTIMIZATION: Clear geodetic bearing cache at start of each tick
        from simulator.utils.geodesics import clear_bearing_cache
        clear_bearing_cache()

        tick_secs = self.tick_secs
        events = []

        self.ccd.begin_tick_snapshot(self.active_units.values())

        for unit in list(self.active_units.values()):
            unit_events = unit.update(self.tick_secs, self)
            events.extend(unit_events)

        # CCD post_update disabled - hit detection only via substepping for close-range (<2km)
        # self.ccd.post_update_ccd(self.active_units.values(), self.tick_secs, sim=self)

        substep_events = self.substepper.run_tick_with_substeps(self, tick_secs)
        events.extend(substep_events)

        # Remove units marked for removal (e.g., boundary violations)
        units_to_remove = []
        for unit_id, unit in self.active_units.items():
            if getattr(unit, 'should_be_removed', False):
                units_to_remove.append(unit_id)

        for unit_id in units_to_remove:
            unit = self.remove_unit(unit_id)
            if unit:
                # Log removal event
                removal_event = UnitRemovedEvent(self, unit, getattr(unit, 'removal_reason', 'unknown'))
                events.append(removal_event)  # will be added once via self.events.extend(events) below
                
                # Remove any countermeasures deployed by this unit
                self._remove_countermeasures_for_parent(unit)

        self.utc_time += timedelta(seconds=tick_secs)

        for unit_id in self.trace_record_units.keys():
            self._store_unit_state(unit_id)

        for fn in self._tick_callbacks:
            fn(self.utc_time)

        self.events.extend(events)
        return events


    def log_event(self, event: Event):
        """
        Log a simulation event.
        :param event: Event instance to log.
        """
        self.events.append(event)

    def _store_unit_state(self, unit_id):
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append((self.utc_time, unit.position.copy(), unit.yaw_deg, unit.speed))

    def _run_engagement_ccd(self, tick_secs: float) -> List[Event]:
        out: List[Event] = []
        engaged_pairs = []
        for u in self.active_units.values():
            if getattr(u, "is_missile", False) and getattr(u, "target", None):
                d_km = units_distance_km(u, u.target)
                if d_km <= 10.0:
                    engaged_pairs.append((u, u.target))

        for missile, tgt in engaged_pairs:
            hit, victim, t_frac = self.hit_calc.check_and_maybe_hit(
                missile=missile,
                potential_targets=[tgt],
                tick_secs=tick_secs,
            )
            if hit:
                evt = missile.build_engagement_event(self, victim, t_frac)
                if evt:
                    out.append(evt)
                    self._handle_event(evt)
        return out