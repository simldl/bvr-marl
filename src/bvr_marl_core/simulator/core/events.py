from abc import ABC


class Event(ABC):
    def __init__(self, name: str, origin):
        self.name = name
        self.origin = origin
        allocator = getattr(origin, "allocate_event_id", None)
        self.event_id = allocator() if callable(allocator) else None
        try:
            self.time_s = float(origin.elapsed_time_s)
        except (AttributeError, TypeError, ValueError):
            self.time_s = None

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} origin={getattr(self.origin, 'id', None)}>"


class UnitRegisteredEvent(Event):
    def __init__(self, origin, registered_unit):
        super().__init__("UnitRegistered", origin)
        self.registered_unit = registered_unit


class UnitDestroyedEvent(Event):
    def __init__(self, origin, unit_killer, unit_destroyed, cause: str | None = None):
        super().__init__("UnitDestroyed", origin)
        self.unit_killer = unit_killer
        self.unit_destroyed = unit_destroyed
        self.cause = cause or (
            "missile" if getattr(unit_killer, "is_missile", False) else "unknown"
        )


class UnitRemovedEvent(Event):
    def __init__(self, origin, removed_unit, reason: str):
        super().__init__("UnitRemoved", origin)
        self.removed_unit = removed_unit
        self.reason = reason


class MissileEngagementEvent(Event):
    def __init__(self, origin, missile, target):
        super().__init__("MissileEngagement", origin)
        self.missile = missile
        self.target = target
        self.cause = "terminal_gate_entry"


class MissileEnteredTerminalRegionEvent(Event):
    def __init__(self, origin, missile, target, range_m: float):
        super().__init__("MissileEnteredTerminalRegion", origin)
        self.missile = missile
        self.target = target
        self.range_m = float(range_m)


class MissileFuzeTriggeredEvent(Event):
    def __init__(self, origin, missile, target, miss_distance_m: float | None):
        super().__init__("MissileFuzeTriggered", origin)
        self.missile = missile
        self.target = target
        self.miss_distance_m = None if miss_distance_m is None else float(miss_distance_m)


class MissileDetonatedEvent(Event):
    def __init__(self, origin, missile, target, record: dict):
        super().__init__("MissileDetonated", origin)
        self.missile = missile
        self.target = target
        self.record = record


class AircraftMortallyHitEvent(Event):
    def __init__(self, origin, aircraft, weapon, scheduled_destruction_time_s: float):
        super().__init__("AircraftMortallyHit", origin)
        self.aircraft = aircraft
        self.weapon = weapon
        self.scheduled_destruction_time_s = float(scheduled_destruction_time_s)


class AircraftDestroyedEvent(UnitDestroyedEvent):
    def __init__(self, origin, unit_killer, aircraft, cause: str | None = None):
        super().__init__(origin, unit_killer, aircraft, cause=cause)
        self.name = "AircraftDestroyed"
        self.aircraft = aircraft


class MissileTerminalEvent(Event):
    """Structured terminal-outcome record for a single missile detonation.

    Emitted once per proximity detonation by the kill resolver
    (``stochastic_on_hit``). Carries a flat ``record`` dict (cheap, no nesting)
    holding launch context, terminal seeker/track state, the closest-approach
    geometry, the kill-probability factorization, the random draw and the final
    kill result — everything needed to attribute an outcome to launch zone,
    guidance, seeker tracking, or the kill model without re-deriving it.
    """

    def __init__(self, origin, missile, target, record: dict):
        super().__init__("MissileTerminal", origin)
        self.missile = missile
        self.target = target
        self.record = record
        self.cause = "proximity_fuze"


class CountermeasureEvent(Event):
    def __init__(self, origin, countermeasure_type: str, unit_affected):
        super().__init__("Countermeasure", origin)
        self.countermeasure_type = countermeasure_type
        self.unit_affected = unit_affected


class UnitTraceEvent(Event):
    def __init__(self, origin, unit, state_dict: dict):
        super().__init__("UnitTrace", origin)
        self.unit = unit
        self.state_dict = state_dict


class CustomSimEvent(Event):
    def __init__(self, name: str, origin, payload: dict | None = None):
        super().__init__(name, origin)
        self.payload = payload or {}
