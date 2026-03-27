from abc import ABC
from typing import Optional


class Event(ABC):
    def __init__(self, name: str, origin):
        self.name = name
        self.origin = origin

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} origin={getattr(self.origin, 'id', None)}>"


class UnitRegisteredEvent(Event):
    def __init__(self, origin, registered_unit):
        super().__init__("UnitRegistered", origin)
        self.registered_unit = registered_unit


class UnitDestroyedEvent(Event):
    def __init__(self, origin, unit_killer, unit_destroyed):
        super().__init__("UnitDestroyed", origin)
        self.unit_killer = unit_killer
        self.unit_destroyed = unit_destroyed


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
