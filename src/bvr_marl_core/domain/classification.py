"""Versioned probabilistic contact-classification schema."""

from __future__ import annotations

from enum import IntEnum

CLASSIFICATION_SCHEMA_VERSION = 1


class ContactClass(IntEnum):
    FIGHTER = 0
    SUPPORT_AIRCRAFT = 1
    MISSILE = 2
    COUNTERMEASURE = 3
    UNKNOWN = 4


CLASSIFICATION_LABELS = tuple(item.name.lower() for item in ContactClass)


def signature_class_probabilities(target) -> tuple[float, ...]:
    """Generate a fallible class belief from sensor-visible signature proxies."""
    if target is None:
        return (0.05, 0.05, 0.05, 0.05, 0.80)
    if bool(getattr(target, "is_countermeasure", False)):
        return (0.02, 0.01, 0.12, 0.80, 0.05)
    speed = float(getattr(target, "speed", 0.0) or 0.0)
    if bool(getattr(target, "is_missile", False)) or speed >= 700.0:
        return (0.05, 0.01, 0.86, 0.03, 0.05)
    rcs = float(getattr(target, "rcs", 3.0) or 3.0)
    if bool(getattr(target, "is_support_asset", False)) or rcs >= 30.0:
        return (0.12, 0.78, 0.01, 0.01, 0.08)
    return (0.80, 0.10, 0.02, 0.01, 0.07)


def most_likely_class(probabilities) -> str:
    if probabilities is None:
        return ContactClass.UNKNOWN.name.lower()
    values = tuple(float(value) for value in probabilities)
    return ContactClass(max(range(len(values)), key=values.__getitem__)).name.lower()
