"""
Observation Space Builder - Backward compatibility layer.

This module now imports from the new modular structure.
The ObservationBuilder class is re-exported from observation module.
"""
from .observation import ObservationBuilder

__all__ = ['ObservationBuilder']
