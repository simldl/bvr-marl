"""Physics calculation modules for action processing.

These classes have moved to bvr_marl_core.physics.constraints so that any
controller (scripted BT, neural, etc.) can use them without importing the
RL stack.  This module re-exports them for backwards compatibility.
"""

from bvr_marl_core.physics.constraints import DragCalculator, EnergyCalculator, EnvelopeCalculator

__all__ = ["EnvelopeCalculator", "DragCalculator", "EnergyCalculator"]
