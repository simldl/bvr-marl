"""Flight envelope and energy constraint calculators.

Pure-physics helpers with no ML/RL dependencies.  These are used by an
extension package's action processor but live here so any controller
(scripted BT, neural, etc.) can import them without pulling in the RL stack.
"""

from bvr_marl_core.physics.constraints.drag_calculator import DragCalculator
from bvr_marl_core.physics.constraints.energy_calculator import EnergyCalculator
from bvr_marl_core.physics.constraints.envelope_calculator import EnvelopeCalculator

__all__ = ["EnvelopeCalculator", "DragCalculator", "EnergyCalculator"]
