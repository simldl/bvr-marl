"""
Extension interfaces for bvr_marl_core.

Defines Protocol-based contracts that extension and RL packages must satisfy
when plugging into the core simulation loop.  Concrete implementations live
in extension packages; only the abstract surface is anchored here.

Available protocols
-------------------
Controller
    Any agent (scripted BT or neural network) that produces actions.
RewardFunction
    Computes scalar rewards from simulation state.
ControllerFactory
    Creates Controller instances from configuration.
ObservationHook
    Optional post-processing hook for observation augmentation.
SensorPlugin
    Any sensor that can be attached to a platform.
WeaponPlugin
    Weapon system that can be attached to a platform.
FlightModelPlugin
    Flight dynamics model plugin.
VisualizerPlugin
    Decoupled rendering/visualization plugin.
ProtocolAdapter
    External protocol/interop adapter (DIS, HLA, Link-16, gRPC, …).
FMUPlugin
    Functional Mock-up Unit integration.
"""

from bvr_marl_core.interfaces.controller import Controller, ScriptedController
from bvr_marl_core.interfaces.factory import ControllerFactory
from bvr_marl_core.interfaces.flight_model import FlightModelPlugin
from bvr_marl_core.interfaces.fmu import FMUPlugin
from bvr_marl_core.interfaces.observation import ObservationHook
from bvr_marl_core.interfaces.protocol_adapter import ProtocolAdapter
from bvr_marl_core.interfaces.reward import RewardFunction
from bvr_marl_core.interfaces.sensor import SensorPlugin
from bvr_marl_core.interfaces.visualizer import VisualizerPlugin
from bvr_marl_core.interfaces.weapon import WeaponPlugin

__all__ = [
    "Controller",
    "ScriptedController",
    "ControllerFactory",
    "ObservationHook",
    "RewardFunction",
    "SensorPlugin",
    "WeaponPlugin",
    "FlightModelPlugin",
    "VisualizerPlugin",
    "ProtocolAdapter",
    "FMUPlugin",
]
