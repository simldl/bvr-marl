"""Tactical modules for air-to-air combat."""

from air_to_air_rl.automation.scripted_control.tactics.bvr_tactics import BVRTactics
from air_to_air_rl.automation.scripted_control.tactics.energy_management import EnergyManager
from air_to_air_rl.automation.scripted_control.tactics.geometry_calculator import GeometryCalculator
from air_to_air_rl.automation.scripted_control.tactics.range_calculator import RangeCalculator

__all__ = ["BVRTactics", "EnergyManager", "GeometryCalculator", "RangeCalculator"]
