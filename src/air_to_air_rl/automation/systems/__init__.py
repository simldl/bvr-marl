"""Automation subsystems."""

from air_to_air_rl.automation.systems.countermeasure_controller import CountermeasureController
from air_to_air_rl.automation.systems.target_manager import TargetManager
from air_to_air_rl.automation.systems.threat_assessment import ThreatAssessment

__all__ = ["ThreatAssessment", "TargetManager", "CountermeasureController"]
