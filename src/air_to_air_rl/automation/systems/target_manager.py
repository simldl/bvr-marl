"""
Target selection and prioritization system for the semi-automated helper.
"""

from typing import Any, Optional

import numpy as np

from air_to_air_rl.simulator.utils.angles import signed_yaw_deg_diff
from air_to_air_rl.simulator.utils.geodesics import geodetic_bearing_deg, geodetic_distance_km


class TargetManager:
    """Manages target selection and prioritization based on tactical considerations."""

    def __init__(self, aircraft, config):
        self.aircraft = aircraft
        self.config = config
        self.current_target = None
        self.target_lock_time = {}  # Track how long we've been locked on each target
        self.last_targets = []  # Previous target candidates for comparison

    def select_optimal_target(self, simulator, threats: list[Any]) -> float:
        """
        Select the optimal target based on tactical priorities.

        Uses the aircraft's sensor system TrackPrioritySystem when available
        to ensure consistency with the aircraft's targeting logic.

        Args:
            simulator: The simulation environment
            threats: list of assessed threats

        Returns:
            Target selection action value (0.0-1.0)
        """
        # Priority 1: Use aircraft's sensor track prioritization if available
        if hasattr(self.aircraft, "sensor") and hasattr(self.aircraft.sensor, "prioritized_tracks"):
            prioritized_tracks = self.aircraft.sensor.get_prio_tracks()

            if prioritized_tracks:
                # Get target candidates
                target_candidates = self._get_target_candidates(simulator)

                if not target_candidates:
                    return 0.0

                # Map prioritized tracks to actual target units
                # The track prioritizer returns (state_vec, score) tuples
                # We need to match these to actual units
                best_target = self._match_track_to_target(prioritized_tracks[0], target_candidates)

                if best_target:
                    self.current_target = best_target

                    # Use aircraft's sensor.select_target for consistency
                    # This returns the selection value directly
                    target_index = target_candidates.index(best_target)
                    normalized_selection = target_index / max(len(target_candidates) - 1, 1)

                    return normalized_selection

        # Fallback: Use local scoring if sensor prioritization not available
        target_candidates = self._get_target_candidates(simulator)

        if not target_candidates:
            return 0.0

        # Score each target
        target_scores = []
        for target in target_candidates:
            score = self._score_target(target, threats)
            target_scores.append((target, score))

        # Sort by score (highest first)
        target_scores.sort(key=lambda x: x[1], reverse=True)

        # Select best target if score is above threshold
        if target_scores and target_scores[0][1] > 0.3:  # Minimum engagement threshold
            best_target = target_scores[0][0]
            self.current_target = best_target

            # Convert to action space value (0.0-1.0)
            target_index = target_candidates.index(best_target)
            normalized_selection = target_index / max(len(target_candidates) - 1, 1)

            return normalized_selection

        return 0.0

    def _match_track_to_target(self, prioritized_track: tuple, candidates: list[Any]) -> Any:
        """
        Match a prioritized track (state vector) to an actual target unit.

        Args:
            prioritized_track: (state_vec, score) tuple from TrackPrioritySystem
            candidates: list of candidate target units

        Returns:
            Matched target unit or None
        """
        if not prioritized_track or len(prioritized_track) < 2:
            return None

        track_state = np.array(prioritized_track[0][:3])  # position [x, y, z]

        # Find the candidate closest to this track position
        min_distance = float("inf")
        best_match = None

        for candidate in candidates:
            try:
                # Convert candidate position to relative coordinates (approximate)
                # The track state is in relative ENU coordinates from aircraft
                candidate_rel = np.array(
                    [
                        (candidate.position.lat - self.aircraft.position.lat) * 111000,  # approx m
                        (candidate.position.lon - self.aircraft.position.lon) * 111000,
                        -(
                            candidate.position.alt - self.aircraft.position.alt
                        ),  # negative because track uses -z
                    ]
                )

                distance = np.linalg.norm(track_state - candidate_rel)

                if distance < min_distance:
                    min_distance = distance
                    best_match = candidate
            except Exception:
                continue

        # Only return match if reasonably close (within 5km)
        if min_distance < 5000:
            return best_match
        return None

    def _get_target_candidates(self, simulator) -> list[Any]:
        """Get all valid target candidates."""
        candidates = []
        for unit in simulator.active_units.values():
            if (
                unit.group != self.aircraft.group
                and not getattr(unit, "is_missile", False)
                and self._is_targetable(unit)
            ):
                candidates.append(unit)

        return candidates

    def _is_targetable(self, target) -> bool:
        """Check if a target is valid for engagement."""
        distance_m = (
            geodetic_distance_km(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                self.aircraft.position.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            * 1000.0
        )

        # Prefer sensor truth when available
        if hasattr(self.aircraft, "sensor"):
            detected = getattr(self.aircraft.sensor, "detected_targets", {})
            if target.id in detected:
                return True  # positively detected

        # Otherwise, fall back to platform capability (radar range if known)
        radar_range = 0
        if hasattr(self.aircraft, "radar") and hasattr(self.aircraft.radar, "max_range_m"):
            radar_range = max(radar_range, getattr(self.aircraft.radar, "max_range_m", 0))

        if radar_range > 0:
            return distance_m <= radar_range

        # Final conservative fallback (no sensor/radar info): allow within 80 km
        return distance_m <= 80_000.0

    def _score_target(self, target, threats: list[Any]) -> float:
        """
        Score a target based on multiple tactical factors.

        Args:
            target: Target unit to score
            threats: Current threat assessment

        Returns:
            Score from 0.0 (avoid) to 1.0 (highest priority)
        """
        score = 0.0
        weights = self.config.target_priority_weights

        # Factor 1: Range (closer is generally better for engagement)
        range_score = self._calculate_range_score(target)
        score += weights["range"] * range_score

        # Factor 2: Threat level (more dangerous targets prioritized)
        threat_score = self._calculate_threat_score(target, threats)
        score += weights["threat_level"] * threat_score

        # Factor 3: Angle/geometry (targets in good firing position)
        angle_score = self._calculate_angle_score(target)
        score += weights["angle"] * angle_score

        # Factor 4: Lock quality (how good our sensor lock is)
        lock_score = self._calculate_lock_quality_score(target)
        score += weights["lock_quality"] * lock_score

        # Apply automation level modifiers
        score = self._apply_automation_level_modifier(score, target, threats)

        return np.clip(score, 0.0, 1.0)

    def _calculate_range_score(self, target) -> float:
        distance_m = (
            geodetic_distance_km(
                self.aircraft.position.lat,
                self.aircraft.position.lon,
                self.aircraft.position.alt,
                target.position.lat,
                target.position.lon,
                target.position.alt,
            )
            * 1000.0
        )

        # Prefer the aircraft's WEZ/DLZ model if available
        if hasattr(self.aircraft, "wez") and callable(
            getattr(self.aircraft.wez, "compute_dlz", None)
        ):
            try:
                dlz = self.aircraft.wez.compute_dlz(target)
                if dlz.r_min_m <= distance_m <= dlz.r_aero_m:
                    return 1.0
                if distance_m < dlz.r_min_m:
                    return 0.5  # too close
                return max(0.0, 1.0 - (distance_m - dlz.r_aero_m) / max(dlz.r_aero_m, 1.0))
            except Exception:
                pass

        # Fallback: prefer medium-long BVR (15–60 km), taper outside
        if distance_m <= 80_000.0:
            if 15_000.0 <= distance_m <= 60_000.0:
                return 1.0
            return max(0.2, 0.7 - abs(distance_m - 37_500.0) / 80_000.0 * 0.5)
        return max(0.05, 1.0 - (distance_m - 80_000.0) / 40_000.0)

    def _calculate_threat_score(self, target, threats: list[Any]) -> float:
        """Calculate score based on how threatening the target is."""
        threat_score = 0.5  # Default moderate threat

        # Check if this target is in our threat list
        for threat in threats:
            if getattr(threat, "source", None) == target:
                threat_level = getattr(threat, "threat_level", "medium")
                if threat_level == "high":
                    threat_score = 1.0
                elif threat_level == "medium":
                    threat_score = 0.7
                elif threat_level == "low":
                    threat_score = 0.3
                break

        # Factor in target capabilities if known
        if hasattr(target, "remaining_missiles"):
            missile_threat = min(target.remaining_missiles / 4.0, 1.0)  # Normalize
            threat_score = max(threat_score, missile_threat)

        return threat_score

    def _calculate_angle_score(self, target) -> float:
        """Calculate score based on engagement geometry."""
        # Calculate bearing to target
        target_bearing = geodetic_bearing_deg(
            self.aircraft.position.lat,
            self.aircraft.position.lon,
            target.position.lat,
            target.position.lon,
        )

        # How far off our nose is the target?
        angle_off_nose = abs(signed_yaw_deg_diff(self.aircraft.yaw_deg, target_bearing))

        # Prefer targets more directly ahead (better for engagement)
        if angle_off_nose <= 30:
            return 1.0
        elif angle_off_nose <= 60:
            return 0.7
        elif angle_off_nose <= 90:
            return 0.4
        else:
            return 0.1  # Targets behind us are low priority

    def _calculate_lock_quality_score(self, target) -> float:
        """Calculate score based on sensor lock quality."""
        if not hasattr(self.aircraft, "sensor"):
            return 0.5

        # Check if we have a radar lock
        if hasattr(self.aircraft.sensor, "has_radar_lock"):
            if self.aircraft.sensor.has_radar_lock(target):
                return 1.0

        # Check detection quality
        detected_targets = getattr(self.aircraft.sensor, "detected_targets", {})
        if target.id in detected_targets:
            detection_data = detected_targets[target.id]
            # If we have velocity data, it's a better lock
            if hasattr(detection_data, "velocity") and detection_data.velocity is not None:
                return 0.8
            else:
                return 0.6

        return 0.2  # Poor or no lock

    def _apply_automation_level_modifier(
        self, base_score: float, target, threats: list[Any]
    ) -> float:
        """Apply modifiers based on automation level setting."""
        level = self.config.automation_level

        if level == "defensive":
            # Defensive mode: prioritize immediate threats, avoid engagement unless necessary
            immediate_threat = any(
                getattr(t, "distance_m", float("inf")) < 10000
                for t in threats
                if getattr(t, "source", None) == target
            )
            if immediate_threat:
                return base_score * 1.2
            else:
                return base_score * 0.8

        elif level == "aggressive":
            # Aggressive mode: more likely to engage, higher scores overall
            return base_score * 1.1

        else:  # balanced
            return base_score

    def update_config(self, config):
        """Update configuration."""
        self.config = config
