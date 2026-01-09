"""
Threat assessment system for the semi-automated helper.
Analyzes the tactical situation and identifies threats to the aircraft.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from simulator.utils.geodesics import geodetic_distance_km, geodetic_bearing_deg
from simulator.utils.angles import signed_yaw_deg_diff


@dataclass
class Threat:
    """Represents a threat to the aircraft."""
    source: Any                 # The threatening unit
    threat_type: str           # 'missile', 'radar_lock', 'aircraft', 'gun'
    threat_level: str          # 'low', 'medium', 'high', 'critical'
    distance_m: float          # Distance to threat
    bearing_deg: float         # Bearing to threat
    closure_rate_mps: float    # Rate of closure (positive = approaching)
    estimated_time_to_impact: Optional[float]  # Seconds until impact (if applicable)
    confidence: float          # Confidence in threat assessment (0.0-1.0)
    
    
class ThreatAssessment:
    """Analyzes the tactical situation and identifies threats."""
    
    def __init__(self, aircraft, config):
        self.aircraft = aircraft
        self.config = config
        self.previous_threats = []
        self.threat_history = {}  # Track threat persistence over time
        
    def assess_threats(self, simulator) -> List[Threat]:
        """
        Assess all current threats to the aircraft.
        
        Args:
            simulator: The simulation environment
            
        Returns:
            List of identified threats, sorted by priority
        """
        threats = []
        
        # Assess missile threats
        threats.extend(self._assess_missile_threats(simulator))
        
        # Assess radar lock threats
        threats.extend(self._assess_radar_threats(simulator))
        
        # Assess aircraft threats (potential missile launchers)
        threats.extend(self._assess_aircraft_threats(simulator))
        
        # Assess gun threats (close-in weapons)
        threats.extend(self._assess_gun_threats(simulator))
        
        # Sort threats by priority (critical first)
        threat_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        threats.sort(key=lambda t: threat_priority.get(t.threat_level, 0), reverse=True)
        
        # Update threat history for persistence tracking
        self._update_threat_history(threats)
        
        self.previous_threats = threats
        return threats
        
    def _assess_missile_threats(self, simulator) -> List[Threat]:
        """Assess incoming missile threats."""
        threats = []
        
        for unit in simulator.active_units.values():
            if not getattr(unit, 'is_missile', False):
                continue
                
            # Check if missile is targeting us
            missile_target = getattr(unit, 'target', None)
            if missile_target != self.aircraft:
                continue
                
            # Calculate threat parameters
            distance = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                unit.position.lat, unit.position.lon, unit.position.alt
            ) * 1000  # Convert to meters
            
            bearing = geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                unit.position.lat, unit.position.lon
            )
            
            # Estimate closure rate and time to impact
            closure_rate = self._estimate_closure_rate(unit)
            time_to_impact = distance / max(closure_rate, 1.0) if closure_rate > 0 else None
            
            # Determine threat level based on distance and missile type
            threat_level = self._classify_missile_threat(unit, distance, time_to_impact)
            
            threat = Threat(
                source=unit,
                threat_type='missile',
                threat_level=threat_level,
                distance_m=distance,
                bearing_deg=bearing,
                closure_rate_mps=closure_rate,
                estimated_time_to_impact=time_to_impact,
                confidence=0.9  # High confidence for detected missiles
            )
            
            threats.append(threat)
            
        return threats
        
    def _assess_radar_threats(self, simulator) -> List[Threat]:
        """Assess radar lock and tracking threats."""
        threats = []
        
        # Check if we're being tracked by enemy radars
        if hasattr(self.aircraft, 'sensor') and hasattr(self.aircraft.sensor, 'passive_radar'):
            radar_contacts = getattr(self.aircraft.sensor.passive_radar, 'contacts', {})
            
            for contact_id, contact_data in radar_contacts.items():
                # Determine if this is an active lock vs passive detection
                lock_strength = getattr(contact_data, 'lock_strength', 0.0)
                if lock_strength < 0.3:  # Weak signal, probably not a lock
                    continue
                    
                # Find the source unit
                source_unit = None
                for unit in simulator.active_units.values():
                    if unit.id == contact_id and unit.group != self.aircraft.group:
                        source_unit = unit
                        break
                        
                if not source_unit:
                    continue
                    
                distance = geodetic_distance_km(
                    self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                    source_unit.position.lat, source_unit.position.lon, source_unit.position.alt
                ) * 1000  # Convert to meters
                
                bearing = geodetic_bearing_deg(
                    self.aircraft.position.lat, self.aircraft.position.lon,
                    source_unit.position.lat, source_unit.position.lon
                )
                
                # Classify threat level based on lock strength and platform capability
                threat_level = self._classify_radar_threat(source_unit, lock_strength, distance)
                
                threat = Threat(
                    source=source_unit,
                    threat_type='radar_lock',
                    threat_level=threat_level,
                    distance_m=distance,
                    bearing_deg=bearing,
                    closure_rate_mps=0.0,  # Radar threats don't have closure rate
                    estimated_time_to_impact=None,
                    confidence=lock_strength
                )
                
                threats.append(threat)
                
        return threats
        
    def _assess_aircraft_threats(self, simulator) -> List[Threat]:
        """Assess threats from enemy aircraft (potential missile launchers)."""
        threats = []
        
        for unit in simulator.active_units.values():
            if (unit.group == self.aircraft.group or 
                getattr(unit, 'is_missile', False) or
                not hasattr(unit, 'missiles')):
                continue
                
            # Check if aircraft has weapons and could engage us
            remaining_missiles = getattr(unit, 'remaining_missiles', 0)
            if remaining_missiles <= 0:
                continue  # No missiles left
                
            distance = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                unit.position.lat, unit.position.lon, unit.position.alt
            ) * 1000  # Convert to meters
            
            # Only consider aircraft within reasonable engagement range
            max_threat_range = 100000  # 100km
            if distance > max_threat_range:
                continue
                
            bearing = geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                unit.position.lat, unit.position.lon
            )
            
            closure_rate = self._estimate_closure_rate(unit)
            
            # Classify threat based on distance, weapons, and geometry
            threat_level = self._classify_aircraft_threat(unit, distance, remaining_missiles)
            
            threat = Threat(
                source=unit,
                threat_type='aircraft',
                threat_level=threat_level,
                distance_m=distance,
                bearing_deg=bearing,
                closure_rate_mps=closure_rate,
                estimated_time_to_impact=None,
                confidence=0.7  # Moderate confidence for aircraft threats
            )
            
            threats.append(threat)
            
        return threats
        
    def _assess_gun_threats(self, simulator) -> List[Threat]:
        """Assess close-in gun threats."""
        threats = []
        
        for unit in simulator.active_units.values():
            if (unit.group == self.aircraft.group or 
                getattr(unit, 'is_missile', False)):
                continue
                
            distance = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                unit.position.lat, unit.position.lon, unit.position.alt
            ) * 1000  # Convert to meters
            
            # Only consider gun threats within gun range (typically <2km)
            max_gun_range = 2000
            if distance > max_gun_range:
                continue
                
            # Check if enemy is pointing at us (gun threat)
            target_bearing = geodetic_bearing_deg(
                unit.position.lat, unit.position.lon,
                self.aircraft.position.lat, self.aircraft.position.lon
            )
            
            angle_off_nose = abs(signed_yaw_deg_diff(unit.yaw_deg, target_bearing))
            
            # Only consider it a gun threat if they're roughly pointing at us
            if angle_off_nose > 30:  # 30 degree cone
                continue
                
            bearing = geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                unit.position.lat, unit.position.lon
            )
            
            closure_rate = self._estimate_closure_rate(unit)
            
            threat = Threat(
                source=unit,
                threat_type='gun',
                threat_level='high',  # Gun threats are always high priority when close
                distance_m=distance,
                bearing_deg=bearing,
                closure_rate_mps=closure_rate,
                estimated_time_to_impact=distance / max(closure_rate, 1.0) if closure_rate > 0 else None,
                confidence=0.8
            )
            
            threats.append(threat)
            
        return threats
        
    def _estimate_closure_rate(self, unit) -> float:
        """Estimate the closure rate between us and another unit."""
        try:
            # Get velocity vectors
            our_velocity = getattr(self.aircraft, 'velocity', None)
            their_velocity = getattr(unit, 'velocity', None)
            
            if our_velocity is None or their_velocity is None:
                return 0.0
                
            # Calculate relative velocity vector
            rel_vx = their_velocity.vx - our_velocity.vx
            rel_vy = their_velocity.vy - our_velocity.vy
            
            # Calculate bearing vector from us to them
            bearing_rad = np.radians(geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                unit.position.lat, unit.position.lon
            ))
            
            bearing_vx = np.sin(bearing_rad)
            bearing_vy = np.cos(bearing_rad)
            
            # Project relative velocity onto bearing vector (positive = approaching)
            closure_rate = rel_vx * bearing_vx + rel_vy * bearing_vy
            
            return closure_rate
            
        except:
            return 0.0
            
    def _classify_missile_threat(self, missile, distance: float, time_to_impact: Optional[float]) -> str:
        """Classify the threat level of an incoming missile."""
        if distance < 2000:  # 2km
            return 'critical'
        elif distance < 5000:  # 5km
            return 'high'
        elif distance < 15000:  # 15km
            return 'medium'
        else:
            return 'low'
            
    def _classify_radar_threat(self, aircraft, lock_strength: float, distance: float) -> str:
        """Classify the threat level of a radar lock."""
        if lock_strength > 0.8 and distance < 30000:  # Strong lock, close
            return 'high'
        elif lock_strength > 0.6:  # Moderate lock
            return 'medium'
        else:
            return 'low'
            
    def _classify_aircraft_threat(self, aircraft, distance: float, missiles: int) -> str:
        """Classify the threat level of an enemy aircraft."""
        # Base threat on distance and capability
        if distance < 20000 and missiles >= 4:  # Close with many missiles
            return 'high'
        elif distance < 40000 and missiles >= 2:  # Medium range with missiles
            return 'medium'
        else:
            return 'low'
            
    def _update_threat_history(self, current_threats: List[Threat]):
        """Update threat history for persistence tracking."""
        # Use a simple incrementing counter instead of sim_time to avoid datetime issues
        current_time = getattr(self, '_update_counter', 0) + 1
        setattr(self, '_update_counter', current_time)
        
        for threat in current_threats:
            threat_id = id(threat.source)
            if threat_id not in self.threat_history:
                self.threat_history[threat_id] = {
                    'first_detected': current_time,
                    'last_seen': current_time,
                    'max_threat_level': threat.threat_level
                }
            else:
                self.threat_history[threat_id]['last_seen'] = current_time
                # Update max threat level if current is higher
                levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                if levels.get(threat.threat_level, 0) > levels.get(self.threat_history[threat_id]['max_threat_level'], 0):
                    self.threat_history[threat_id]['max_threat_level'] = threat.threat_level
                    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get a summary of the current threat situation."""
        if not self.previous_threats:
            return {'total_threats': 0, 'highest_threat': 'none', 'immediate_danger': False}
            
        threat_counts = {}
        highest_threat = 'low'
        immediate_danger = False
        
        for threat in self.previous_threats:
            threat_type = threat.threat_type
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
            
            # Update highest threat level
            levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            if levels.get(threat.threat_level, 0) > levels.get(highest_threat, 0):
                highest_threat = threat.threat_level
                
            # Check for immediate danger
            if threat.threat_level in ['critical', 'high'] and threat.distance_m < 10000:
                immediate_danger = True
                
        return {
            'total_threats': len(self.previous_threats),
            'threat_counts': threat_counts,
            'highest_threat': highest_threat,
            'immediate_danger': immediate_danger
        }
        
    def update_config(self, config):
        """Update configuration."""
        self.config = config