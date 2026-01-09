"""
Metrics Helper for BVR AI.

Provides tactical metrics:
- DLZ zone determination
- A-pole & F-pole estimates
- MAR (Minimum Abort Range)
- SQI (continuous)
- Lock quality proxy (LOS rate, ATA margin to FOV edge)
- Missile inventory by class
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from simulator.utils.geodesics import geodetic_distance_km


class MetricsHelper:
    """
    Tactical metrics calculator for BVR combat.
    Computes A-pole, F-pole, MAR, and other engagement metrics.
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft

    def get_dlz_zone_metrics(self, target) -> Dict[str, Any]:
        """
        Get DLZ zone metrics for target.

        Returns:
        - zone: str (R1/R2/R3/R4)
        - nez_visible: bool
        - in_min_range: bool (< R_min)
        - in_nez: bool (R_min < range < R_tr)
        - in_launch_envelope: bool (R_tr < range < R_aero)
        - beyond_max: bool (> R_aero)
        """
        if not target or not hasattr(self.aircraft, 'wez'):
            return {
                'valid': False,
                'zone': 'R4',
                'nez_visible': False,
                'in_min_range': False,
                'in_nez': False,
                'in_launch_envelope': False,
                'beyond_max': True
            }

        try:
            wez = self.aircraft.wez
            dlz = wez.compute_dlz(target)
            slant_range = wez._slant_range_m(self.aircraft, target)
            zone = wez.zone_for_range(slant_range, dlz)
            nez_vis = wez.nez_visible(slant_range, dlz, show_in=("R2", "R3"))

            return {
                'valid': True,
                'zone': zone,
                'nez_visible': nez_vis,
                'in_min_range': slant_range < dlz.r_min_m,
                'in_nez': dlz.r_min_m <= slant_range < dlz.r_tr_m,
                'in_launch_envelope': dlz.r_tr_m <= slant_range < dlz.r_aero_m,
                'beyond_max': slant_range >= dlz.r_aero_m,
                'slant_range_m': slant_range
            }
        except Exception as e:
            return {
                'valid': False,
                'zone': 'R4',
                'nez_visible': False,
                'in_min_range': False,
                'in_nez': False,
                'in_launch_envelope': False,
                'beyond_max': True,
                'error': str(e)
            }

    def estimate_f_pole(self, target) -> Dict[str, Any]:
        """
        Estimate F-pole (range when missile goes active/pitbull).

        For Fox-3 missiles:
        - F-pole = range when missile's seeker activates
        - Typically 10-20km from target depending on missile type

        Returns:
        - f_pole_range_m: estimated F-pole range
        - time_to_f_pole_s: estimated time until F-pole (if missile in flight)
        - is_estimate: bool (True - this is an approximation)
        """
        if not target:
            return {
                'valid': False,
                'f_pole_range_m': 0.0,
                'time_to_f_pole_s': 0.0,
                'is_estimate': True
            }

        try:
            # Get missile parameters
            missile_types = getattr(self.aircraft, 'missile_types', [])
            if not missile_types:
                # No missiles - return defaults
                return {
                    'valid': False,
                    'f_pole_range_m': 0.0,
                    'time_to_f_pole_s': 0.0,
                    'is_estimate': True
                }

            # Create temp missile to get active seeker range
            missile_cls = missile_types[0]
            try:
                temp_missile = missile_cls(None, None, self.aircraft, self.aircraft.map_limits, self.aircraft.group)
                # For Fox-3, F-pole is typically when missile goes active
                # Estimate: 15km from target (typical Fox-3 active range)
                active_range = getattr(temp_missile.radar, 'max_range_m', 20000) if hasattr(temp_missile, 'radar') else 20000
                f_pole_range = min(active_range * 0.75, 20000)  # Conservative estimate: 75% of seeker range or 20km
            except:
                f_pole_range = 15000  # Default 15km

            # Time to F-pole (rough estimate based on closure rate)
            slant_range = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                target.position.lat, target.position.lon, target.position.alt
            ) * 1000.0

            # Estimate missile flight time (assuming ~Mach 3-4 average speed)
            avg_missile_speed = 1000.0  # ~Mach 3
            time_to_f_pole = max(0.0, (slant_range - f_pole_range) / avg_missile_speed)

            return {
                'valid': True,
                'f_pole_range_m': f_pole_range,
                'time_to_f_pole_s': time_to_f_pole,
                'is_estimate': True
            }
        except Exception as e:
            return {
                'valid': False,
                'f_pole_range_m': 0.0,
                'time_to_f_pole_s': 0.0,
                'is_estimate': True,
                'error': str(e)
            }

    def estimate_a_pole(self, target) -> Dict[str, Any]:
        """
        Estimate A-pole (range when missile arrives at target).

        For active missiles (Fox-3):
        - A-pole is less relevant (can go cold after F-pole)

        For SARH missiles (Fox-1):
        - A-pole = 0 (must maintain lock until impact)

        Returns:
        - a_pole_range_m: estimated A-pole range
        - time_to_a_pole_s: estimated time until impact
        - requires_lock_at_impact: bool
        """
        if not target:
            return {
                'valid': False,
                'a_pole_range_m': 0.0,
                'time_to_a_pole_s': 0.0,
                'requires_lock_at_impact': False
            }

        try:
            # Determine missile guidance type
            missile_types = getattr(self.aircraft, 'missile_types', [])
            if not missile_types:
                return {
                    'valid': False,
                    'a_pole_range_m': 0.0,
                    'time_to_a_pole_s': 0.0,
                    'requires_lock_at_impact': False
                }

            missile_cls = missile_types[0]
            try:
                temp_missile = missile_cls(None, None, self.aircraft, self.aircraft.map_limits, self.aircraft.group)
                fox_type = getattr(temp_missile, 'fox_type', 3)
                requires_lock = fox_type == 1  # Fox-1 (SARH) requires continuous lock
            except:
                fox_type = 3
                requires_lock = False

            # A-pole is always 0 for practical purposes (missile at target)
            slant_range = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                target.position.lat, target.position.lon, target.position.alt
            ) * 1000.0

            # Estimate time to impact (assuming ~Mach 3-4 average)
            avg_missile_speed = 1000.0
            time_to_impact = slant_range / avg_missile_speed

            return {
                'valid': True,
                'a_pole_range_m': 0.0,  # A-pole is at target
                'time_to_a_pole_s': time_to_impact,
                'requires_lock_at_impact': requires_lock,
                'fox_type': fox_type
            }
        except Exception as e:
            return {
                'valid': False,
                'a_pole_range_m': 0.0,
                'time_to_a_pole_s': 0.0,
                'requires_lock_at_impact': False,
                'error': str(e)
            }

    def estimate_mar(self, threat_aircraft) -> Dict[str, Any]:
        """
        Estimate MAR (Minimum Abort Range).

        MAR is the minimum range at which we can successfully escape
        an enemy missile if we turn cold immediately.

        Returns:
        - mar_range_m: estimated MAR
        - inside_mar: bool (are we inside MAR now?)
        - escape_heading_deg: recommended escape heading
        """
        if not threat_aircraft:
            return {
                'valid': False,
                'mar_range_m': 0.0,
                'inside_mar': False,
                'escape_heading_deg': 0.0
            }

        try:
            # Calculate range to threat
            threat_range = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                threat_aircraft.position.lat, threat_aircraft.position.lon, threat_aircraft.position.alt
            ) * 1000.0

            # Base MAR calculation (conservative: ~60-70% of threat missile range)
            # Assume modern BVR missile: ~60-100km range
            threat_missile_range = 80000.0  # Assume 80km threat missile
            base_mar = threat_missile_range * 0.65

            # Adjust for own speed (faster = smaller MAR)
            our_speed = self.aircraft.speed
            speed_factor = max(0.7, 300.0 / max(our_speed, 1.0))

            # Adjust for geometry (head-on requires more distance)
            from simulator.utils.geodesics import geodetic_bearing_deg
            from simulator.utils.angles import signed_yaw_deg_diff

            bearing_to_threat = geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                threat_aircraft.position.lat, threat_aircraft.position.lon
            )
            bearing_from_threat = geodetic_bearing_deg(
                threat_aircraft.position.lat, threat_aircraft.position.lon,
                self.aircraft.position.lat, self.aircraft.position.lon
            )
            threat_aspect = abs(signed_yaw_deg_diff(
                getattr(threat_aircraft, 'yaw_deg', 0.0),
                bearing_from_threat
            ))

            aspect_factor = 1.2 if threat_aspect < 30 else 1.0

            mar = base_mar * speed_factor * aspect_factor
            mar = min(mar, 50000)  # Cap at 50km

            # Escape heading (opposite of threat bearing)
            escape_heading = (bearing_to_threat + 180.0) % 360.0

            inside_mar = threat_range < mar

            return {
                'valid': True,
                'mar_range_m': mar,
                'inside_mar': inside_mar,
                'escape_heading_deg': escape_heading,
                'threat_range_m': threat_range
            }
        except Exception as e:
            return {
                'valid': False,
                'mar_range_m': 0.0,
                'inside_mar': False,
                'escape_heading_deg': 0.0,
                'error': str(e)
            }

    def get_sqi(self, target) -> Dict[str, Any]:
        """
        Get SQI (Target Intercept Probability with Instant launch).

        Returns continuous quality metric [0..1] based on:
        - Distance (closer is better within envelope)
        - Closure rate
        - Aspect angle
        - Altitude/energy state
        """
        if not target or not hasattr(self.aircraft, 'wez'):
            return {
                'valid': False,
                'sqi': 0.0
            }

        try:
            wez = self.aircraft.wez
            sqi = wez.sqi(self.aircraft, target, missile=None, dlz=None)

            return {
                'valid': True,
                'sqi': sqi
            }
        except Exception as e:
            return {
                'valid': False,
                'sqi': 0.0,
                'error': str(e)
            }

    def get_lock_quality(self, target) -> Dict[str, Any]:
        """
        Get lock quality proxy metrics.

        Returns:
        - has_lock: bool
        - ata_margin_deg: margin to FOV edge (how much room before losing lock)
        - range_margin: margin to radar max range
        - lock_strength: composite quality [0..1]
        """
        if not target:
            return {
                'valid': False,
                'has_lock': False,
                'ata_margin_deg': 0.0,
                'range_margin': 0.0,
                'lock_strength': 0.0
            }

        try:
            # Check lock status
            has_lock = False
            if hasattr(self.aircraft, 'sensor') and self.aircraft.sensor:
                has_lock = self.aircraft.sensor.has_radar_lock(target)

            # Calculate ATA margin
            from simulator.utils.geodesics import geodetic_bearing_deg
            from simulator.utils.angles import signed_yaw_deg_diff

            bearing = geodetic_bearing_deg(
                self.aircraft.position.lat, self.aircraft.position.lon,
                target.position.lat, target.position.lon
            )
            ata = abs(signed_yaw_deg_diff(self.aircraft.yaw_deg, bearing))
            fov_half = self.aircraft.radar.h_fov_deg / 2.0
            ata_margin = max(0.0, fov_half - ata)

            # Calculate range margin
            slant_range = geodetic_distance_km(
                self.aircraft.position.lat, self.aircraft.position.lon, self.aircraft.position.alt,
                target.position.lat, target.position.lon, target.position.alt
            ) * 1000.0
            radar_max = self.aircraft.radar.max_range_m
            range_margin = max(0.0, radar_max - slant_range)

            # Composite lock strength [0..1]
            ata_score = min(1.0, ata_margin / 30.0)  # Full score if >30deg margin
            range_score = min(1.0, range_margin / 20000.0)  # Full score if >20km margin
            lock_strength = (ata_score * 0.6 + range_score * 0.4) if has_lock else 0.0

            return {
                'valid': True,
                'has_lock': has_lock,
                'ata_margin_deg': ata_margin,
                'range_margin_m': range_margin,
                'lock_strength': lock_strength
            }
        except Exception as e:
            return {
                'valid': False,
                'has_lock': False,
                'ata_margin_deg': 0.0,
                'range_margin_m': 0.0,
                'lock_strength': 0.0,
                'error': str(e)
            }

    def get_missile_inventory_by_class(self) -> Dict[str, int]:
        """
        Get missile inventory broken down by Fox type.

        Returns dict with Fox-1, Fox-2, Fox-3 counts.
        NOTE: Current implementation simplifies all as Fox-3.
        FUTURE: Implement proper per-missile-type tracking.
        """
        try:
            remaining = getattr(self.aircraft, 'remaining_missiles', 0)
            if remaining == 0 and hasattr(self.aircraft, 'weapons'):
                remaining = getattr(self.aircraft.weapons, 'remaining_missiles', 0)

            # FUTURE: Implement actual inventory tracking by missile class
            # For now, assume all missiles are Fox-3 (AMRAAM-like)
            return {
                'fox1': 0,      # SARH (Sparrow, Skyflash)
                'fox2': 0,      # IR (Sidewinder, Python)
                'fox3': remaining  # Active radar (AMRAAM, Meteor)
            }
        except:
            return {'fox1': 0, 'fox2': 0, 'fox3': 0}

    def get_comprehensive_metrics(self, target, primary_threat=None) -> Dict[str, Any]:
        """
        Get all tactical metrics in one call.

        Args:
            target: Selected target
            primary_threat: Primary threat aircraft (for MAR)

        Returns:
            Comprehensive metrics dict
        """
        return {
            'dlz_zone': self.get_dlz_zone_metrics(target),
            'f_pole': self.estimate_f_pole(target),
            'a_pole': self.estimate_a_pole(target),
            'mar': self.estimate_mar(primary_threat) if primary_threat else {'valid': False},
            'sqi': self.get_sqi(target),
            'lock_quality': self.get_lock_quality(target),
            'missile_inventory': self.get_missile_inventory_by_class()
        }
