"""
EWWorld: Global manager for electronic warfare interactions.
Coordinates jamming effects and deception returns across all radars in the simulation.
"""

from typing import List, Dict, Tuple, Any
import math
from radar.core.utils import geodetic_to_enu


class EWWorld:
    """
    Central coordinator for ECM (Electronic Counter-Measures) in the simulation.

    Manages:
    - Jamming power accumulation from multiple enemy jammers
    - Deception return (ghost) injection into victim radars
    - Cooperative jamming assist allocation (future)
    """

    def __init__(self, sim):
        """
        Initialize EWWorld.

        Args:
            sim: Simulator instance with access to active_units registry
        """
        self.sim = sim

    def collect_incoming(self, victim_radar, t: float) -> Tuple[List[Tuple[float, float, float]], List[Dict[str, Any]]]:
        """
        Collect all incoming jamming effects on a victim radar.

        This is the main hook called by each radar during its update cycle.
        For aircraft radars (jam_susceptible=True), accumulates:
        - Per-jammer info (azimuth, elevation, power) for angular weighting
        - Deception returns (ghosts) from enemy DRFM techniques

        For missile seekers (jam_susceptible=False), returns zero effect.

        Args:
            victim_radar: The radar being jammed
            t: Current simulation time (seconds)

        Returns:
            (jammer_info, ghosts) where:
                jammer_info: List of (jammer_az_deg, jammer_el_deg, J_power) tuples
                ghosts: List of synthetic detection dicts with is_deception=True
        """
        # Missile seekers are immune to jamming
        if not getattr(victim_radar, "jam_susceptible", True):
            return [], []

        # Get victim position
        victim_owner = getattr(victim_radar, "owner", None)
        if victim_owner is None or not hasattr(victim_owner, "position"):
            return [], []

        victim_position = victim_owner.position
        victim_group = getattr(victim_owner, "group", None)

        # Accumulate jamming from all enemy units with ECM
        jammer_info = []
        ghosts = []

        for unit in self.sim.active_units.values():
            # Skip if no ECM system
            ecm = getattr(unit, "ecm", None)
            if ecm is None:
                continue

            # Skip friendlies (same group)
            if unit.group == victim_group:
                continue

            # Compute jammer LOS in victim frame
            jammer_enu = geodetic_to_enu(
                unit.position.lat, unit.position.lon, unit.position.alt,
                victim_position.lat, victim_position.lon, victim_position.alt
            )
            e, n, u = jammer_enu
            jammer_az_deg = math.degrees(math.atan2(n, e))  # 0° = East, CCW
            jammer_el_deg = math.degrees(math.atan2(u, math.sqrt(e*e + n*n)))

            # Compute jammer power
            J_power = ecm.compute_J(victim_radar, t, victim_position)
            jammer_info.append((jammer_az_deg, jammer_el_deg, J_power))

            # Collect deception returns (ghosts)
            burst = ecm.deception_burst(victim_radar, victim_position, t)
            ghosts.extend(burst)

        return jammer_info, ghosts

    def schedule_assist(self, own_radar, team_radars, enemy_tracks, t: float, K: int = 2):
        """
        Cooperative jamming assist scheduling.

        Uses TrackPrioritySystem to:
        - Identify highest-priority enemy radars threatening teammates
        - Allocate jammer time/beam toward threats in assist sectors
        - Coordinate multi-ship jamming for maximum effect

        Args:
            own_radar: Ownship radar/ECM system
            team_radars: List of friendly radars in datalink
            enemy_tracks: List of enemy track records
            t: Current simulation time
            K: Number of top threats to jam

        Notes:
            Light implementation using threat scoring.
            Focuses jamming on K highest-priority threats.
        """
        if not hasattr(own_radar, 'owner') or own_radar.owner is None:
            return

        # Get ECM system
        ecm = getattr(own_radar.owner, 'ecm', None)
        if ecm is None:
            return

        try:
            # Simple distance-based prioritization
            scored_tracks = []
            for track in enemy_tracks:
                # Unpack track tuple
                tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count, is_deception, suspect_deception, engagement_id, jammer_id, engageable = track

                if tgt is None or suspect_deception:
                    continue

                # Simple threat score: inverse distance (closer = higher priority)
                dist = float((state[0]**2 + state[1]**2 + state[2]**2)**0.5)
                threat_score = 1.0 / (dist + 1.0)

                scored_tracks.append((track, threat_score, tgt))

            # Sort by threat score (descending)
            scored_tracks.sort(key=lambda x: x[1], reverse=True)

            # Focus jamming on top K threats
            for track, score, tgt in scored_tracks[:K]:
                # Check if teammate needs assist
                if self._teammate_needs_assist(own_radar.owner, team_radars, tgt):
                    # Point jammer at this threat (future: steer mainlobe/time slot)
                    # ecm.point_at(tgt)  # Placeholder for beam steering
                    pass

        except Exception:
            # Graceful degradation
            pass

    def _teammate_needs_assist(self, ownship, team_radars, threat):
        """
        Check if a teammate needs jamming assist against a threat.

        Args:
            ownship: Own aircraft
            team_radars: List of (radar, position) tuples
            threat: Threat target

        Returns:
            True if assist is needed
        """
        # Placeholder: implement geometry check for assist sectors
        return False  # Conservative default
