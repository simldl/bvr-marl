"""
ECM (Electronic Counter-Measures) emitter for generating jamming effects
and deception returns (false targets/ghosts) that affect aircraft radars only.
"""

import math
import numpy as np
from typing import Optional, Set, List, Dict, Any
from radar.core.utils import geodetic_to_enu


class ECMEmitter:
    """
    Constant jammer that:
      - degrades victim radar SNR via additive noise power J
      - injects synthetic "ghost" detections (deception)

    Affects aircraft radars only; missile seekers are unaffected.
    """

    def __init__(
        self,
        owner,
        erp_w: float,
        bw_hz: float,
        hop_period_s: float = 0.1,
        techniques: Optional[Set[str]] = None,
        assist_ahead_deg: float = 45.0,
        assist_behind_deg: float = 45.0,
        np_rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize ECM emitter.

        Args:
            owner: The unit that owns this jammer (aircraft)
            erp_w: Effective Radiated Power in watts
            bw_hz: Jamming bandwidth in Hz
            hop_period_s: Frequency hopping period in seconds
            techniques: Set of jamming techniques (e.g., "drfm_multi_false")
            assist_ahead_deg: Assist sector ahead (degrees)
            assist_behind_deg: Assist sector behind (degrees)
            np_rng: NumPy random generator for reproducibility
        """
        self.owner = owner
        self.erp_w = abs(float(erp_w))  # Convert negative ERP to positive
        self.bw_hz = float(bw_hz)
        self.hop_period_s = float(hop_period_s)
        self.techniques = techniques if techniques is not None else {"drfm_multi_false"}
        self.assist_ahead_deg = float(assist_ahead_deg)
        self.assist_behind_deg = float(assist_behind_deg)

        # Random number generator for deception
        if np_rng is not None:
            self.np_rng = np_rng
        else:
            # Get seed from owner ID, handling both int and string IDs
            owner_id = getattr(owner, 'id', 0) if owner is not None else 0
            if isinstance(owner_id, str):
                # Convert string ID to integer using hash
                seed = abs(hash(owner_id)) % (2**32)
            else:
                seed = int(owner_id) if owner_id else 0
            self.np_rng = np.random.default_rng(seed)

        # Internal state
        self._hop_timer = 0.0
        self._current_freq_hz = None  # Current jamming frequency (if hopping)
        self._on_target = True  # Whether jammer is currently on victim frequency

    def compute_J(
        self,
        victim_radar,
        t: float,
        victim_position
    ) -> float:
        """
        Compute additive jammer power J (watts) received at victim radar.

        J = (ERP * G_rx * λ^2) / ((4π)^2 * R^2 * L)

        Simplified: J ∝ ERP / R^2 * overlap_factor

        Args:
            victim_radar: The radar being jammed
            t: Current simulation time (seconds)
            victim_position: Position of the victim radar

        Returns:
            Jammer power J in watts
        """
        if self.owner is None or not hasattr(self.owner, 'position'):
            return 0.0

        # Range in victim ENU
        e, n, u = geodetic_to_enu(
            self.owner.position.lat, self.owner.position.lon, self.owner.position.alt,
            victim_position.lat, victim_position.lon, victim_position.alt
        )
        R = math.sqrt(e*e + n*n + u*u)
        if R < 1.0:
            R = 1.0

        c = 3e8
        lambda_m = c / float(victim_radar.freq_hz)

        # Simplified free-space link
        J_power = self.erp_w * (lambda_m ** 2) / ((4.0 * math.pi * R) ** 2)

        # Pattern/overlap model placeholder
        overlap_factor = 1.0

        # Frequency/time overlap
        freq_overlap = self._compute_frequency_overlap(victim_radar.freq_hz)
        duty_cycle = 1.0

        J_effective = J_power * overlap_factor * freq_overlap * duty_cycle
        return max(0.0, J_effective)


    def _compute_frequency_overlap(self, victim_freq_hz: float) -> float:
        """
        Compute frequency/time overlap factor [0, 1].

        Models frequency agile radars slipping jammer coverage:
        - If hop timer phase aligns, jammer is on-target (return 1.0)
        - If misaligned, jammer misses victim frequency (return 0.0)
        - Duty cycle: fraction of hop period jammer is on-target

        Args:
            victim_freq_hz: Victim radar frequency

        Returns:
            Overlap factor (1.0 = on-target, 0.0 = missed, or fractional duty)
        """
        # Simplified binary model: 70% duty cycle (jammer on-target 70% of the time)
        # More realistic: phase = (hop_timer % hop_period) / hop_period
        # If phase < duty_threshold, jammer is on-target
        duty_threshold = 0.7  # 70% duty cycle

        phase = self._hop_timer / self.hop_period_s if self.hop_period_s > 0 else 1.0
        phase_normalized = phase % 1.0  # Wrap to [0, 1)

        # Binary model: on-target if phase < duty threshold
        if phase_normalized < duty_threshold:
            return 1.0
        else:
            return 0.0

    def deception_burst(
        self,
        victim_radar,
        victim_position,
        t: float
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic "ghost" detections (false targets) for victim radar.

        These ghosts:
          - Have is_deception=True
          - Have engagement_id = jammer unit id
          - Have explicit T=None (not trusted/ground-truth)
          - Include all required fields (dop, lat/lon/alt)
          - Flow through clustering and tracking like normal contacts

        Args:
            victim_radar: The radar being deceived
            victim_position: Position of the victim radar
            t: Current simulation time

        Returns:
            List of synthetic detections (ghosts)
        """
        ghosts = []

        if "drfm_multi_false" not in self.techniques:
            return ghosts

        if self.owner is None or not hasattr(self.owner, 'position'):
            return ghosts

        # Get victim radar FOV limits
        h_fov_deg = getattr(victim_radar, 'h_fov_deg', 120.0) / 2.0  # Half-angle
        v_fov_deg = getattr(victim_radar, 'v_fov_deg', 60.0) / 2.0   # Half-angle

        # Compute jammer LOS in victim radar's local ENU frame
        jammer_enu = geodetic_to_enu(
            self.owner.position.lat, self.owner.position.lon, self.owner.position.alt,
            victim_position.lat, victim_position.lon, victim_position.alt
        )

        # Convert ENU to spherical (az, el, range) using mathematical azimuth:
        e, n, u = jammer_enu
        true_range = math.sqrt(e*e + n*n + u*u)
        true_az_rad = math.atan2(n, e)  # 0° = +East, CCW to +North
        true_el_rad = math.atan2(u, math.sqrt(e*e + n*n))

        true_az_deg = math.degrees(true_az_rad)
        true_el_deg = math.degrees(true_el_rad)

        # Generate 3-5 ghosts with random offsets
        num_ghosts = self.np_rng.integers(3, 6)

        for k in range(num_ghosts):
            # Random offsets in local frame
            az_offset = self.np_rng.normal(0.0, 5.0)  # degrees
            el_offset = self.np_rng.normal(0.0, 3.0)  # degrees
            range_offset = self.np_rng.normal(0.0, 0.1 * true_range)  # 10% range spread

            ghost_az = true_az_deg + az_offset
            ghost_el = true_el_deg + el_offset
            ghost_range = max(1.0, true_range + range_offset)

            # FOV filtering: reject ghosts outside radar FOV
            if abs(ghost_az) > h_fov_deg or abs(ghost_el) > v_fov_deg:
                continue

            # SNR for ghost (should be plausible)
            ghost_snr_db = self.np_rng.uniform(
                victim_radar.snr_threshold_db + 2.0,
                victim_radar.snr_threshold_db + 10.0
            )

            # Compute realistic Doppler jitter (DRFM with radial-rate noise)
            lambda_m = 3e8 / float(victim_radar.freq_hz)
            vr_mps = self.np_rng.normal(0.0, 30.0)  # ±30 m/s radial velocity jitter
            dop_hz = 2.0 * vr_mps / lambda_m

            ghost = {
                "az": ghost_az,
                "el": ghost_el,
                "d": ghost_range,
                "dop": float(dop_hz),  # Realistic Doppler jitter
                "snr_db": ghost_snr_db,
                "is_deception": True,
                "engagement_id": self.owner.id,  # Jammer unit ID
                "jammer_id": self.owner.id,
                "T": None,  # Explicit None to avoid KeyError in clusterer
            }

            ghosts.append(ghost)

        return ghosts

    def update(self, dt: float):
        """
        Update jammer state (e.g., frequency hopping timer).

        Args:
            dt: Time step in seconds
        """
        self._hop_timer += dt

        if self._hop_timer >= self.hop_period_s:
            self._hop_timer = 0.0
            # Frequency hop (placeholder - could randomize frequency here)
            # For now, just reset timer
