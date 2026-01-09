import math
import torch
import torch.nn.functional as F
from typing import Optional

# OPTIMIZATION: Global cache for torch.linspace results
# Reduces repeated linspace computations during LUT building
_LINSPACE_CACHE = {}

def _cached_linspace(start: float, end: float, steps: int, device: torch.device) -> torch.Tensor:
    """Cached version of torch.linspace for common range/RCS configurations."""
    cache_key = (start, end, steps, str(device))
    if cache_key not in _LINSPACE_CACHE:
        _LINSPACE_CACHE[cache_key] = torch.linspace(start, end, steps, device=device)
    return _LINSPACE_CACHE[cache_key]

class DetectionLUT:
    """Builds and queries a lookup table for detection probability."""
    def __init__(self, freq_hz: float, tx_power_w: float, gain: float,
                 max_range_m: float, snr_threshold_db: float,
                 max_rcs: float = 100.0, rcs_bins: int = 100, dist_bins: int = 100,
                 device: Optional[torch.device] = None):
        self.freq_hz = freq_hz
        self.tx_power_w = tx_power_w
        self.gain = gain
        self.max_range_m = max_range_m
        self.snr_threshold_db = snr_threshold_db
        self._max_rcs = max_rcs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_lut(rcs_bins, dist_bins)

    def _build_lut(self, rcs_bins: int, dist_bins: int):
        wavelength = 3e8 / self.freq_hz
        # OPTIMIZATION: Use cached linspace for common configurations
        distances = _cached_linspace(1e-1, self.max_range_m, dist_bins, self.device)
        rcs_values = _cached_linspace(0.01, self._max_rcs, rcs_bins, self.device)
        self._distances = distances
        self._rcs_values = rcs_values

        # OPTIMIZATION: Cache min/max as Python floats to avoid .item() calls in hot path
        self._dmin = float(distances[0])
        self._dmax = float(distances[-1])
        self._rmin = float(rcs_values[0])
        self._rmax = float(rcs_values[-1])

        D, S = torch.meshgrid(distances, rcs_values, indexing='ij')

        # Radar Equation Parameters (fixed)
        k = 1.38e-23
        T0 = 290
        B = 1e6
        Fn = 3
        Ls = 2

        snr_lin = (self.tx_power_w * (self.gain**2) * wavelength**2 * S) / (
            (4 * math.pi)**3 * D**4 * k * T0 * B * Fn * Ls + 1e-12
        )
        snr_db = 10 * torch.log10(snr_lin + 1e-12)

        p_det = torch.sigmoid((snr_db - self.snr_threshold_db) / 2.0)
        self._lut = p_det.unsqueeze(0).unsqueeze(0)


    def get_probability(self, dist: float, rcs: float) -> float:
        """Interpolate detection probability from the LUT for a given dist and rcs."""

        if rcs >= self._max_rcs:
            return 1.0

        # OPTIMIZATION: Use cached Python floats instead of .item() calls
        dn = 2 * (dist - self._dmin) / (self._dmax - self._dmin) - 1
        rn = 2 * (rcs - self._rmin) / (self._rmax - self._rmin) - 1

        # OPTIMIZATION: Use Python max/min instead of torch.clamp to avoid tensor creation
        dn = max(-1.0, min(1.0, dn))
        rn = max(-1.0, min(1.0, rn))

        # Single tensor creation for grid_sample
        grid = torch.tensor([[[[rn, dn]]]], device=self.device)
        prob = float(F.grid_sample(self._lut, grid, align_corners=True)[0, 0, 0, 0])
        return prob
