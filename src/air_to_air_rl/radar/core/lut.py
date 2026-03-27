import math
from typing import Optional

import numpy as np
import torch

# Device selection resolved once at import time — avoids repeated syscall on every construction
_CUDA_AVAILABLE: bool = torch.cuda.is_available()

_LINSPACE_CACHE = {}  # Global cache for torch.linspace results

# Global LUT instance cache keyed by construction parameters — radars with identical
# configurations share the same pre-built LUT object (read-only after construction).
_LUT_INSTANCE_CACHE: dict = {}


def _cached_linspace(start: float, end: float, steps: int, device: torch.device) -> torch.Tensor:
    """Cached version of torch.linspace for common range/RCS configurations."""
    cache_key = (start, end, steps, str(device))
    if cache_key not in _LINSPACE_CACHE:
        _LINSPACE_CACHE[cache_key] = torch.linspace(start, end, steps, device=device)
    return _LINSPACE_CACHE[cache_key]


class DetectionLUT:
    """Builds and queries a lookup table for detection probability."""

    def __init__(
        self,
        freq_hz: float,
        tx_power_w: float,
        gain: float,
        max_range_m: float,
        snr_threshold_db: float,
        max_rcs: float = 100.0,
        rcs_bins: int = 100,
        dist_bins: int = 100,
        device: torch.device | None = None,
    ):
        self.freq_hz = freq_hz
        self.tx_power_w = tx_power_w
        self.gain = gain
        self.max_range_m = max_range_m
        self.snr_threshold_db = snr_threshold_db
        self._max_rcs = max_rcs
        self.device = device or torch.device("cuda" if _CUDA_AVAILABLE else "cpu")
        self._build_lut(rcs_bins, dist_bins)

    def _build_lut(self, rcs_bins: int, dist_bins: int):
        wavelength = 3e8 / self.freq_hz
        distances = _cached_linspace(1e-1, self.max_range_m, dist_bins, self.device)
        rcs_values = _cached_linspace(0.01, self._max_rcs, rcs_bins, self.device)
        self._distances = distances
        self._rcs_values = rcs_values

        self._dmin = float(distances[0])
        self._dmax = float(distances[-1])
        self._rmin = float(rcs_values[0])
        self._rmax = float(rcs_values[-1])

        D, S = torch.meshgrid(distances, rcs_values, indexing="ij")

        k = 1.38e-23
        T0 = 290
        B = 1e6
        Fn = 3
        Ls = 2

        snr_lin = (self.tx_power_w * (self.gain**2) * wavelength**2 * S) / (
            (4 * math.pi) ** 3 * D**4 * k * T0 * B * Fn * Ls + 1e-12
        )
        snr_db = 10 * torch.log10(snr_lin + 1e-12)

        p_det = torch.sigmoid((snr_db - self.snr_threshold_db) / 2.0)
        # Keep torch tensor for backward compat; also store numpy array for fast scalar queries
        self._lut = p_det.unsqueeze(0).unsqueeze(0)
        self._lut_np = p_det.detach().cpu().numpy()  # shape: (dist_bins, rcs_bins)
        self._lut_H = dist_bins
        self._lut_W = rcs_bins

    def get_probability(self, dist: float, rcs: float) -> float:
        """Interpolate detection probability from the LUT for a given dist and rcs.

        Uses direct bilinear interpolation on the pre-built NumPy array instead of
        allocating temporary torch tensors and calling grid_sample per query.
        Matches grid_sample align_corners=True convention exactly.
        """
        if rcs >= self._max_rcs:
            return 1.0

        dn = max(-1.0, min(1.0, 2.0 * (dist - self._dmin) / (self._dmax - self._dmin) - 1.0))
        rn = max(-1.0, min(1.0, 2.0 * (rcs - self._rmin) / (self._rmax - self._rmin) - 1.0))

        # grid_sample convention: x=rn → W (rcs) axis, y=dn → H (dist) axis, align_corners=True
        H = self._lut_H
        W = self._lut_W
        row_f = (dn + 1.0) * 0.5 * (H - 1)
        col_f = (rn + 1.0) * 0.5 * (W - 1)

        r0 = int(row_f)
        c0 = int(col_f)
        r1 = r0 + 1 if r0 + 1 < H else r0
        c1 = c0 + 1 if c0 + 1 < W else c0
        dr = row_f - r0
        dc = col_f - c0

        lut = self._lut_np
        return float(
            (1.0 - dr) * (1.0 - dc) * lut[r0, c0]
            + (1.0 - dr) * dc * lut[r0, c1]
            + dr * (1.0 - dc) * lut[r1, c0]
            + dr * dc * lut[r1, c1]
        )

    @classmethod
    def get_or_create(
        cls,
        freq_hz: float,
        tx_power_w: float,
        gain: float,
        max_range_m: float,
        snr_threshold_db: float,
        max_rcs: float = 100.0,
        rcs_bins: int = 100,
        dist_bins: int = 100,
        device: torch.device | None = None,
    ) -> "DetectionLUT":
        """Return a cached LUT for identical configurations; build once, share across instances.

        The LUT is read-only after construction so sharing is safe.
        """
        resolved_device = device or torch.device("cuda" if _CUDA_AVAILABLE else "cpu")
        cache_key = (
            freq_hz,
            tx_power_w,
            gain,
            max_range_m,
            snr_threshold_db,
            max_rcs,
            rcs_bins,
            dist_bins,
            str(resolved_device),
        )
        instance = _LUT_INSTANCE_CACHE.get(cache_key)
        if instance is None:
            instance = cls(
                freq_hz,
                tx_power_w,
                gain,
                max_range_m,
                snr_threshold_db,
                max_rcs=max_rcs,
                rcs_bins=rcs_bins,
                dist_bins=dist_bins,
                device=resolved_device,
            )
            _LUT_INSTANCE_CACHE[cache_key] = instance
        return instance
