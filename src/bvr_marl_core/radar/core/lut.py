import math

import numpy as np

_LUT_INSTANCE_CACHE: dict = {}


class DetectionLUT:
    """Builds and queries a lookup table for detection probability.

    Constructed using pure NumPy — no PyTorch dependency.
    All query-time access uses fast bilinear interpolation on the pre-built array.
    """

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
        processing_gain_db: float = 0.0,
    ):
        self.freq_hz = freq_hz
        self.tx_power_w = tx_power_w
        self.gain = gain
        self.max_range_m = max_range_m
        self.snr_threshold_db = snr_threshold_db
        # Coherent pulse-integration + Doppler-processing gain (dB). The single-pulse
        # radar equation below omits the SNR improvement a pulse-Doppler radar earns
        # by integrating a whole coherent processing interval (typically 20-30 dB).
        # Without it, R^4 falloff collapses effective detection to a small fraction
        # of the antenna's real range. Applied as a flat SNR multiplier.
        self.processing_gain_db = processing_gain_db
        self._max_rcs = max_rcs
        self._build_lut(rcs_bins, dist_bins)

    # Lowest RCS resolved by the table (m^2). VLO fighters (e.g. F-22 ~1e-4 m^2)
    # sit far below the old 0.01 floor, so a linear grid clamped them all to 0.01
    # and erased the stealth advantage. RCS spans 5+ orders of magnitude, so the
    # axis is sampled logarithmically between this floor and max_rcs.
    _RCS_FLOOR = 1e-5

    def _build_lut(self, rcs_bins: int, dist_bins: int):
        wavelength = 3e8 / self.freq_hz
        distances = np.linspace(1e-1, self.max_range_m, dist_bins)
        rcs_values = np.logspace(math.log10(self._RCS_FLOOR), math.log10(self._max_rcs), rcs_bins)

        self._dmin = float(distances[0])
        self._dmax = float(distances[-1])
        # RCS axis is log-spaced; store the log bounds used for normalisation.
        self._log_rmin = math.log10(float(rcs_values[0]))
        self._log_rmax = math.log10(float(rcs_values[-1]))
        self._rmin = float(rcs_values[0])
        self._rmax = float(rcs_values[-1])
        self._lut_H = dist_bins
        self._lut_W = rcs_bins

        D, S = np.meshgrid(distances, rcs_values, indexing="ij")

        k = 1.38e-23
        T0 = 290
        B = 1e6
        Fn = 3
        Ls = 2

        snr_lin = (self.tx_power_w * (self.gain**2) * wavelength**2 * S) / (
            (4 * math.pi) ** 3 * D**4 * k * T0 * B * Fn * Ls + 1e-12
        )
        # Add coherent integration / Doppler-processing gain (flat dB offset).
        snr_db = 10.0 * np.log10(snr_lin + 1e-12) + self.processing_gain_db

        # Sigmoid detection probability: sigma((SNR_dB - threshold) / 2)
        x = (snr_db - self.snr_threshold_db) / 2.0
        self._lut_np: np.ndarray = 1.0 / (1.0 + np.exp(-x))  # shape: (dist_bins, rcs_bins)

    def get_probability(self, dist: float, rcs: float) -> float:
        """Interpolate detection probability from the LUT for a given dist and rcs.

        Uses bilinear interpolation on the pre-built NumPy array (align_corners=True
        convention, matching the original grid_sample behaviour).
        """
        # RCS axis is log-spaced: normalise on log10(rcs), flooring vanishingly
        # small / non-positive values at the low bound and clamping high-RCS
        # targets to the final column while still interpolating over range.
        log_rcs = math.log10(min(max(rcs, self._RCS_FLOOR), self._max_rcs))
        dn = max(-1.0, min(1.0, 2.0 * (dist - self._dmin) / (self._dmax - self._dmin) - 1.0))
        rn = max(
            -1.0,
            min(1.0, 2.0 * (log_rcs - self._log_rmin) / (self._log_rmax - self._log_rmin) - 1.0),
        )

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
        device=None,  # Accepted for backwards compatibility with callers; ignored.
        processing_gain_db: float = 0.0,
    ) -> "DetectionLUT":
        """Return a cached LUT for identical configurations; build once, share across instances."""
        cache_key = (
            freq_hz,
            tx_power_w,
            gain,
            max_range_m,
            snr_threshold_db,
            max_rcs,
            rcs_bins,
            dist_bins,
            processing_gain_db,
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
                processing_gain_db=processing_gain_db,
            )
            _LUT_INSTANCE_CACHE[cache_key] = instance
        return instance
