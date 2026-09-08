"""Search-sector dwell/revisit scheduling for operational radars."""

from __future__ import annotations

from dataclasses import dataclass

from bvr_marl_core.simulator.utils.angles import signed_yaw_deg_diff


@dataclass(frozen=True, slots=True)
class Dwell:
    sequence: int
    center_azimuth_offset_deg: float
    center_elevation_offset_deg: float
    horizontal_width_deg: float
    vertical_width_deg: float
    duration_s: float
    revisit_interval_s: float


class ScanScheduler:
    """Round-robin search sectors with an explicit dwell and revisit interval."""

    def __init__(
        self,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        sectors: int = 4,
        dwell_duration_s: float = 1.0,
    ):
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.sectors = max(1, int(sectors))
        self.dwell_duration_s = max(1e-6, float(dwell_duration_s))
        self.sequence = 0
        self._elapsed_s = 0.0

    def next_dwell(self, duration_s: float) -> Dwell:
        duration_s = max(0.0, float(duration_s))
        self._elapsed_s += duration_s
        complete = self._elapsed_s + 1e-12 >= self.dwell_duration_s
        evaluated_duration_s = self.dwell_duration_s if complete else 0.0
        width = self.horizontal_fov_deg / self.sectors
        index = self.sequence % self.sectors
        center = -self.horizontal_fov_deg / 2.0 + width * (index + 0.5)
        dwell = Dwell(
            sequence=self.sequence,
            center_azimuth_offset_deg=center,
            center_elevation_offset_deg=0.0,
            horizontal_width_deg=width,
            vertical_width_deg=self.vertical_fov_deg,
            duration_s=evaluated_duration_s,
            revisit_interval_s=self.dwell_duration_s * self.sectors,
        )
        if complete:
            self._elapsed_s = max(0.0, self._elapsed_s - self.dwell_duration_s)
            self.sequence += 1
        return dwell

    @staticmethod
    def admits(target_azimuth_deg: float, target_elevation_deg: float, dwell: Dwell) -> bool:
        azimuth_error = abs(
            signed_yaw_deg_diff(dwell.center_azimuth_offset_deg, target_azimuth_deg)
        )
        elevation_error = abs(float(target_elevation_deg) - dwell.center_elevation_offset_deg)
        return (
            azimuth_error <= dwell.horizontal_width_deg / 2.0
            and elevation_error <= dwell.vertical_width_deg / 2.0
        )
