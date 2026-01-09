"""Radar Cross Section (RCS) vs horizontal angle (azimuth angle) for multiple aircraft types."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from simulator.core.helpers import Position
from aircrafts.types.f22 import F22
from aircrafts.types.f35 import F35
from aircrafts.types.su57 import Su57
from aircrafts.types.eurofighter import Eurofighter
from radar.core.utils import _effective_rcs


def plot_rcs_horizontal_angle() -> plt.Figure:
    """
    Plot RCS vs horizontal angle (azimuth angle) for multiple aircraft types using Cartesian plot.

    Shows how RCS varies around the aircraft (0° nose-on, 90° beam, 180° tail-on, 270° beam opposite),
    demonstrating front/beam/tail characteristics and the classic "bow-tie" pattern of low nose-on
    and tail-on RCS but high beam RCS.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create aircraft instances at a reference position
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    map_limits = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}

    # Create one instance of each aircraft type
    # F22, F35, Eurofighter: use (position, yaw, speed, group, map_limits, min_alt, max_alt)
    f22 = F22(position=ref_pos, yaw_deg=0.0, speed_mps=250.0, group="blue",
              map_limits=map_limits, min_alt_m=0.0, max_alt_m=20000.0)

    f35 = F35(position=ref_pos, yaw_deg=0.0, speed_mps=250.0, group="blue",
              map_limits=map_limits, min_alt_m=0.0, max_alt_m=20000.0)

    eurofighter = Eurofighter(position=ref_pos, yaw_deg=0.0, speed_mps=250.0,
                             group="blue", map_limits=map_limits, min_alt_m=0.0, max_alt_m=20000.0)

    # Su57: use different constructor signature
    su57 = Su57(position=ref_pos, yaw_deg=0.0, speed_mps=250.0, group="red", map_limits=map_limits)

    # Azimuth angles from 0° to 360°
    azimuth_angles_deg = np.linspace(0, 360, 361)

    # Calculate RCS for each aircraft at each azimuth angle
    rcs_f22 = np.zeros_like(azimuth_angles_deg)
    rcs_f35 = np.zeros_like(azimuth_angles_deg)
    rcs_su57 = np.zeros_like(azimuth_angles_deg)
    rcs_eurofighter = np.zeros_like(azimuth_angles_deg)

    for i, az_angle in enumerate(azimuth_angles_deg):
        # Calculate radar position at given azimuth, 100 km away, same altitude
        horiz_dist = 100000.0
        radar_lat = ref_pos.lat + (horiz_dist / 111000.0) * np.cos(np.radians(az_angle))
        radar_lon = ref_pos.lon + (horiz_dist / 111000.0 / np.cos(np.radians(ref_pos.lat))) * np.sin(np.radians(az_angle))

        radar_pos = Position(
            lat=radar_lat,
            lon=radar_lon,
            alt=ref_pos.alt
        )

        rcs_f22[i] = _effective_rcs(f22, radar_pos)
        rcs_f35[i] = _effective_rcs(f35, radar_pos)
        rcs_su57[i] = _effective_rcs(su57, radar_pos)
        rcs_eurofighter[i] = _effective_rcs(eurofighter, radar_pos)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Plot RCS curves
    ax.plot(azimuth_angles_deg, rcs_f22, color='blue', linewidth=2.5,
            label=f'F-22 (σ₀={f22.rcs:.4f} m²)')
    ax.plot(azimuth_angles_deg, rcs_f35, color='cyan', linewidth=2.0, linestyle='--',
            label=f'F-35 (σ₀={f35.rcs:.4f} m²)')
    ax.plot(azimuth_angles_deg, rcs_su57, color='red', linewidth=2.0, linestyle='-.',
            label=f'Su-57 (σ₀={su57.rcs:.4f} m²)')
    ax.plot(azimuth_angles_deg, rcs_eurofighter, color='green', linewidth=2.0, linestyle=':',
            label=f'Eurofighter (σ₀={eurofighter.rcs:.4f} m²)')

    # Add reference lines
    ax.axvline(x=0.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(x=90.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(x=180.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(x=270.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    # Set labels (no title)
    ax.set_xlabel('Azimuth Angle [deg] (0°=Nose, 90°=Right Beam, 180°=Tail, 270°=Left Beam)', fontsize=28)
    ax.set_ylabel('Radar Cross Section (RCS) [m²]', fontsize=28)

    # Set x-axis limits
    ax.set_xlim([0, 360])
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

    # Log scale for RCS
    ax.set_yscale('log')

    # Grid
    ax.grid(True, which='both', alpha=0.3)

    # Legend with background
    ax.legend(fontsize=28, loc='upper left', framealpha=0.9)

    # Tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = plot_rcs_horizontal_angle()

    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graphics')
    os.makedirs(graphics_dir, exist_ok=True)

    save_path = os.path.join(graphics_dir, 'rcs_horizontal_angle.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')

    plt.show()
