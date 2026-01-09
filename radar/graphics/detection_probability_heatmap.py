"""Detection probability heatmap for Eurofighter radar against a single Eurofighter."""
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
from aircrafts.types.eurofighter import Eurofighter
from radar.core.lut import DetectionLUT


def plot_detection_probability_heatmap() -> plt.Figure:
    """
    Plot detection probability heatmap for Eurofighter radar against a single target Eurofighter.

    Creates a 2D heatmap showing detection probability as a function of:
    - Distance (x-axis): 0 to 100 km
    - RCS (y-axis): 0 to 3.5 m² (Eurofighter RCS range)

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create Eurofighter instance to get radar parameters
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    map_limits = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}

    eurofighter = Eurofighter(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0
    )

    # Get Eurofighter radar parameters from config dict
    cfg = eurofighter.config

    # Create Detection LUT using Eurofighter radar parameters
    lut = DetectionLUT(
        freq_hz=cfg.get("radar_frequency_hz", 10e9),
        tx_power_w=cfg.get("radar_tx_power_w", 18e3),
        gain=10 ** (cfg.get("radar_antenna_gain_db", 36.0) / 10),  # Convert dB to linear
        max_range_m=100_000.0,  # 100 km as specified
        snr_threshold_db=cfg.get("radar_snr_threshold_db", 9.0),
        max_rcs=3.5,  # Maximum RCS for Eurofighter
        rcs_bins=256,
        dist_bins=256
    )

    # Eurofighter RCS values (based on the aircraft configuration)
    eurofighter_rcs_min = 0.10
    eurofighter_rcs_nominal = 3.0

    # Create 2D grid for heatmap
    distances_km = np.linspace(0, 100, 256)
    distances_m = distances_km * 1000
    rcs_values = np.linspace(0, 3.5, 256)

    # Create meshgrid
    dist_mesh, rcs_mesh = np.meshgrid(distances_m, rcs_values, indexing='ij')

    # Calculate detection probability for each point
    prob_mesh = np.zeros_like(dist_mesh)
    for i in range(dist_mesh.shape[0]):
        for j in range(dist_mesh.shape[1]):
            prob_mesh[i, j] = lut.get_probability(dist_mesh[i, j], rcs_mesh[i, j])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Plot heatmap
    im = ax.contourf(distances_km, rcs_values, prob_mesh.T, levels=20, cmap='RdYlGn')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detection Probability', fontsize=24)

    # Mark Eurofighter nominal RCS
    ax.axhline(y=eurofighter_rcs_nominal, color='blue', linestyle='--',
               linewidth=2, label=f'Eurofighter RCS (nominal: {eurofighter_rcs_nominal:.1f} m²)')

    # Labels (no title)
    ax.set_xlabel('Distance [km]', fontsize=28)
    ax.set_ylabel('Radar Cross Section (RCS) [m²]', fontsize=28)
    ax.legend(fontsize=28, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()

    # Print radar parameters for reference
    print(f"\nRadar Parameters:")
    print(f"  Frequency: {cfg.get('radar_frequency_hz', 10e9) / 1e9:.1f} GHz")
    print(f"  TX Power: {cfg.get('radar_tx_power_w', 18e3) / 1e3:.1f} kW")
    print(f"  Antenna Gain: {cfg.get('radar_antenna_gain_db', 36.0):.1f} dB")
    print(f"  SNR Threshold: {cfg.get('radar_snr_threshold_db', 9.0):.1f} dB")
    print(f"  Max Range: 100 km")
    print(f"\nEurofighter RCS:")
    print(f"  Nominal: {eurofighter_rcs_nominal:.2f} m²")
    print(f"  Minimum: {eurofighter_rcs_min:.2f} m²")

    return fig


if __name__ == '__main__':
    fig = plot_detection_probability_heatmap()

    # Save to radar/graphics folder
    graphics_dir = os.path.dirname(__file__)

    # Save as PNG
    png_path = os.path.join(graphics_dir, 'detection_probability_heatmap.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved PNG to {png_path}')

    plt.show()
