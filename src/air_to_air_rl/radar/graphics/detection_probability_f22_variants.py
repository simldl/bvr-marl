"""Detection probability heatmaps for F-22 radar against F-22 and Eurofighter targets."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.radar.core.lut import DetectionLUT
from air_to_air_rl.simulator.core.helpers import Position


def plot_f22_vs_f22() -> plt.Figure:
    """
    Plot detection probability heatmap for F-22 radar against F-22 target.

    Creates a 2D heatmap showing detection probability as a function of:
    - Distance (x-axis): 0 to 100 km
    - RCS (y-axis): 0 to 1.5 m² (F-22 RCS range)

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create F-22 instance to get radar parameters
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    map_limits = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}

    f22 = F22(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    # Get F-22 radar parameters from config dict
    cfg = f22.config

    # Create Detection LUT using F-22 radar parameters
    lut = DetectionLUT(
        freq_hz=cfg.get("radar_frequency_hz", 10e9),
        tx_power_w=cfg.get("radar_tx_power_w", 5e3),
        gain=10 ** (cfg.get("radar_antenna_gain_db", 34.0) / 10),  # Convert dB to linear
        max_range_m=100_000.0,  # 100 km as specified
        snr_threshold_db=cfg.get("radar_snr_threshold_db", 10.0),
        max_rcs=1.5,  # Maximum RCS for F-22
        rcs_bins=256,
        dist_bins=256,
    )

    # F-22 RCS values (based on the aircraft configuration)
    f22_rcs_min = 0.003
    f22_rcs_nominal = 0.0069  # Target RCS value

    # Create 2D grid for heatmap
    distances_km = np.linspace(0, 100, 256)
    distances_m = distances_km * 1000
    rcs_values = np.linspace(0, 1.5, 256)

    # Create meshgrid
    dist_mesh, rcs_mesh = np.meshgrid(distances_m, rcs_values, indexing="ij")

    # Calculate detection probability for each point
    prob_mesh = np.zeros_like(dist_mesh)
    for i in range(dist_mesh.shape[0]):
        for j in range(dist_mesh.shape[1]):
            prob_mesh[i, j] = lut.get_probability(dist_mesh[i, j], rcs_mesh[i, j])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Full heatmap
    im1 = ax1.contourf(distances_km, rcs_values, prob_mesh.T, levels=20, cmap="RdYlGn")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Detection Probability", fontsize=24)

    # Add contour lines
    contours = ax1.contour(
        distances_km,
        rcs_values,
        prob_mesh.T,
        levels=[0.5, 0.9],
        colors="black",
        linewidths=1.5,
        alpha=0.5,
    )
    ax1.clabel(contours, inline=True, fontsize=24, fmt="P=%.1f")

    # Mark F-22 nominal RCS
    ax1.axhline(
        y=f22_rcs_nominal,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"F-22 RCS (nominal: {f22_rcs_nominal:.4f} m²)",
    )

    # Labels and title
    ax1.set_xlabel("Distance [km]", fontsize=28)
    ax1.set_ylabel("Radar Cross Section (RCS) [m²]", fontsize=28)
    ax1.set_title(
        "Detection Probability Heatmap\n(F-22 Radar vs F-22 Target)", fontsize=32, fontweight="bold"
    )
    ax1.legend(fontsize=24)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", which="major", labelsize=24)

    # Subplot 2: Zoomed in on F-22 RCS range
    zoom_rcs_min = f22_rcs_min
    zoom_rcs_max = f22_rcs_nominal * 100

    # Create zoomed grid
    rcs_zoom = np.linspace(zoom_rcs_min, zoom_rcs_max, 256)
    dist_zoom_mesh, rcs_zoom_mesh = np.meshgrid(distances_m, rcs_zoom, indexing="ij")

    prob_zoom = np.zeros_like(dist_zoom_mesh)
    for i in range(dist_zoom_mesh.shape[0]):
        for j in range(dist_zoom_mesh.shape[1]):
            prob_zoom[i, j] = lut.get_probability(dist_zoom_mesh[i, j], rcs_zoom_mesh[i, j])

    im2 = ax2.contourf(distances_km, rcs_zoom, prob_zoom.T, levels=20, cmap="RdYlGn")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Detection Probability", fontsize=24)

    # Add contour lines
    contours2 = ax2.contour(
        distances_km,
        rcs_zoom,
        prob_zoom.T,
        levels=[0.5, 0.9],
        colors="black",
        linewidths=1.5,
        alpha=0.5,
    )
    ax2.clabel(contours2, inline=True, fontsize=24, fmt="P=%.1f")

    # Mark F-22 nominal RCS
    ax2.axhline(
        y=f22_rcs_nominal,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"F-22 RCS (nominal: {f22_rcs_nominal:.4f} m²)",
    )

    # Labels and title
    ax2.set_xlabel("Distance [km]", fontsize=28)
    ax2.set_ylabel("Radar Cross Section (RCS) [m²]", fontsize=28)
    ax2.set_title(
        "Detection Probability Heatmap (Zoomed)\n(F-22 RCS Range)", fontsize=32, fontweight="bold"
    )
    ax2.legend(fontsize=24)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()

    # Print radar parameters for reference
    print("\nF-22 vs F-22 Radar Parameters:")
    print(f"  Frequency: {cfg.get('radar_frequency_hz', 10e9) / 1e9:.1f} GHz")
    print(f"  TX Power: {cfg.get('radar_tx_power_w', 5e3) / 1e3:.1f} kW")
    print(f"  Antenna Gain: {cfg.get('radar_antenna_gain_db', 34.0):.1f} dB")
    print(f"  SNR Threshold: {cfg.get('radar_snr_threshold_db', 10.0):.1f} dB")
    print("  Max Range: 100 km")
    print("\nF-22 RCS:")
    print(f"  Nominal: {f22_rcs_nominal:.4f} m²")
    print(f"  Minimum: {f22_rcs_min:.4f} m²")

    return fig


def plot_f22_vs_eurofighter() -> plt.Figure:
    """
    Plot detection probability heatmap for F-22 radar against Eurofighter target.

    Creates a 2D heatmap showing detection probability as a function of:
    - Distance (x-axis): 0 to 100 km
    - RCS (y-axis): 0 to 3.5 m² (Eurofighter RCS range)

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create F-22 instance to get radar parameters
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    map_limits = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}

    f22 = F22(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    # Get F-22 radar parameters from config dict
    cfg = f22.config

    # Create Detection LUT using F-22 radar parameters
    lut = DetectionLUT(
        freq_hz=cfg.get("radar_frequency_hz", 10e9),
        tx_power_w=cfg.get("radar_tx_power_w", 5e3),
        gain=10 ** (cfg.get("radar_antenna_gain_db", 34.0) / 10),  # Convert dB to linear
        max_range_m=100_000.0,  # 100 km as specified
        snr_threshold_db=cfg.get("radar_snr_threshold_db", 10.0),
        max_rcs=3.5,  # Maximum RCS for Eurofighter
        rcs_bins=256,
        dist_bins=256,
    )

    # Eurofighter RCS values (based on the aircraft configuration)
    eurofighter_rcs_min = 0.10
    eurofighter_rcs_nominal = 3.0

    # Create 2D grid for heatmap
    distances_km = np.linspace(0, 100, 256)
    distances_m = distances_km * 1000
    rcs_values = np.linspace(0, 3.5, 256)

    # Create meshgrid
    dist_mesh, rcs_mesh = np.meshgrid(distances_m, rcs_values, indexing="ij")

    # Calculate detection probability for each point
    prob_mesh = np.zeros_like(dist_mesh)
    for i in range(dist_mesh.shape[0]):
        for j in range(dist_mesh.shape[1]):
            prob_mesh[i, j] = lut.get_probability(dist_mesh[i, j], rcs_mesh[i, j])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Full heatmap
    im1 = ax1.contourf(distances_km, rcs_values, prob_mesh.T, levels=20, cmap="RdYlGn")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Detection Probability", fontsize=24)

    # Add contour lines
    contours = ax1.contour(
        distances_km,
        rcs_values,
        prob_mesh.T,
        levels=[0.5, 0.9],
        colors="black",
        linewidths=1.5,
        alpha=0.5,
    )
    ax1.clabel(contours, inline=True, fontsize=24, fmt="P=%.1f")

    # Mark Eurofighter nominal RCS
    ax1.axhline(
        y=eurofighter_rcs_nominal,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Eurofighter RCS (nominal: {eurofighter_rcs_nominal:.1f} m²)",
    )

    # Labels and title
    ax1.set_xlabel("Distance [km]", fontsize=11)
    ax1.set_ylabel("Radar Cross Section (RCS) [m²]", fontsize=11)
    ax1.set_title(
        "Detection Probability Heatmap\n(F-22 Radar vs Eurofighter Target)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Zoomed in on Eurofighter RCS range
    zoom_rcs_min = eurofighter_rcs_min
    zoom_rcs_max = eurofighter_rcs_nominal * 1.2

    # Create zoomed grid
    rcs_zoom = np.linspace(zoom_rcs_min, zoom_rcs_max, 256)
    dist_zoom_mesh, rcs_zoom_mesh = np.meshgrid(distances_m, rcs_zoom, indexing="ij")

    prob_zoom = np.zeros_like(dist_zoom_mesh)
    for i in range(dist_zoom_mesh.shape[0]):
        for j in range(dist_zoom_mesh.shape[1]):
            prob_zoom[i, j] = lut.get_probability(dist_zoom_mesh[i, j], rcs_zoom_mesh[i, j])

    im2 = ax2.contourf(distances_km, rcs_zoom, prob_zoom.T, levels=20, cmap="RdYlGn")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Detection Probability", fontsize=24)

    # Add contour lines
    contours2 = ax2.contour(
        distances_km,
        rcs_zoom,
        prob_zoom.T,
        levels=[0.5, 0.9],
        colors="black",
        linewidths=1.5,
        alpha=0.5,
    )
    ax2.clabel(contours2, inline=True, fontsize=24, fmt="P=%.1f")

    # Mark Eurofighter nominal RCS
    ax2.axhline(
        y=eurofighter_rcs_nominal,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Eurofighter RCS (nominal: {eurofighter_rcs_nominal:.1f} m²)",
    )

    # Labels and title
    ax2.set_xlabel("Distance [km]", fontsize=11)
    ax2.set_ylabel("Radar Cross Section (RCS) [m²]", fontsize=11)
    ax2.set_title(
        "Detection Probability Heatmap (Zoomed)\n(Eurofighter RCS Range)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print radar parameters for reference
    print("\nF-22 vs Eurofighter Radar Parameters:")
    print(f"  Frequency: {cfg.get('radar_frequency_hz', 10e9) / 1e9:.1f} GHz")
    print(f"  TX Power: {cfg.get('radar_tx_power_w', 5e3) / 1e3:.1f} kW")
    print(f"  Antenna Gain: {cfg.get('radar_antenna_gain_db', 34.0):.1f} dB")
    print(f"  SNR Threshold: {cfg.get('radar_snr_threshold_db', 10.0):.1f} dB")
    print("  Max Range: 100 km")
    print("\nEurofighter RCS:")
    print(f"  Nominal: {eurofighter_rcs_nominal:.2f} m²")
    print(f"  Minimum: {eurofighter_rcs_min:.2f} m²")

    return fig


if __name__ == "__main__":
    # Generate F-22 vs F-22 heatmap
    print("\n" + "=" * 60)
    print("Generating F-22 vs F-22 heatmap...")
    print("=" * 60)
    fig1 = plot_f22_vs_f22()

    graphics_dir = os.path.dirname(__file__)
    png_path1 = os.path.join(graphics_dir, "detection_probability_f22_vs_f22.png")
    plt.savefig(png_path1, dpi=150, bbox_inches="tight")
    print(f"Saved PNG to {png_path1}")
    plt.close(fig1)

    # Generate F-22 vs Eurofighter heatmap
    print("\n" + "=" * 60)
    print("Generating F-22 vs Eurofighter heatmap...")
    print("=" * 60)
    fig2 = plot_f22_vs_eurofighter()

    png_path2 = os.path.join(graphics_dir, "detection_probability_f22_vs_eurofighter.png")
    plt.savefig(png_path2, dpi=150, bbox_inches="tight")
    print(f"Saved PNG to {png_path2}")
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("All heatmaps generated successfully!")
    print("=" * 60)
