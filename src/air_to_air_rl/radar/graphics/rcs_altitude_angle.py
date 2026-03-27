"""Radar Cross Section (RCS) vs altitude angle (elevation angle) for multiple aircraft types."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.aircrafts.types.eurofighter import Eurofighter
from air_to_air_rl.aircrafts.types.f22 import F22
from air_to_air_rl.aircrafts.types.f35 import F35
from air_to_air_rl.aircrafts.types.su57 import Su57
from air_to_air_rl.radar.core.utils import _effective_rcs
from air_to_air_rl.simulator.core.helpers import Position


def plot_rcs_altitude_angle() -> plt.Figure:
    """
    Plot RCS vs altitude angle (elevation angle) for multiple aircraft types.

    Shows how RCS varies when observed from above (positive angles) vs below (negative angles),
    demonstrating dorsal (top) vs ventral (bottom) sensitivity.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create aircraft instances at a reference position
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    map_limits = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}

    # Create one instance of each aircraft type
    # F22, F35, Eurofighter: use (position, yaw, speed, group, map_limits, min_alt, max_alt)
    f22 = F22(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    f35 = F35(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    eurofighter = Eurofighter(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=map_limits,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    # Su57: use different constructor signature
    su57 = Su57(position=ref_pos, yaw_deg=0.0, speed_mps=250.0, group="red", map_limits=map_limits)

    # Altitude angles from -90° (looking from below) to +90° (looking from above)
    altitude_angles_deg = np.linspace(-90, 90, 181)

    # Calculate RCS for each aircraft at each altitude angle
    # For simplicity, we fix azimuth at 0° (nose-on) to isolate elevation dependency
    rcs_f22 = np.zeros_like(altitude_angles_deg)
    rcs_f35 = np.zeros_like(altitude_angles_deg)
    rcs_su57 = np.zeros_like(altitude_angles_deg)
    rcs_eurofighter = np.zeros_like(altitude_angles_deg)

    for i, el_angle in enumerate(altitude_angles_deg):
        # Calculate horizontal distance (100 km away)
        horiz_dist = 100000.0
        vert_dist = horiz_dist * np.tan(np.radians(el_angle))

        # Radar position relative to target
        radar_pos = Position(
            lat=ref_pos.lat + (horiz_dist / 111000.0), lon=ref_pos.lon, alt=ref_pos.alt + vert_dist
        )

        rcs_f22[i] = _effective_rcs(f22, radar_pos)
        rcs_f35[i] = _effective_rcs(f35, radar_pos)
        rcs_su57[i] = _effective_rcs(su57, radar_pos)
        rcs_eurofighter[i] = _effective_rcs(eurofighter, radar_pos)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Plot RCS curves for different aircraft types
    ax.plot(
        altitude_angles_deg,
        rcs_f22,
        color="blue",
        linewidth=2.5,
        label=f"F-22 (σ₀={f22.rcs:.4f} m²)",
    )
    ax.plot(
        altitude_angles_deg,
        rcs_f35,
        color="cyan",
        linewidth=2.0,
        linestyle="--",
        label=f"F-35 (σ₀={f35.rcs:.4f} m²)",
    )
    ax.plot(
        altitude_angles_deg,
        rcs_su57,
        color="red",
        linewidth=2.0,
        linestyle="-.",
        label=f"Su-57 (σ₀={su57.rcs:.4f} m²)",
    )
    ax.plot(
        altitude_angles_deg,
        rcs_eurofighter,
        color="green",
        linewidth=2.0,
        linestyle=":",
        label=f"Eurofighter (σ₀={eurofighter.rcs:.4f} m²)",
    )

    # Add reference lines for dorsal/ventral boundary
    ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    # Set labels (no title)
    ax.set_xlabel("Altitude Angle / Elevation Angle [deg]", fontsize=28)
    ax.set_ylabel("Radar Cross Section (RCS) [m²]", fontsize=28)

    # Set x-axis limits
    ax.set_xlim([-90, 90])

    # Log scale for RCS to better show variation across range
    ax.set_yscale("log")

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Legend with background
    ax.legend(fontsize=28, loc="upper left", framealpha=0.9)

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_rcs_altitude_angle()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "rcs_altitude_angle.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
