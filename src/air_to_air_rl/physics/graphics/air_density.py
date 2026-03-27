"""Air density vs altitude graphic based on ISA model."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.physics.physics import AirLayer


def plot_air_density() -> plt.Figure:
    """
    Plot air density vs altitude according to ISA model.

    Uses the actual AirLayer implementation from the physics module.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create air layer (uses ISA model)
    air = AirLayer()

    # Create altitude range (0 to 25000 m)
    altitude_m = np.linspace(0, 25_000, 300)

    # Get density for each altitude using actual physics implementation
    density_kg_m3 = np.array([air.get_density(alt) for alt in altitude_m])

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot density
    ax.plot(altitude_m, density_kg_m3, color="blue", linewidth=2.0)

    # Set labels and title
    ax.set_xlabel("Altitude [m]", fontsize=28)
    ax.set_ylabel("Density [kg/m³]", fontsize=28)
    ax.set_title("Air Density vs Altitude", fontsize=32, fontweight="bold")

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_air_density()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "air_density.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
