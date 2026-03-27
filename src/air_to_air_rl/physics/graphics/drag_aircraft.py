"""Fighter Jet drag vs Mach number graphic."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.physics.aircraft import AircraftPhysics
from air_to_air_rl.physics.physics import get_speed_of_sound


def plot_aircraft_drag() -> plt.Figure:
    """
    Plot Fighter Jet drag components vs Mach number at 5000 m altitude.

    Uses the actual AircraftPhysics implementation for accurate drag computations.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create Fighter Jet physics model with generic fighter jet parameters
    # Fighter Jet specs: mass_kg=19700.0, reference_area_m2=78.0
    aircraft = AircraftPhysics(AircraftPhysics.Params(mass_kg=19_700.0, reference_area_m2=78.0))

    # Altitude for calculation
    alt_m = 5_000.0

    # Mach range
    mach = np.linspace(0.05, 2.1, 400)

    # Speed of sound at 5000 m
    a = get_speed_of_sound(alt_m)

    # Compute velocities
    V_mps = a * mach

    # Get air density at altitude
    rho = aircraft.air.get_density(alt_m)

    # Compute drag components
    D_parasite = np.zeros_like(mach)
    D_induced = np.zeros_like(mach)
    D_total = np.zeros_like(mach)

    W = aircraft.mass_kg * aircraft.g  # Weight in Newtons

    for i, v_mps in enumerate(V_mps):
        # Base drag coefficient
        cd0 = aircraft.get_base_drag_cd(v_mps, alt_m)

        # Parasite drag
        q = 0.5 * rho * v_mps**2  # Dynamic pressure
        D_parasite[i] = q * aircraft.A_m2 * cd0

        # Induced drag: CL = W / (0.5 * rho * V^2 * S)
        cl = W / (0.5 * rho * v_mps**2 * aircraft.A_m2) if v_mps > 1e-3 else 0.0
        D_induced[i] = q * aircraft.A_m2 * aircraft.K_ind * cl**2

        # Total drag
        D_total[i] = D_parasite[i] + D_induced[i]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Convert to kN
    D_parasite_kN = D_parasite / 1000.0
    D_induced_kN = D_induced / 1000.0
    D_total_kN = D_total / 1000.0

    # Plot drag components
    ax.plot(mach, D_parasite_kN, color="gray", linewidth=2.0, label=r"$D_{\text{parasite/wave}}$")
    ax.plot(mach, D_induced_kN, color="red", linewidth=2.0, label=r"$D_{\text{induced}}$")
    ax.plot(mach, D_total_kN, color="blue", linewidth=2.0, label=r"$D_{\text{total}}$")

    # Set labels (no title)
    ax.set_xlabel("Mach Number $M$", fontsize=28)
    ax.set_ylabel("Drag $D$ [kN]", fontsize=28)

    # Set y-axis limit
    ax.set_ylim([0, 500])

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    ax.legend(fontsize=28, loc="upper left")

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_aircraft_drag()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "drag_aircraft.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
