"""Fighter Jet available thrust vs Mach number graphic (with afterburner)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.physics.aircraft import AircraftPhysics
from air_to_air_rl.physics.physics import get_speed_of_sound


def plot_aircraft_thrust() -> plt.Figure:
    """
    Plot Fighter Jet available thrust vs Mach number at 5000 m altitude.

    Shows both military (dry) and afterburner (wet) thrust using the actual
    AircraftPhysics.get_engine_force() implementation.

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

    # Compute thrust
    thrust_mil = np.zeros_like(mach)  # Military (dry) thrust
    thrust_ab = np.zeros_like(mach)  # Afterburner (wet) thrust

    for i, v_mps in enumerate(V_mps):
        # Military thrust (throttle at 0.5)
        thrust_mil[i] = aircraft.get_engine_force(v_mps, alt_m, throttle=0.5)
        # Afterburner thrust (throttle at 1.0)
        thrust_ab[i] = aircraft.get_engine_force(v_mps, alt_m, throttle=1.0)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Plot thrust curves
    ax.plot(
        mach,
        thrust_mil / 1000,
        color="red",
        linewidth=2.0,
        linestyle="--",
        label="Military Thrust (50%)",
    )
    ax.plot(mach, thrust_ab / 1000, color="blue", linewidth=2.5, label="Afterburner Thrust (100%)")

    # Set labels (no title)
    ax.set_xlabel("Mach Number $M$", fontsize=28)
    ax.set_ylabel("Available Thrust [kN]", fontsize=28)

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    ax.legend(fontsize=28, loc="upper left")

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_aircraft_thrust()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "thrust_aircraft.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
