"""Missile turn rate vs Mach number graphic."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.physics.missiles import MissilePhysics
from air_to_air_rl.physics.physics import get_speed_of_sound


def plot_missile_turn_rate() -> plt.Figure:
    """
    Plot missile turn rate vs Mach number at 5000 m altitude.

    Uses the actual MissilePhysics implementation for instantaneous turn rate
    computations.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create missile physics model with generic missile parameters
    # Generic air-to-air missile: mass_kg=300.0, reference_area_m2=0.08
    # (Representative of medium-range missiles like AMRAAM family)
    missile = MissilePhysics(MissilePhysics.Params(mass_kg=300.0, reference_area_m2=0.08))

    # Altitude for calculation
    alt_m = 5_000.0

    # Mach range
    mach = np.linspace(0.15, 5.0, 400)

    # Speed of sound at 5000 m
    a = get_speed_of_sound(alt_m)

    # Compute velocities
    V_mps = a * mach

    # Compute turn rates
    turn_rate_inst = np.zeros_like(mach)
    turn_rate_ca_max = np.zeros_like(mach)
    turn_rate_n_max = np.zeros_like(mach)
    turn_rate_comp = np.zeros_like(mach)

    W = missile.mass_kg * missile.g  # Weight in Newtons
    g = missile.g

    for i, v_mps in enumerate(V_mps):
        if v_mps < 1e-3:
            continue

        rho = missile.air.get_density(alt_m)
        q = 0.5 * rho * v_mps**2

        # Get CA_max (axial force coefficient used for maneuverability)
        ca_max = missile.get_ca_max(v_mps, alt_m)

        # Load factor from CA_max
        # Note: CA is axial coefficient (side force), different from CL (lift)
        n_ca = (q * missile.A_m2 * ca_max) / W

        if n_ca >= 1.0:
            omega_ca = (g / v_mps) * math.sqrt(max(0, n_ca**2 - 1.0))
            turn_rate_ca_max[i] = math.degrees(omega_ca)
        else:
            turn_rate_ca_max[i] = 0.0

        # Turn rate limited by n_max structural limit
        omega_nmax = (g / v_mps) * math.sqrt(max(0, missile.n_max**2 - 1.0))
        turn_rate_n_max[i] = math.degrees(omega_nmax)

        # Instantaneous turn rate (minimum of CA-limited and structure-limited)
        turn_rate_inst[i] = min(turn_rate_ca_max[i], turn_rate_n_max[i])

        # Actual turn rate (same as instantaneous for missiles)
        turn_rate_comp[i] = turn_rate_inst[i]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Plot components
    ax.plot(
        mach,
        turn_rate_ca_max,
        color="green",
        linewidth=2.0,
        linestyle="--",
        label=r"$\dot\chi(C_A^{\max})$",
    )
    ax.plot(
        mach,
        turn_rate_n_max,
        color="red",
        linewidth=2.0,
        linestyle="--",
        label=r"$\dot\chi(n_{\max} = 30)$",
    )
    ax.plot(mach, turn_rate_comp, color="blue", linewidth=2.5, label="Instantaneous Turn Rate")

    # Set labels (no title)
    ax.set_xlabel("Mach Number $M$", fontsize=28)
    ax.set_ylabel(r"Turn Rate $\dot{\chi}$ [deg/s]", fontsize=28)

    # Set y-axis limits
    ax.set_ylim([0, 40])

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    ax.legend(fontsize=28, loc="upper right")

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_missile_turn_rate()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "turn_rate_missile.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
