"""Air-to-Air Missile available thrust vs flight time and phase graphic."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_rl.missiles.core.phases import MissilePhaseManager


def plot_missile_thrust() -> plt.Figure:
    """
    Plot missile available thrust vs flight time, showing different phases.

    Uses the actual MissilePhaseManager to show realistic thrust profiles for
    different missile types through boost, middle (sustain), and terminal phases.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create different missile phase managers for comparison
    # Generic medium-range missile (AMRAAM-like): 8s motor burn
    generic_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 50.0},
        "middle": {"duration_s": 30.0, "thrust_kN": 45.0},
        "terminal": {"duration_s": 50.0, "thrust_kN": 0.0},
    }

    # Long-range missile (Meteor-like): 60s motor burn
    longrange_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 55.0},
        "middle": {"duration_s": 60.0, "thrust_kN": 40.0},
        "terminal": {"duration_s": 50.0, "thrust_kN": 0.0},
    }

    # Short-range missile (Sidewinder-like): 6s motor burn
    shortrange_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 65.0},
        "middle": {"duration_s": 6.0, "thrust_kN": 50.0},
        "terminal": {"duration_s": 30.0, "thrust_kN": 0.0},
    }

    # Time range for visualization
    time_s = np.linspace(0, 150, 1000)

    # Create phase managers for each missile type
    # Generic: 8s motor burn (boost 20s + middle 30s = 50s total before terminal)
    generic_mgr = MissilePhaseManager(generic_phases, life_time_s=160.0, motor_burn_s=50.0)

    # Long-range: 60s motor burn
    longrange_mgr = MissilePhaseManager(longrange_phases, life_time_s=160.0, motor_burn_s=80.0)

    # Short-range: 6s motor burn
    shortrange_mgr = MissilePhaseManager(shortrange_phases, life_time_s=80.0, motor_burn_s=26.0)

    # Compute thrust over time for each missile type
    thrust_generic = np.zeros_like(time_s)
    thrust_longrange = np.zeros_like(time_s)
    thrust_shortrange = np.zeros_like(time_s)

    for i, t in enumerate(time_s):
        generic_mgr.update(t)
        longrange_mgr.update(t)
        shortrange_mgr.update(t)

        thrust_generic[i] = generic_mgr.get_thrust_kN()
        thrust_longrange[i] = longrange_mgr.get_thrust_kN()
        thrust_shortrange[i] = shortrange_mgr.get_thrust_kN()

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Plot thrust curves for different missile types
    ax.plot(time_s, thrust_generic, color="blue", linewidth=2.5, label="Medium-Range (AMRAAM-like)")
    ax.plot(
        time_s,
        thrust_longrange,
        color="red",
        linewidth=2.0,
        linestyle="--",
        label="Long-Range (Meteor-like)",
    )
    ax.plot(
        time_s,
        thrust_shortrange,
        color="green",
        linewidth=2.0,
        linestyle=":",
        label="Short-Range (Sidewinder-like)",
    )

    # Add phase boundary annotations
    ax.axvline(x=20, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    ax.text(20, 60, "Boost→Middle", fontsize=24, rotation=90, va="top")

    ax.axvline(x=50, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    ax.text(50, 60, "Motor Burnout\n(Generic)", fontsize=24, rotation=90, va="top")

    # Set labels (no title)
    ax.set_xlabel("Flight Time [s]", fontsize=28)
    ax.set_ylabel("Available Thrust [kN]", fontsize=28)

    # Set y-axis limits
    ax.set_ylim([0, 70])

    # Grid
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    ax.legend(fontsize=28, loc="upper right")

    # Tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_missile_thrust()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, "thrust_missile.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()
