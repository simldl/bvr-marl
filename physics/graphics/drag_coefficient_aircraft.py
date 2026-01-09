"""Fighter Jet parasite drag coefficient (CD0) vs Mach number graphic."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from physics.aircraft import AircraftPhysics
from physics.physics import get_speed_of_sound


def plot_aircraft_cd0() -> plt.Figure:
    """
    Plot Fighter Jet parasite drag coefficient vs Mach number at 5000 m altitude.

    Uses the actual AircraftPhysics.get_base_drag_cd() implementation.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Create Fighter Jet physics model with generic fighter jet parameters
    # Fighter Jet specs: mass_kg=19700.0, reference_area_m2=78.0
    aircraft = AircraftPhysics(AircraftPhysics.Params(
        mass_kg=19_700.0,
        reference_area_m2=78.0
    ))

    # Altitude for calculation
    alt_m = 5_000.0

    # Mach range
    mach = np.linspace(0.05, 2.1, 400)

    # Speed of sound at 5000 m
    a = get_speed_of_sound(alt_m)

    # Compute velocities
    V_mps = a * mach

    # Compute drag coefficients
    cd0 = np.zeros_like(mach)

    for i, v_mps in enumerate(V_mps):
        # Get base drag coefficient from physics implementation
        cd0[i] = aircraft.get_base_drag_cd(v_mps, alt_m)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Plot CD0
    ax.plot(mach, cd0, color='blue', linewidth=2.5)

    # Set labels and title
    ax.set_xlabel('Mach Number $M$', fontsize=28)
    ax.set_ylabel('Parasite Drag Coefficient $C_{D0}$', fontsize=28)
    ax.set_title('Fighter Jet at 5000 m', fontsize=32, fontweight='bold')

    # Grid
    ax.grid(True, which='both', alpha=0.3)

    # Legend
    # ax.legend(fontsize=24, loc='upper left')

    # Tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = plot_aircraft_cd0()
    # Save to graphics folder in project root
    graphics_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'graphics')
    os.makedirs(graphics_dir, exist_ok=True)
    save_path = os.path.join(graphics_dir, 'drag_coefficient_aircraft.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')
    plt.show()
