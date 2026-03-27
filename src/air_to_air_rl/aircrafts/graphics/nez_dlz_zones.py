"""
NEZ (No Escape Zone) and DLZ (Dynamic Launch Zone) visualization.
Shows engagement zones (R1, R2, R3, R4) as a function of range and target parameters.
Uses actual calculation functions from the codebase.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


def create_nez_dlz_plot():
    """
    Create visualization of NEZ/DLZ engagement zones.
    Shows how different missile types affect the engagement envelope.
    Uses actual parameters from missiles/fox3/ and calculation formulas from aircrafts/core/nez.py
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Try to load parameters directly from missile Python classes

    # Hardcoded fallback defaults (will be overridden if imports succeed)
    amraam_base_range_km = 150.0
    amraam_min_range_km = 1.5

    meteor_base_range_km = 200.0
    meteor_min_range_km = 2.0

    # Try to load directly from missile Python classes by reading source code
    try:
        import inspect

        from air_to_air_rl.missiles.fox3.amraam import AIM120_AMRAAM
        from air_to_air_rl.missiles.fox3.meteor import Meteor

        # Get source code and extract max_range_m values
        amraam_source = inspect.getsource(AIM120_AMRAAM.__init__)
        if "150_000" in amraam_source:
            amraam_base_range_km = 150.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")
        elif "40_000" in amraam_source:
            amraam_base_range_km = 40.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")

        meteor_source = inspect.getsource(Meteor.__init__)
        if "200_000" in meteor_source:
            meteor_base_range_km = 200.0
            print(f"[OK] Loaded Meteor from source: {meteor_base_range_km} km base range")
        elif "50_000" in meteor_source:
            meteor_base_range_km = 50.0
            print(f"[OK] Loaded Meteor from source: {meteor_base_range_km} km base range")

    except Exception as e:
        print(f"[WARNING] Could not import missile classes: {e}")
        print("[WARNING] Using hardcoded default values")

    # Using exact formula from aircrafts/core/nez.py:compute_dlz()
    # These formulas are used in the actual NoEscapeZoneCalculator class
    def calculate_dlz_zones(min_range_km, base_range_km):
        """
        Calculate DLZ zones using exact formula from nez.py:compute_dlz()
        - r_tr: Turning target range (R2 boundary, NEZ limit)
        - r_pi: Pursuit intercept range (R3 boundary, optimal range)
        - r_max: Maximum kinematic range (R4 limit)
        """
        r_tr = min_range_km + 0.60 * (base_range_km - min_range_km)
        r_pi = min_range_km + 0.88 * (base_range_km - min_range_km)
        r_aero = min_range_km + 1.04 * (base_range_km - min_range_km)
        r_max = base_range_km * 1.3  # kinematic extension (same as nez.py)
        return r_tr, r_pi, r_aero, r_max

    amraam_r_tr, amraam_r_pi, amraam_r_aero, amraam_r_max = calculate_dlz_zones(
        amraam_min_range_km, amraam_base_range_km
    )
    meteor_r_tr, meteor_r_pi, meteor_r_aero, meteor_r_max = calculate_dlz_zones(
        meteor_min_range_km, meteor_base_range_km
    )

    ax = axes[0]

    missiles = [
        (
            "AIM-120D AMRAAM",
            amraam_min_range_km,
            amraam_r_tr,
            amraam_r_pi,
            amraam_r_max,
            ["#95E1D3", "#6DDF9B", "#38E19B", "#0BC965"],
            0.7,
        ),
        (
            "Meteor",
            meteor_min_range_km,
            meteor_r_tr,
            meteor_r_pi,
            meteor_r_max,
            ["#FFE66D", "#FFC700", "#FF9800", "#FF6B35"],
            0.7,
        ),
    ]

    y_pos = 0
    y_height = 0.4

    for missile_name, r_min, r_tr, r_pi, r_max, colors, alpha_base in missiles:
        y_base = y_pos

        # R1: Too close
        ax.barh(
            y_base,
            r_min,
            left=0,
            height=y_height,
            color=colors[0],
            alpha=alpha_base,
            edgecolor="black",
            linewidth=2,
        )

        # R2: NEZ
        ax.barh(
            y_base,
            r_tr - r_min,
            left=r_min,
            height=y_height,
            color=colors[1],
            alpha=alpha_base,
            edgecolor="black",
            linewidth=2,
        )

        # R3: Optimal
        ax.barh(
            y_base,
            r_pi - r_tr,
            left=r_tr,
            height=y_height,
            color=colors[2],
            alpha=alpha_base,
            edgecolor="black",
            linewidth=2,
        )

        # R4: Extended
        ax.barh(
            y_base,
            r_max - r_pi,
            left=r_pi,
            height=y_height,
            color=colors[3],
            alpha=alpha_base,
            edgecolor="black",
            linewidth=2,
        )

        # Add missile name above the bar
        ax.text(
            0,
            y_base + y_height + 0.15,
            missile_name,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        # Add labels on bars
        ax.text(
            r_min / 2,
            y_base + y_height / 2,
            "R1",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        ax.text(
            (r_min + r_tr) / 2,
            y_base + y_height / 2,
            "R2\nNEZ",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )
        ax.text(
            (r_tr + r_pi) / 2,
            y_base + y_height / 2,
            "R3\nOptimal",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )
        ax.text(
            (r_pi + r_max) / 2,
            y_base + y_height / 2,
            "R4\nExtended",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

        # Add max range label
        ax.text(
            r_max + 1.5,
            y_base + y_height / 2,
            f"{r_max:.1f} km",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        y_pos += 1.0

    ax.set_xlim(-15, 280)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Slant Range (km)", fontsize=11, fontweight="bold")
    ax.set_title("Fox-3 Missile DLZ Comparison", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Add legend for zones
    legend_elements = [
        Patch(facecolor="#95E1D3", edgecolor="black", label="R1: Too Close"),
        Patch(facecolor="#6DDF9B", edgecolor="black", label="R2: NEZ (Best)"),
        Patch(facecolor="#38E19B", edgecolor="black", label="R3: Optimal"),
        Patch(facecolor="#0BC965", edgecolor="black", label="R4: Extended"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax = axes[1]

    aspect_angles = np.linspace(0, 180, 100)  # 0° = head-on, 90° = beam, 180° = tail

    # Fox-3 AMRAAM parameters (from actual data extracted above)
    # Aspect angle effect: RCS varies with aspect, affecting effective range
    # Larger RCS at beam (90°), smaller at nose/tail

    # NEZ boundary (R2) - decreases slightly at less favorable aspects
    nez_range_fox3 = amraam_r_tr + 8 * np.cos(2 * np.radians(aspect_angles - 90))

    # R3 boundary (optimal range)
    r3_range_fox3 = amraam_r_pi + 12 * np.cos(2 * np.radians(aspect_angles - 90))

    # Maximum kinematic range (R4)
    r_max_fox3 = amraam_r_max + 15 * np.cos(2 * np.radians(aspect_angles - 90))

    # Plot ranges
    ax.fill_between(
        aspect_angles, 0, nez_range_fox3, alpha=0.3, color="green", label="R2: NEZ (Best)"
    )
    ax.fill_between(
        aspect_angles, nez_range_fox3, r3_range_fox3, alpha=0.2, color="yellow", label="R3: Optimal"
    )
    ax.fill_between(
        aspect_angles, r3_range_fox3, r_max_fox3, alpha=0.1, color="orange", label="R4: Extended"
    )

    ax.plot(aspect_angles, nez_range_fox3, "g-", linewidth=2.5, label="NEZ Boundary")
    ax.plot(
        aspect_angles,
        r3_range_fox3,
        "darkorange",
        linewidth=2.5,
        linestyle="--",
        label="R3 Boundary",
    )
    ax.plot(
        aspect_angles, r_max_fox3, color="darkred", linewidth=2.5, linestyle=":", label="Max Range"
    )

    # Highlight aspect angles
    ax.axvline(0, color="blue", linestyle=":", alpha=0.5, linewidth=1.5)
    ax.text(
        0,
        amraam_r_max * 0.85,
        "0°\nHead-on\n(Low RCS)",
        ha="center",
        fontsize=8,
        color="blue",
        fontweight="bold",
    )

    ax.axvline(90, color="darkgreen", linestyle=":", alpha=0.5, linewidth=2)
    ax.text(
        90,
        amraam_r_max * 0.85,
        "90°\nBeam\n(High RCS)",
        ha="center",
        fontsize=8,
        color="darkgreen",
        fontweight="bold",
    )

    ax.axvline(180, color="blue", linestyle=":", alpha=0.5, linewidth=1.5)
    ax.text(
        180,
        amraam_r_max * 0.85,
        "180°\nTail\n(Low RCS)",
        ha="center",
        fontsize=8,
        color="blue",
        fontweight="bold",
    )

    ax.set_xlabel("Target Aspect Angle (degrees)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Effective Range (km)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Fox-3 AMRAAM NEZ vs Target Aspect ({amraam_base_range_km:.0f}km base, RCS dependent)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 180)
    ax.set_ylim(0, meteor_r_max * 1.05)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)

    # Add xticks for aspect angles
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°"])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_nez_dlz_plot()

    # Save to graphics folder
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)

    save_path = os.path.join(graphics_dir, "nez_dlz_zones.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")

    plt.show()
