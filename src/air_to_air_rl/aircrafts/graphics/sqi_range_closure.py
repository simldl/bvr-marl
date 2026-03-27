"""
SQI vs Range & Closure Rate visualization.
Shows missile shot quality as a function of range and closure rate.
Uses calculation function from aircrafts/core/nez.py:sqi()
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def get_amraam_params():
    """Load AMRAAM parameters from codebase."""
    amraam_base_range_km = 150.0  # max_range_m: 150,000
    amraam_min_range_km = 1.5  # min_range_m: 1,500

    try:
        import inspect

        from air_to_air_rl.missiles.fox3.amraam import AIM120_AMRAAM

        amraam_source = inspect.getsource(AIM120_AMRAAM.__init__)
        if "150_000" in amraam_source:
            amraam_base_range_km = 150.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")
        elif "40_000" in amraam_source:
            amraam_base_range_km = 40.0
            print(f"[OK] Loaded AMRAAM from source: {amraam_base_range_km} km base range")
    except Exception as e:
        print(f"Using hardcoded AMRAAM values: {e}")

    return amraam_base_range_km, amraam_min_range_km


def compute_sqi(distance_normalized, closure_rate_norm, aspect_cos, rho_ratio=1.0):
    """
    Compute SQI using exact logistic function from aircrafts/core/nez.py

    Args:
        distance_normalized: Range factor (0=max range, 1=min range)
        closure_rate_norm: Closure rate / 400 m/s (clipped to [-1, 1])
        aspect_cos: Cosine of relative heading angle
        rho_ratio: Altitude density ratio (default 1.0 for sea level)

    Returns:
        SQI value [0, 1] where 0=poor and 1=excellent probability of kill
    """
    # SQI logistic model parameters from aircrafts/core/nez.py:sqi()
    a0 = -1.4  # baseline threshold
    a_d = 3.0  # distance/range factor (phi_d)
    a_Vc = 1.2  # closure rate factor (normalized by 400 m/s)
    a_th = 0.8  # aspect angle factor (cosine of relative heading)
    a_rho = 0.25  # altitude/density factor

    x = (
        a0
        + a_d * distance_normalized
        + a_Vc * closure_rate_norm
        + a_th * aspect_cos
        + a_rho * (rho_ratio - 1)
    )
    return 1.0 / (1.0 + np.exp(-x))


def create_sqi_range_closure_plot():
    """Create SQI vs Range & Closure Rate plot."""
    amraam_base_range_km, _ = get_amraam_params()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set font to match AIAA LaTeX template (Times/serif)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    range_km = np.linspace(0, amraam_base_range_km, 100)
    closure_rates_mps = [
        -400,
        -200,
        0,
        100,
        200,
        400,
        600,
    ]  # meters per second (negative = diverging)

    for closure_rate in closure_rates_mps:
        # Normalize closure rate
        closure_rate_norm = np.clip(closure_rate / 400, -1, 1)

        # Normalize distance (distance normalized decreases with range)
        distance_normalized = 1.0 - np.clip(range_km / amraam_base_range_km, 0, 1)

        sqi_values = []
        for d_norm in distance_normalized:
            sqi = compute_sqi(d_norm, closure_rate_norm, 0.0)  # neutral aspect
            sqi_values.append(sqi)

        if closure_rate == 0:
            label = "Stationary"
        elif closure_rate < 0:
            label = f"Diverging: {abs(closure_rate)} m/s"
        else:
            label = f"Closure: {closure_rate} m/s"
        ax.plot(
            range_km, sqi_values, linewidth=2.5, label=label, marker="o", markersize=4, markevery=10
        )

    ax.axhline(
        0.55, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Shot Threshold (0.55)"
    )
    ax.fill_between(range_km, 0.55, 1.0, alpha=0.1, color="green", label="Good Shot Zone")
    ax.fill_between(range_km, 0, 0.55, alpha=0.1, color="red", label="Poor Shot Zone")

    ax.set_xlabel("Slant Range (km)", fontsize=28)
    ax.set_ylabel("SQI (Shot Quality Index) [0-1]", fontsize=28)
    ax.set_xlim(0, amraam_base_range_km)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=28, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(axis="both", which="major", labelsize=24)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_sqi_range_closure_plot()

    # Save to graphics folder
    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)

    save_path = os.path.join(graphics_dir, "sqi_range_closure.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")

    plt.show()
