"""
Architecture diagram showing component relationships and data flow in the air-to-air RL simulation.
Visualizes how Physics, Radar, Missiles, Aircraft, and RL Environment work together.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def create_architecture_diagram() -> plt.Figure:
    """
    Create a comprehensive architecture diagram showing component interactions.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7, 9.6, "Simulation Architecture", fontsize=18, fontweight="bold", ha="center")

    colors = {
        "physics": "#E8F4F8",
        "radar": "#FFE8D6",
        "missiles": "#E8F8E8",
        "aircraft": "#F8E8F4",
        "environment": "#F8F8E8",
        "simulator": "#F0F0F0",
    }

    boxes = {
        "physics": {"x": 0.3, "y": 7.8, "w": 3.5, "h": 1.6},
        "radar": {"x": 10.2, "y": 7.8, "w": 3.5, "h": 1.6},
        "simulator": {"x": 5, "y": 6.5, "w": 4, "h": 1.2},
        "missiles": {"x": 0.3, "y": 4.8, "w": 3.5, "h": 2.0},
        "aircraft": {"x": 10.2, "y": 4.8, "w": 3.5, "h": 2.0},
        "environment": {"x": 3.5, "y": 2.2, "w": 7, "h": 2.0},
    }

    sim_box = FancyBboxPatch(
        (boxes["simulator"]["x"], boxes["simulator"]["y"]),
        boxes["simulator"]["w"],
        boxes["simulator"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="black",
        facecolor=colors["simulator"],
        linewidth=2.5,
    )
    ax.add_patch(sim_box)
    ax.text(7, 7.25, "Simulator", fontsize=11, fontweight="bold", ha="center")
    ax.text(7, 6.85, "Unit management & Event system", fontsize=8, ha="center", style="italic")

    physics_box = FancyBboxPatch(
        (boxes["physics"]["x"], boxes["physics"]["y"]),
        boxes["physics"]["w"],
        boxes["physics"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="#0066CC",
        facecolor=colors["physics"],
        linewidth=2.5,
    )
    ax.add_patch(physics_box)
    ax.text(
        2.05, 9.2, "Physics Layer", fontsize=11, fontweight="bold", ha="center", color="#0066CC"
    )
    ax.text(2.05, 8.85, "Aircraft & Missile Physics", fontsize=8, ha="center", style="italic")
    ax.text(2.05, 8.5, "• Aerodynamics (CD, CL, CA)", fontsize=7.5, ha="center")
    ax.text(2.05, 8.15, "• Turn rates & load factors", fontsize=7.5, ha="center")

    radar_box = FancyBboxPatch(
        (boxes["radar"]["x"], boxes["radar"]["y"]),
        boxes["radar"]["w"],
        boxes["radar"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="#FF6600",
        facecolor=colors["radar"],
        linewidth=2.5,
    )
    ax.add_patch(radar_box)
    ax.text(
        11.95, 9.2, "Radar System", fontsize=11, fontweight="bold", ha="center", color="#FF6600"
    )
    ax.text(11.95, 8.85, "Detection & Tracking", fontsize=8, ha="center", style="italic")
    ax.text(11.95, 8.5, "• RCS calculation & Doppler", fontsize=7.5, ha="center")
    ax.text(11.95, 8.15, "• Lock management", fontsize=7.5, ha="center")

    missile_box = FancyBboxPatch(
        (boxes["missiles"]["x"], boxes["missiles"]["y"]),
        boxes["missiles"]["w"],
        boxes["missiles"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="#00AA00",
        facecolor=colors["missiles"],
        linewidth=2.5,
    )
    ax.add_patch(missile_box)
    ax.text(
        2.05, 6.5, "Missile System", fontsize=11, fontweight="bold", ha="center", color="#00AA00"
    )
    ax.text(2.05, 6.15, "Guidance & Flight Management", fontsize=8, ha="center", style="italic")
    ax.text(2.05, 5.8, "• Flight phases (boost/middle/terminal)", fontsize=7.5, ha="center")
    ax.text(
        2.05,
        5.45,
        "• Selection and calculation of guidance for navigation",
        fontsize=7.5,
        ha="center",
    )

    aircraft_box = FancyBboxPatch(
        (boxes["aircraft"]["x"], boxes["aircraft"]["y"]),
        boxes["aircraft"]["w"],
        boxes["aircraft"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="#6600CC",
        facecolor=colors["aircraft"],
        linewidth=2.5,
    )
    ax.add_patch(aircraft_box)
    ax.text(
        11.95, 6.5, "Aircraft System", fontsize=11, fontweight="bold", ha="center", color="#6600CC"
    )
    ax.text(11.95, 6.15, "Flight Control & Sensors", fontsize=8, ha="center", style="italic")
    ax.text(11.95, 5.8, "• Weapon systems & Countermeasures", fontsize=7.5, ha="center")
    ax.text(11.95, 5.45, "• Control surfaces (pitch/roll/yaw)", fontsize=7.5, ha="center")

    env_box = FancyBboxPatch(
        (boxes["environment"]["x"], boxes["environment"]["y"]),
        boxes["environment"]["w"],
        boxes["environment"]["h"],
        boxstyle="round,pad=0.1",
        edgecolor="#CCAA00",
        facecolor=colors["environment"],
        linewidth=2.5,
    )
    ax.add_patch(env_box)
    ax.text(7, 3.95, "RL Environment", fontsize=11, fontweight="bold", ha="center", color="#CCAA00")
    ax.text(7, 3.6, "Agent Training & Learning", fontsize=8, ha="center", style="italic")
    ax.text(7, 3.25, "• Observation & Action Spaces", fontsize=7.5, ha="center")
    ax.text(7, 2.9, "• Training & Feedback", fontsize=7.5, ha="center")

    def get_box_edge(box_name, direction):
        """Get edge point of box in direction: 'top', 'bottom', 'left', 'right'"""
        b = boxes[box_name]
        cx = b["x"] + b["w"] / 2  # center x
        cy = b["y"] + b["h"] / 2  # center y

        if direction == "top":
            return (cx, b["y"] + b["h"])
        elif direction == "bottom":
            return (cx, b["y"])
        elif direction == "left":
            return (b["x"], cy)
        elif direction == "right":
            return (b["x"] + b["w"], cy)
        elif direction == "top_right":
            return (b["x"] + b["w"], b["y"] + b["h"])
        elif direction == "top_left":
            return (b["x"], b["y"] + b["h"])
        elif direction == "bottom_right":
            return (b["x"] + b["w"], b["y"])
        elif direction == "bottom_left":
            return (b["x"], b["y"])

    ax.add_patch(
        FancyArrowPatch(
            get_box_edge("physics", "right"),
            get_box_edge("simulator", "top"),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=1.5,
            color="grey",
            alpha=0.6,
            connectionstyle="arc3,rad=-0.3",
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            get_box_edge("radar", "left"),
            get_box_edge("simulator", "top"),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=1.5,
            color="grey",
            alpha=0.6,
            connectionstyle="arc3,rad=0.3",
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            get_box_edge("missiles", "right"),
            get_box_edge("simulator", "bottom"),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=1.5,
            color="grey",
            alpha=0.6,
            connectionstyle="arc3,rad=0.3",
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            get_box_edge("aircraft", "left"),
            get_box_edge("simulator", "bottom"),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=1.5,
            color="grey",
            alpha=0.6,
            connectionstyle="arc3,rad=-0.3",
        )
    )

    ax.add_patch(
        FancyArrowPatch(
            get_box_edge("simulator", "bottom"),
            get_box_edge("environment", "top"),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=2,
            color="black",
            alpha=0.8,
            connectionstyle="arc3,rad=0",
        )
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = create_architecture_diagram()

    graphics_dir = os.path.join(os.path.dirname(__file__), "..", "..", "graphics")
    os.makedirs(graphics_dir, exist_ok=True)

    save_path = os.path.join(graphics_dir, "architecture_diagram.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")

    plt.show()
