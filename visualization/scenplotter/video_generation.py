import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from scenplotter.scenario_plotter import (
    ScenarioPlotter, PlotConfig, StatusMessage, TopLeftMessage,
    Airplane, PolyLine, Missile as DrawableMissile, Arc, RadarCone, ScaleBar
)

from simulator.utils.map_limits import MapLimits
from reinforcement_learning.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from aircrafts.aircraft import Aircraft
from missiles.missile import Missile

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
colors = {
    "red_outline": (1, 0, 0, 1),
    "red_fill": (1, 0.4, 0.4, 1),
    "blue_outline": (0, 0, 1, 1),
    "blue_fill": (0.4, 0.4, 1, 1),
}


# ---------------------------------------------------------------------------
# Plotter‑Initialisierung
# ---------------------------------------------------------------------------

def initialize_plotter(map_limits: MapLimits | None = None):
    """Erzeugt einen ScenarioPlotter mit dunklem Hintergrund."""
    if map_limits is None:
        # Standard: 300 km × 300 km Quadrat um (0/0)
        map_limits = MapLimits(left_lon=-1.8, bottom_lat=-1.8, right_lon=1.8, top_lat=1.8)

    plot_config = PlotConfig()
    plot_config.units_scale = 20.0
    plot_config.background_color = "#808080"
    plot_config.borders_color = "#FFFFFF"

    return ScenarioPlotter(map_extents=map_limits, dpi=200, config=plot_config), map_limits


# ---------------------------------------------------------------------------
# Hilfsfunktion für Statusmeldungen
# ---------------------------------------------------------------------------

def update_status_message_queue(simulator, new_messages, message_queue, retention_time: int = 10):
    """Behält Meldungen für *retention_time* Sekunden und gibt String zurück."""
    current_time = simulator.utc_time
    message_queue[:] = [
        (msg, t) for msg, t in message_queue if (current_time - t).total_seconds() < retention_time
    ]
    for msg in new_messages:
        message_queue.append((msg, current_time))
    return "\n".join(msg for msg, _ in message_queue)


# ---------------------------------------------------------------------------
# Drawable Helper Functions
# ---------------------------------------------------------------------------

def plot_aircraft(unit: Aircraft, side: str, trace_record: dict, show_trace: bool = True, action_state: dict = None):
    """Returns drawable objects for an aircraft (including trace)."""
    altitude = unit.position.alt

    # Build enhanced info text with tactical information
    info_lines = []

    # Add agent name if available (format: A0, A1, B0, B1)
    if action_state and 'agent_name' in action_state:
        info_lines.append(action_state['agent_name'])

    info_lines.extend([
        f"Alt: {altitude:0.0f} m",
        f"Speed: {unit.speed/343:.2f} Mach"
    ])

    # Add target visibility info (sensor_tracks contains detected targets)
    # Only count enemy aircraft (different group)
    visible_count = 0
    if hasattr(unit, "sensor") and unit.sensor and hasattr(unit.sensor, "sensor_tracks"):
        for track_data in unit.sensor.sensor_tracks:
            # track_data format: (track_id, state, covariance, target_object, ...)
            if len(track_data) > 3 and track_data[3] is not None:
                target = track_data[3]
                # Only count if target is from different group (enemy)
                if hasattr(target, 'group') and target.group != unit.group:
                    visible_count += 1
    info_lines.append(f"Visible: {visible_count}")

    # Add missile count
    if hasattr(unit, "weapon_system"):
        missiles_remaining = len([m for m in unit.weapon_system.missiles if m.status == "loaded"])
        info_lines.append(f"MSL: {missiles_remaining}")

    # Add commanded actions if available
    if action_state:
        ps_cmd = action_state.get('ps_cmd_filtered', 0.0)
        n_cmd = action_state.get('n_cmd_filtered', 1.0)
        phi_cmd = action_state.get('phi_cmd_filtered', 0.0)
        info_lines.append(f"Ps: {ps_cmd:.1f} m/s")
        info_lines.append(f"n: {n_cmd:.2f}")
        info_lines.append(f"φ: {phi_cmd:.1f}°")

    info_text = "\n".join(info_lines)

    drawables = [
        Airplane(
            lat=unit.position.lat,
            lon=unit.position.lon,
            yaw_deg=unit.yaw_deg,
            edge_color=colors[f"{side}_outline"],
            fill_color=colors[f"{side}_fill"],
            info_text=info_text,
            zorder=10,
        )
    ]

    if show_trace and unit.id in trace_record:
        trace = [(pos.lat, pos.lon) for _, pos, _, _ in trace_record[unit.id]]
        drawables.append(
            PolyLine(
                points=trace,
                line_width=1,
                dash=(2, 2),
                edge_color=colors[f"{side}_outline"],
                zorder=1,
            )
        )
    return drawables


def plot_missile(unit: Missile, side: str, trace_record: dict):
    """Drawable objects for a missile (including trace & target marking)."""
    drawables = []
    missile_color = (0.85, 0.75, 0.1, 1)
    target_line_color = (0.5, 0.6, 0.1, 0.5)

    if unit.id in trace_record:
        trace = [(pos.lat, pos.lon) for _, pos, _, _ in trace_record[unit.id]]
        drawables.append(
            PolyLine(
                points=trace,
                line_width=1.5,
                dash=(2, 2),
                edge_color=missile_color,
                zorder=4,
            )
        )

    drawables.append(
        DrawableMissile(
            lat=unit.position.lat,
            lon=unit.position.lon,
            yaw_deg=unit.yaw_deg,
            edge_color=colors[f"{side}_outline"],
            fill_color=colors[f"{side}_fill"],
            info_text=None,
            zorder=9,
        )
    )

    if getattr(unit, "target", None) is not None:
        target_pos = unit.target.position
        drawables.extend(
            [
                Arc(
                    center_lat=target_pos.lat,
                    center_lon=target_pos.lon,
                    radius=0.2,
                    angle1=0,
                    angle2=360,
                    edge_color=missile_color,
                    fill_color=missile_color,
                    line_width=1,
                    zorder=5,
                ),
                PolyLine(
                    points=[
                        (unit.position.lat, unit.position.lon),
                        (target_pos.lat, target_pos.lon),
                    ],
                    line_width=1.5,
                    dash=None,
                    edge_color=target_line_color,
                    zorder=5,
                ),
            ]
        )
    return drawables


def plot_radar_cone(unit: Aircraft, side: str):
    """Display radar cone based on the new radar class."""
    radar = getattr(unit, "radar", None)
    if radar is None:
        return []

    delta = radar.h_fov_deg / 2  # horizontaler Öffnungswinkel / 2
    radius = radar.max_range_m / 1000.0  # Meter → Kilometer

    angle1 = (unit.yaw_deg - delta + 360) % 360
    angle2 = (unit.yaw_deg + delta) % 360

    return [
        RadarCone(
            center_lat=unit.position.lat,
            center_lon=unit.position.lon,
            radius=radius,
            angle1=angle1,
            angle2=angle2,
            edge_color=colors[f"{side}_outline"],
            fill_color=(
                colors[f"{side}_fill"][0],
                colors[f"{side}_fill"][1],
                colors[f"{side}_fill"][2],
                0.3,
            ),
            line_width=2,
            zorder=10,
        )
    ]


# ---------------------------------------------------------------------------
# Static Snapshot
# ---------------------------------------------------------------------------

def plot_simulation(simulator, output_file: Path | str, map_limits: MapLimits):
    """Save current simulation state as PNG."""
    plotter = ScenarioPlotter(map_extents=map_limits, config=PlotConfig())
    drawables = []

    for unit in simulator.active_units.values():
        side = "blue" if unit.group == "agent" else "red"
        if isinstance(unit, Aircraft):
            drawables.extend(plot_aircraft(unit, side, simulator.trace_record_units))
        elif isinstance(unit, Missile):
            drawables.extend(plot_missile(unit, side, simulator.trace_record_units))

    # Scale bar
    margin = 20  # px
    scale_length = 100  # km
    scale_bar = ScaleBar(
        length_km=scale_length,
        x=plotter.img_width - margin - (plotter._get_image_distance(scale_length) / 2),
        y=margin,
        line_width=2,
        edge_color=(1, 1, 1, 1),
        zorder=50,
    )
    drawables.append(scale_bar)

    plotter.to_png(output_file, drawables)
    print(f"Plot saved to {output_file}")


# ---------------------------------------------------------------------------
# Live Simulation / Video Output
# ---------------------------------------------------------------------------

def live_simulation(
    trained_model,
    env: BVRMultiAgentEnv,
    frames: int = 200,
    interval: int = 100,
    save_video: bool = False,
    video_output_file: Path = Path("output/optics_test_run.mp4"),
):
    """Runs a live simulation and displays/saves a video."""
    observations, _ = env.reset()

    plotter, _ = initialize_plotter(env.map_limits)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")
    img = ax.imshow([[0]], interpolation="none")

    previous_aircraft = {}
    status_message_queue = []
    action_states = {}  # Store action states per unit
    cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}  # Track cumulative rewards
    current_step_rewards = {}  # Track rewards from current step

    # ------------------------------------------------------------
    def update_frame(frame_index):
        nonlocal observations, previous_aircraft, action_states, cumulative_rewards, current_step_rewards

        # Check if episode is done
        if not env.agents or len(env.agents) == 0:
            # Episode ended, reset the environment
            observations, _ = env.reset()
            previous_aircraft.clear()
            action_states.clear()
            cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}
            current_step_rewards.clear()

        # Aktionen berechnen
        actions = {}
        for agent_id in env.agents:
            if agent_id in observations:
                actions[agent_id] = trained_model.compute_single_action(observations[agent_id], agent_id)

        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        # Update reward tracking
        current_step_rewards = rewards.copy()
        for agent_id, reward in rewards.items():
            if agent_id != "__all__":  # Skip the special "__all__" key
                cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + reward

        # Extract action states from environment (if available)
        if hasattr(env, 'action_processor') and hasattr(env.action_processor, 'state'):
            action_states = env.action_processor.state.copy()

        # Check if episode just ended
        if terminateds.get("__all__", False) or truncateds.get("__all__", False):
            # Reset for next episode
            observations, _ = env.reset()
            previous_aircraft.clear()
            action_states.clear()
            cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}
            current_step_rewards.clear()

        # Drawables einsammeln
        drawables = []
        current_aircraft = {}

        for unit in env.simulator.active_units.values():
            side = "blue" if unit.group == "agent" else "red"
            if isinstance(unit, Aircraft):
                # Get action state for this unit if available
                unit_action_state = action_states.get(unit.id, None)
                drawables.extend(plot_aircraft(unit, side, env.simulator.trace_record_units, action_state=unit_action_state))
                drawables.extend(plot_radar_cone(unit, side))
                current_aircraft[unit.id] = True
            elif isinstance(unit, Missile):
                drawables.extend(plot_missile(unit, side, env.simulator.trace_record_units))

        # Messages for destroyed aircraft
        new_events = []
        destroyed = set(previous_aircraft.keys()) - set(current_aircraft.keys())
        for did in destroyed:
            new_events.append(f"Aircraft {did} destroyed.")
        status_str = update_status_message_queue(env.simulator, new_events, status_message_queue)
        if status_str:
            drawables.append(StatusMessage(status_str, zorder=100))

        previous_aircraft = current_aircraft

        # Duration + Sim status + Rewards
        status_lines = [f"Duration: {env.simulator.elapsed_time_s:.1f} s"]

        # Calculate team rewards
        agent_team_reward = sum(cumulative_rewards.get(aid, 0.0) for aid in env.agent_ids if aid in cumulative_rewards)
        opponent_team_reward = sum(cumulative_rewards.get(oid, 0.0) for oid in env.opponent_ids if oid in cumulative_rewards)
        agent_step_reward = sum(current_step_rewards.get(aid, 0.0) for aid in env.agent_ids if aid in current_step_rewards)
        opponent_step_reward = sum(current_step_rewards.get(oid, 0.0) for oid in env.opponent_ids if oid in current_step_rewards)

        status_lines.append(f"Blue Team: {agent_team_reward:+.1f} (step: {agent_step_reward:+.1f})")
        status_lines.append(f"Red Team: {opponent_team_reward:+.1f} (step: {opponent_step_reward:+.1f})")

        drawables.append(TopLeftMessage("\n".join(status_lines), zorder=100))
        if env.simulator.status_text is not None:
            drawables.append(StatusMessage(env.simulator.status_text, zorder=100))

        # Scale bar
        margin = 20
        scale_length = 100
        scale_bar = ScaleBar(
            length_km=scale_length,
            x=plotter.img_width - margin - (plotter._get_image_distance(scale_length) / 2),
            y=margin,
            line_width=2,
            edge_color=(1, 1, 1, 1),
            zorder=50,
        )
        drawables.append(scale_bar)

        # Rendering via temp‑PNG
        temp_file = Path("temp_frame.png")
        plotter.to_png(str(temp_file), drawables)
        img.set_data(plt.imread(str(temp_file)))
        return (img,)

    # ------------------------------------------------------------
    anim = animation.FuncAnimation(fig, update_frame, frames=frames, interval=interval, blit=True)

    if save_video:
        video_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Try multiple writer options in order of preference
        writers_to_try = [
            ('ffmpeg', lambda fps: animation.FFMpegWriter(fps=fps)),
            ('pillow', lambda fps: animation.PillowWriter(fps=fps)),
        ]

        fps = max(1, 1000 // interval)
        saved = False

        for writer_name, writer_factory in writers_to_try:
            try:
                if animation.writers.is_available(writer_name):
                    writer = writer_factory(fps)
                    # Change extension based on writer
                    if writer_name == 'pillow':
                        video_output_file = video_output_file.with_suffix('.gif')
                    anim.save(str(video_output_file), writer=writer)
                    print(f"✓ Video saved to {video_output_file} using {writer_name}")
                    saved = True
                    break
            except Exception as e:
                print(f"✗ Failed to save with {writer_name}: {e}")
                continue

        if not saved:
            print(f"\n{'='*60}")
            print("ERROR: Could not save video - no suitable writer found!")
            print("Available options to fix this:")
            print("1. Install FFmpeg: conda install -c conda-forge ffmpeg")
            print("2. Or install Pillow: pip install pillow")
            print("3. Or set save_video: false in viz_config.yaml")
            print(f"{'='*60}\n")
            # Show the animation anyway
            plt.show()
    else:
        plt.show()
