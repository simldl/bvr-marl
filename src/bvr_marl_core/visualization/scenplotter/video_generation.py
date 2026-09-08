from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bvr_marl_core.simulator import MapLimits
from bvr_marl_core.visualization import can_show_matplotlib_window
from bvr_marl_core.visualization.scenario_overlays import (
    build_scenario_overlay_drawables,
    build_scenario_status_lines,
    select_display_limits,
)
from bvr_marl_core.visualization.scenplotter import (
    AWACS,
    Airplane,
    Arc,
    MapLabel,
    PlotConfig,
    PolyLine,
    RadarCone,
    ScaleBar,
    ScenarioPlotter,
    StatusMessage,
    TopLeftMessage,
)
from bvr_marl_core.visualization.scenplotter import (
    Missile as DrawableMissile,
)
from bvr_marl_core.visualization.scenplotter.label_manager import FighterLabelManager


@runtime_checkable
class SimulationEnv(Protocol):
    """Protocol describing the subset of a gym environment used by live_simulation.

    Any object that implements this interface can be passed to live_simulation —
    including BVRMultiAgentEnv but also custom wrappers or test doubles — without
    requiring a direct import of BVRMultiAgentEnv here.
    """

    agents: list
    agent_ids: list
    opponent_ids: list
    map_limits: MapLimits
    agent_to_unit_id: dict
    simulator: Any  # bvr_marl_core.simulator.Simulator

    def reset(self) -> tuple: ...
    def step(self, actions: dict) -> tuple: ...


AFFILIATION_COLORS = {
    "friendly": {"outline": (0.0, 0.4, 1.0, 1.0), "fill": (0.4, 0.6, 1.0, 1.0)},  # Blue
    "hostile": {"outline": (1.0, 0.0, 0.0, 1.0), "fill": (1.0, 0.4, 0.4, 1.0)},  # Red
    "neutral": {"outline": (0.0, 0.7, 0.0, 1.0), "fill": (0.4, 0.8, 0.4, 1.0)},  # Green
    "unknown": {"outline": (0.9, 0.8, 0.0, 1.0), "fill": (1.0, 0.9, 0.4, 1.0)},  # Yellow
}

# Distinct colors for support-asset (AWACS) radar cones — cyan/orange so they
# don't visually clash with fighter radar cones.
SUPPORT_RADAR_COLORS = {
    "friendly": {"outline": (0.0, 0.85, 0.85, 1.0), "fill": (0.3, 0.9, 0.9, 1.0)},  # Cyan
    "hostile": {"outline": (1.0, 0.50, 0.0, 1.0), "fill": (1.0, 0.7, 0.3, 1.0)},  # Orange
    "neutral": {"outline": (0.6, 0.9, 0.0, 1.0), "fill": (0.7, 1.0, 0.3, 1.0)},  # Lime
    "unknown": {"outline": (0.8, 0.6, 1.0, 1.0), "fill": (0.9, 0.7, 1.0, 1.0)},  # Lavender
}

# Side-to-affiliation mapping
SIDE_AFFILIATION = {
    "blue": "friendly",
    "red": "hostile",
}

# Aircraft-class-name to flying_objects symbol type mapping
_AIRCRAFT_TYPE_MAP = {
    "F22": "f22",
    "F35": "f35",
    "Eurofighter": "eurofighter",
    "Su57": "su57",
    "AWACS": "awacs",
    "F15EX": "f15ex",
    "DebugPlane": "f22",  # fallback to f22 silhouette
}

# Aircraft-class-name to military designator display names
_AIRCRAFT_DISPLAY_NAMES = {
    "F22": "F-22A",
    "F35": "F-35A",
    "Eurofighter": "EF-2000",
    "Su57": "Su-57",
    "AWACS": "E-7A",
    "F15EX": "F-15EX",
    "DebugPlane": "F-22A",
}

# Missile-class-name to display names
_MISSILE_DISPLAY_NAMES = {
    "AIM120_AMRAAM": "AMRAAM",
    "Meteor": "Meteor",
    "K77M": "K-77M",
    "R77_1": "R-77-1",
    "R37M": "R-37M",
    "LongRangeMissile": "BVR-MSL",
    "AIM7_Sparrow": "AIM-7",
    "AIM9_Sidewinder": "AIM-9",
    "Python_5": "Python-5",
    "Skyflash": "Skyflash",
}


def _get_aircraft_type(unit) -> str:
    """Return the flying_objects symbol key for an aircraft unit."""
    return _AIRCRAFT_TYPE_MAP.get(type(unit).__name__, "f22")


def initialize_plotter(
    map_limits: MapLimits | None = None,
    symbol_registry=None,
    symbol_mode: str = "nato",
    dpi: int = 200,
    symbol_scale: float = 2.0,
    show_text: bool = True,
):
    """Create a ScenarioPlotter with dark tactical background.

    symbol_scale controls how large aircraft/missile sprites are drawn relative
    to the base PNG size.  2.0 = double size (default).  Font size and
    label spacing are scaled proportionally.
    """
    if map_limits is None:
        map_limits = MapLimits(left_lon=-1.8, bottom_lat=-1.8, right_lon=1.8, top_lat=1.8)

    plot_config = PlotConfig()
    plot_config.units_scale = 20.0
    # Dark tactical palette — dark blue-grey background with muted geographic features
    plot_config.background_color = "#1a2535"
    plot_config.borders_color = "#8090a8"  # muted blue-grey — visible but not distracting
    plot_config.coastlines_color = "#5a6e82"  # slightly darker than borders
    plot_config.grid_color = "#25364a"  # very subtle — just enough to see the grid
    plot_config.symbol_mode = symbol_mode
    plot_config.show_text = bool(show_text)
    # Encode scale into the requested PNG size so Cairo renders 1:1 (no upscaling = sharp).
    plot_config.symbol_size = max(32, round(32 * symbol_scale))
    plot_config.missile_symbol_size = max(20, round(20 * symbol_scale * 0.5))
    plot_config.symbol_scale = 1.0  # PNG is already the right display size
    plot_config.missile_symbol_scale = 1.0
    # Scale font and label spacing proportionally with the symbol size
    plot_config.sprites_info_font_size = max(10, round(10 * symbol_scale))
    plot_config.sprites_info_spacing = max(26, round(26 * symbol_scale))

    return ScenarioPlotter(
        map_extents=map_limits,
        dpi=dpi,
        config=plot_config,
        symbol_registry=symbol_registry,
    ), map_limits


def update_status_message_queue(simulator, new_messages, message_queue, retention_time: int = 10):
    """Keep messages for *retention_time* seconds and return combined string."""
    current_time = simulator.utc_time
    message_queue[:] = [
        (msg, t) for msg, t in message_queue if (current_time - t).total_seconds() < retention_time
    ]
    for msg in new_messages:
        message_queue.append((msg, current_time))
    return "\n".join(msg for msg, _ in message_queue)


def plot_aircraft(
    unit,
    side: str,
    trace_record: dict,
    show_trace: bool = True,
    action_state: dict = None,
    label_priority: int = 0,
    label_detail: str = "compact",
):
    """Returns drawable objects for an aircraft (including trace)."""
    affiliation = SIDE_AFFILIATION.get(side, "unknown")
    outline = AFFILIATION_COLORS[affiliation]["outline"]
    fill = AFFILIATION_COLORS[affiliation]["fill"]

    info_text = _aircraft_label_text(unit, action_state, detail=label_detail)

    # Choose sprite based on aircraft type
    is_awacs = getattr(unit, "is_support_asset", False)
    aircraft_type = _get_aircraft_type(unit)
    if is_awacs:
        sprite = AWACS(
            lat=unit.position.lat,
            lon=unit.position.lon,
            yaw_deg=unit.yaw_deg,
            edge_color=outline,
            fill_color=fill,
            info_text=None,
            zorder=10,
            affiliation=affiliation,
            aircraft_type=aircraft_type,
        )
    else:
        sprite = Airplane(
            lat=unit.position.lat,
            lon=unit.position.lon,
            yaw_deg=unit.yaw_deg,
            edge_color=outline,
            fill_color=fill,
            info_text=None,
            zorder=10,
            affiliation=affiliation,
            aircraft_type=aircraft_type,
        )

    drawables = [
        sprite,
        MapLabel(
            lat=unit.position.lat,
            lon=unit.position.lon,
            text=info_text,
            unit_id=unit.id,
            priority=label_priority,
            affiliation=affiliation,
            cluster_name=_AIRCRAFT_DISPLAY_NAMES.get(type(unit).__name__, type(unit).__name__),
        ),
    ]

    if show_trace and unit.id in trace_record:
        trace = [(pos.lat, pos.lon) for _, pos, _, _ in trace_record[unit.id]]
        drawables.append(
            PolyLine(
                points=trace,
                line_width=1,
                dash=(2, 2),
                edge_color=outline,
                zorder=3,
            )
        )
    return drawables


def _aircraft_label_text(unit, action_state: dict | None, *, detail: str) -> str:
    """Build compact map text or the selected-unit detail label."""
    identity = (
        str(action_state["agent_name"])
        if action_state and action_state.get("agent_name")
        else str(getattr(unit, "id", type(unit).__name__))
    )
    aircraft = _AIRCRAFT_DISPLAY_NAMES.get(type(unit).__name__, type(unit).__name__)
    altitude = float(unit.position.alt)
    mach = float(unit.speed) / 343.0
    if detail == "short":
        return f"{identity} · {aircraft}"
    if detail != "full":
        return f"{identity} · {aircraft}\n{altitude / 1000:.1f} km · M{mach:.2f}"

    remaining = getattr(unit, "remaining_missiles", 0)
    if hasattr(unit, "weapons"):
        remaining = getattr(unit.weapons, "remaining_missiles", remaining)
    tracks = _visible_track_count(unit)
    fuel = getattr(unit, "fuel", None)
    fuel_text = f" · fuel {float(fuel):.0f}" if isinstance(fuel, (int, float)) else ""
    return (
        f"{identity} · {aircraft}\n"
        f"ALT {altitude / 1000:.1f} km · M{mach:.2f}{fuel_text}\n"
        f"MSL {remaining} · tracks {tracks}"
    )


def _visible_track_count(unit) -> int:
    sensor = getattr(unit, "sensor", None)
    tracks = getattr(sensor, "sensor_tracks", ()) if sensor is not None else ()
    return sum(
        1
        for track in tracks
        if getattr(track, "engageable", False) and not getattr(track, "suspect_deception", False)
    )


def _deterministic_focus(active_units, rule: str) -> object | None:
    """Resolve a non-interactive focus id for saved video rendering."""
    aircraft = sorted(
        (unit for unit in active_units if getattr(unit, "unit_kind", None) == "aircraft"),
        key=lambda unit: str(unit.id),
    )
    if not aircraft or rule == "none":
        return None
    if rule == "first":
        return aircraft[0].id
    if rule == "nearest_threat":
        candidates = []
        aircraft_ids = {unit.id for unit in aircraft}
        for missile in active_units:
            if getattr(missile, "unit_kind", None) != "missile":
                continue
            target = getattr(missile, "target", None)
            if target is None or getattr(target, "id", None) not in aircraft_ids:
                continue
            dx = float(missile.position.lon) - float(target.position.lon)
            dy = float(missile.position.lat) - float(target.position.lat)
            candidates.append((dx * dx + dy * dy, str(target.id), target.id))
        return min(candidates)[2] if candidates else aircraft[0].id
    raise ValueError("video_focus_rule must be one of: none, first, nearest_threat")


def plot_missile(unit, side: str, trace_record: dict, active_unit_ids: set | None = None):
    """Drawable objects for a missile (including trace & target marking)."""
    affiliation = SIDE_AFFILIATION.get(side, "unknown")
    outline = AFFILIATION_COLORS[affiliation]["outline"]
    fill = AFFILIATION_COLORS[affiliation]["fill"]
    missile_trace_color = (0.85, 0.75, 0.1, 1)
    target_line_color = (0.5, 0.6, 0.1, 0.5)

    drawables = []

    if unit.id in trace_record:
        trace = [(pos.lat, pos.lon) for _, pos, _, _ in trace_record[unit.id]]
        drawables.append(
            PolyLine(
                points=trace,
                line_width=1.5,
                dash=(2, 2),
                edge_color=missile_trace_color,
                zorder=4,
            )
        )

    drawables.append(
        DrawableMissile(
            lat=unit.position.lat,
            lon=unit.position.lon,
            yaw_deg=unit.yaw_deg,
            edge_color=outline,
            fill_color=fill,
            info_text=None,
            zorder=9,
            affiliation=affiliation,
        )
    )

    target = getattr(unit, "target", None)
    target_is_live = (
        target is not None
        and not getattr(target, "is_countermeasure", False)
        and (active_unit_ids is None or getattr(target, "id", None) in active_unit_ids)
    )
    if target_is_live:
        target_pos = target.position

        # Target marker and line
        drawables.extend(
            [
                Arc(
                    center_lat=target_pos.lat,
                    center_lon=target_pos.lon,
                    radius=0.2,
                    angle1=0,
                    angle2=360,
                    edge_color=outline,
                    fill_color=(outline[0], outline[1], outline[2], 0.5),
                    line_width=1,
                    zorder=5,
                ),
                PolyLine(
                    points=[
                        (unit.position.lat, unit.position.lon),
                        (target_pos.lat, target_pos.lon),
                    ],
                    line_width=1.5,
                    dash=(4, 4),
                    edge_color=target_line_color,
                    zorder=5,
                ),
            ]
        )
    return drawables


_CM_COLORS = {
    "flare": (1.0, 1.0, 0.0),  # pure yellow
    "chaff": (1.0, 0.85, 0.0),  # golden yellow
    "decoy": (1.0, 0.65, 0.0),  # amber
}
_CM_RADIUS_KM = 1.5  # visual dot radius in km


def plot_countermeasure(unit, _side: str) -> list:
    """Draw a yellow dot for an active countermeasure (flare / chaff / decoy).

    Opacity fades from full to 25 % as the countermeasure ages toward expiry.
    """
    cm_type = getattr(unit, "cm_type", "flare")
    age_s = getattr(unit, "age_s", 0.0)
    lifetime = getattr(unit, "lifetime_s", 5.0)
    fade = 1.0 - age_s / max(lifetime, 0.001)
    alpha = max(0.25, fade)
    r, g, b = _CM_COLORS.get(cm_type, (1.0, 1.0, 0.0))
    return [
        Arc(
            center_lat=unit.position.lat,
            center_lon=unit.position.lon,
            radius=_CM_RADIUS_KM,
            angle1=0,
            angle2=360,
            fill_color=(r, g, b, alpha * 0.85),
            edge_color=(r * 0.7, g * 0.7, b * 0.7, 1.0),
            line_width=1,
            zorder=6,
        )
    ]


def _find_50pct_range_km(lut, max_range_m: float, rcs: float) -> float:
    """Binary-search for the range (km) at which detection probability = 50 %.

    Returns the full max_range if Pd ≥ 0.5 even at max range; returns a very
    short range (1 km) if the target is essentially undetectable even at close
    quarters.
    """
    lo, hi = 1_000.0, max_range_m
    if lut.get_probability(lo, rcs) < 0.5:
        return lo / 1000.0  # undetectable — minimal cone
    if lut.get_probability(hi, rcs) >= 0.5:
        return hi / 1000.0  # easy target — full cone
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if lut.get_probability(mid, rcs) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0 / 1000.0


def plot_radar_cone(unit, side: str, enemy_rcs: float | None = None):
    """Display radar cone based on the new radar class.

    The radius always represents the configured instrumented radar range.
    When *enemy_rcs* is provided, detection probability affects opacity only;
    changing the geometry to the least-observable opponent made the displayed
    range misleadingly small (most visibly for the Su-57).
    """
    radar = getattr(unit, "radar", None)
    if radar is None:
        return []

    affiliation = SIDE_AFFILIATION.get(side, "unknown")
    is_support = getattr(unit, "is_support_asset", False)

    if is_support:
        # AWACS: distinct cyan/orange palette so cones don't blend with fighters.
        # AWACS detection is omnidirectional. Keep 360 as an unnormalised end
        # angle so Cairo draws a full circle rather than an empty path.
        color_table = SUPPORT_RADAR_COLORS
        angle1 = 0.0
        angle2 = 360.0
        radius = min(radar.max_range_m / 1000.0, 250.0)
        fill_alpha = 0.15
    else:
        color_table = AFFILIATION_COLORS
        delta = radar.h_fov_deg / 2
        radius = radar.max_range_m / 1000.0
        if enemy_rcs is not None:
            pd_at_max = radar.lut.get_probability(radar.max_range_m, enemy_rcs)
            fill_alpha = max(0.04, min(0.20, 0.20 * pd_at_max))
        else:
            fill_alpha = 0.15

        angle1 = (unit.yaw_deg - delta + 360) % 360
        angle2 = (unit.yaw_deg + delta) % 360

    outline = color_table[affiliation]["outline"]
    fill = color_table[affiliation]["fill"]

    return [
        RadarCone(
            center_lat=unit.position.lat,
            center_lon=unit.position.lon,
            radius=radius,
            angle1=angle1,
            angle2=angle2,
            edge_color=outline,
            fill_color=(fill[0], fill[1], fill[2], fill_alpha),
            line_width=2,
            zorder=2,
        )
    ]


def plot_simulation(simulator, output_file: Path | str, map_limits: MapLimits):
    """Save current simulation state as PNG."""
    plotter = ScenarioPlotter(map_extents=map_limits, config=PlotConfig())
    drawables = []

    for unit in simulator.active_units.values():
        side = "blue" if unit.group == "agent" else "red"
        if getattr(unit, "unit_kind", None) == "aircraft":
            drawables.extend(plot_aircraft(unit, side, simulator.trace_record_units))
        elif getattr(unit, "unit_kind", None) == "missile":
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


def _try_create_symbol_registry(mode: str = "nato"):
    """Try to create a SymbolRegistry; return None if symbols are unavailable or mode is procedural."""
    if mode == "procedural":
        return None
    try:
        from bvr_marl_core.visualization.scenplotter import SymbolRegistry

        registry = SymbolRegistry(mode=mode)
        if registry.available_symbols():
            return registry
    except (ImportError, AttributeError, TypeError):
        pass
    return None


def live_simulation(
    trained_model,
    env: SimulationEnv,
    frames: int = 200,
    interval: int = 100,
    save_video: bool = False,
    video_output_file: Path = Path("output/optics_test_run.mp4"),
    scenario_name: str | None = None,
    show_line_overlay: bool = False,
    line_east_km: float | None = None,
    map_extents_mode: str | None = None,
    symbol_mode: str = "nato",
    dpi: int = 200,
    symbol_scale: float = 2.0,
    show_radar_cones: bool = True,
    show_text: bool = True,
    focus_unit_id: str | None = None,
    video_focus_rule: str = "none",
):
    """Runs a live simulation and displays/saves a video."""
    render_offscreen = False
    if not can_show_matplotlib_window(plt.get_backend()):
        if not save_video:
            raise RuntimeError(
                "Live visualization requires an interactive Matplotlib backend and a display, "
                f"but the active backend is {plt.get_backend()!r}. On desktop hosts, set "
                "MPLBACKEND=TkAgg or install a GUI backend. On headless Linux, run under "
                "X11/Wayland/Xvfb or pass --save-video to render off-screen."
            )

        print(
            f"Matplotlib backend {plt.get_backend()!r} cannot open a live window; "
            "rendering frames off-screen because --save-video is enabled."
        )
        plt.switch_backend("Agg")
        render_offscreen = True

    observations, _ = env.reset()

    # Try to load symbols for the chosen mode
    symbol_registry = _try_create_symbol_registry(mode=symbol_mode)

    # Use full_map_limits (includes AWACS side zones) when available
    display_limits = select_display_limits(
        env,
        scenario_name,
        map_extents_mode=map_extents_mode,
    )
    line_of_engagement_east_m = None if line_east_km is None else float(line_east_km) * 1000.0
    plotter, _ = initialize_plotter(
        display_limits,
        symbol_registry=symbol_registry,
        symbol_mode=symbol_mode,
        dpi=dpi,
        symbol_scale=symbol_scale,
        show_text=show_text,
    )

    # Calculate aspect ratio from map limits to properly display non-square maps
    map_width = abs(display_limits.right_lon - display_limits.left_lon)
    map_height = abs(display_limits.top_lat - display_limits.bottom_lat)
    aspect_ratio = map_width / map_height if map_height > 0 else 1.0

    # Scale the display window proportionally to DPI (base height 10 at 200 dpi)
    fig_height = 10 * (dpi / 200)
    fig_width = fig_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # image fills entire figure
    ax.axis("off")
    # Initialise with the correct frame size so the axes never resizes on frame 0.
    img = ax.imshow(
        np.zeros((plotter.img_height, plotter.img_width, 4), dtype=np.uint8),
        interpolation="none",
    )
    # Aircraft are rendered into the Cairo frame, so this transparent collection
    # supplies native Matplotlib hit testing without duplicating visible symbols.
    hit_artist = ax.scatter([], [], s=900, facecolors="none", edgecolors="none", picker=True)
    hit_unit_ids: list[object] = []
    selected_unit_id: list[object | None] = [focus_unit_id]
    label_manager = FighterLabelManager()

    def _select_from_event(event) -> None:
        if event.inaxes is not ax or not hit_unit_ids:
            return
        contains, details = hit_artist.contains(event)
        indices = details.get("ind", ()) if contains else ()
        if len(indices):
            selected_unit_id[0] = hit_unit_ids[int(indices[0])]
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _select_from_event)
    fig.canvas.mpl_connect("button_press_event", _select_from_event)

    previous_aircraft = {}
    status_message_queue = []
    action_states = {}  # Store action states per unit
    cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}  # Track cumulative rewards
    current_step_rewards = {}  # Track rewards from current step

    # Frame buffer for in-flight video recording. Frames are appended during
    # live playback and flushed only when an episode ends, so each completed
    # episode produces exactly one video file.
    _frame_buffer: list = []
    _episode_index: list = [0]  # mutable int-wrapper (no nonlocal needed)
    _fps = max(1, 1000 // interval)

    def _flush_frame_buffer(output_base: Path) -> None:
        """Save the current frame buffer to a GIF (or MP4 via ffmpeg) and clear it."""
        if not _frame_buffer:
            return
        ep = _episode_index[0]
        _episode_index[0] += 1
        suffix = f"_ep{ep:03d}"
        try:
            if animation.writers.is_available("ffmpeg"):
                import imageio  # type: ignore

                out_path = output_base.with_stem(output_base.stem + suffix)
                imageio.mimwrite(str(out_path), [f[:, :, :3] for f in _frame_buffer], fps=_fps)
                print(f"Video saved (episode {ep}): {out_path}")
            else:
                from PIL import Image  # type: ignore

                out_path = output_base.with_stem(output_base.stem + suffix).with_suffix(".gif")
                pil_frames = [Image.fromarray(f[:, :, :3]) for f in _frame_buffer]
                duration_ms = max(1, 1000 // _fps)
                pil_frames[0].save(
                    str(out_path),
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
                print(f"Video saved (episode {ep}): {out_path}  (GIF - install ffmpeg for MP4)")
        except Exception as exc:
            print(f"Warning: could not save episode {ep} video: {exc}")
        finally:
            _frame_buffer.clear()

    def update_frame(_frame_index):
        nonlocal \
            observations, \
            previous_aircraft, \
            action_states, \
            cumulative_rewards, \
            current_step_rewards

        # Check if episode is done
        if not env.agents or len(env.agents) == 0:
            # Episode ended, reset the environment
            observations, _ = env.reset()
            previous_aircraft.clear()
            action_states.clear()
            cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}
            current_step_rewards.clear()

        # Compute actions
        actions = {}
        for agent_id in env.agents:
            if agent_id in observations:
                actions[agent_id] = trained_model.compute_single_action(
                    observations[agent_id], agent_id
                )

        observations, rewards, terminateds, truncateds, _ = env.step(actions)

        # Update reward tracking
        current_step_rewards = rewards.copy()
        for agent_id, reward in rewards.items():
            if agent_id != "__all__":  # Skip the special "__all__" key
                cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + reward

        # Extract action states from environment (if available)
        if hasattr(env, "action_processor") and hasattr(env.action_processor, "state"):
            action_states = env.action_processor.state.copy()

        episode_done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        # Collect drawables
        drawables = build_scenario_overlay_drawables(
            display_limits,
            scenario_name,
            force_line_overlay=show_line_overlay,
            line_of_engagement_east_m=line_of_engagement_east_m,
        )
        current_aircraft = {}
        unit_to_agent = {uid: aid for aid, uid in env.agent_to_unit_id.items()}
        active_units = list(env.simulator.active_units.values())
        active_aircraft_ids = {
            unit.id for unit in active_units if getattr(unit, "unit_kind", None) == "aircraft"
        }
        configured_focus = next(
            (
                unit_id
                for unit_id in active_aircraft_ids
                if focus_unit_id is not None and str(unit_id) == str(focus_unit_id)
            ),
            None,
        )
        if save_video:
            render_focus_id = configured_focus or _deterministic_focus(
                active_units, video_focus_rule
            )
        else:
            render_focus_id = selected_unit_id[0]
            if render_focus_id not in active_aircraft_ids:
                render_focus_id = configured_focus
                selected_unit_id[0] = render_focus_id
        threat_ids = {
            getattr(getattr(unit, "target", None), "id", None)
            for unit in active_units
            if getattr(unit, "unit_kind", None) == "missile"
        }

        # Pre-compute the minimum (most stealthy) enemy RCS for each side so
        # that each fighter's radar cone reflects its detection capability
        # against the hardest-to-find opponent.
        _enemy_min_rcs: dict[str, float | None] = {"blue": None, "red": None}
        for _u in active_units:
            if getattr(_u, "unit_kind", None) == "aircraft" and not getattr(
                _u, "is_support_asset", False
            ):
                _rcs = getattr(_u, "rcs", 10.0)
                _side_key = "blue" if _u.group == "agent" else "red"
                _enemy_key = "red" if _side_key == "blue" else "blue"
                cur = _enemy_min_rcs[_enemy_key]
                if cur is None or _rcs < cur:
                    _enemy_min_rcs[_enemy_key] = _rcs

        map_labels = []
        hit_positions = []
        hit_ids = []
        for unit in active_units:
            side = "blue" if unit.group == "agent" else "red"
            if getattr(unit, "unit_kind", None) == "aircraft":
                # Build action state with agent name injected
                unit_action_state = dict(action_states.get(unit.id, {}))
                agent_id = unit_to_agent.get(unit.id, str(unit.id))
                unit_action_state["agent_name"] = agent_id.upper()
                priority = 2 if unit.id == render_focus_id else (1 if unit.id in threat_ids else 0)
                aircraft_drawables = plot_aircraft(
                    unit,
                    side,
                    env.simulator.trace_record_units,
                    action_state=unit_action_state,
                    label_priority=priority,
                    label_detail=(
                        "full" if priority == 2 else ("short" if priority == 1 else "compact")
                    ),
                )
                drawables.extend(aircraft_drawables)
                map_labels.extend(
                    drawable for drawable in aircraft_drawables if isinstance(drawable, MapLabel)
                )
                if show_radar_cones:
                    drawables.extend(
                        plot_radar_cone(unit, side, enemy_rcs=_enemy_min_rcs.get(side))
                    )
                current_aircraft[unit.id] = True
                px, py, _ = plotter._get_image_xya(
                    unit.position.lat, unit.position.lon, unit.yaw_deg
                )
                hit_positions.append((px, plotter.img_height - py))
                hit_ids.append(unit.id)
            elif getattr(unit, "unit_kind", None) == "missile":
                drawables.extend(
                    plot_missile(
                        unit,
                        side,
                        env.simulator.trace_record_units,
                        active_unit_ids=set(env.simulator.active_units.keys()),
                    )
                )
            elif getattr(unit, "is_countermeasure", False):
                drawables.extend(plot_countermeasure(unit, side))

        if show_text:
            label_manager.layout(map_labels, plotter)
        if render_focus_id in active_aircraft_ids:
            focused = next(unit for unit in active_units if unit.id == render_focus_id)
            side = "blue" if focused.group == "agent" else "red"
            outline = AFFILIATION_COLORS[SIDE_AFFILIATION.get(side, "unknown")]["outline"]
            drawables.append(
                Arc(
                    center_lat=focused.position.lat,
                    center_lon=focused.position.lon,
                    radius=max(2.0, plotter.cfg.symbol_size * 0.7 / plotter.pixels_km),
                    angle1=0,
                    angle2=360,
                    edge_color=(1.0, 1.0, 1.0, 0.95),
                    fill_color=(outline[0], outline[1], outline[2], 0.10),
                    line_width=2.0,
                    zorder=11,
                )
            )

        hit_unit_ids[:] = hit_ids
        hit_artist.set_offsets(
            np.asarray(hit_positions, dtype=float)
            if hit_positions
            else np.empty((0, 2), dtype=float)
        )

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
        status_lines.extend(
            build_scenario_status_lines(
                scenario_name,
                force_line_overlay=show_line_overlay,
                line_of_engagement_east_m=line_of_engagement_east_m,
            )
        )

        # Calculate team rewards
        agent_team_reward = sum(
            cumulative_rewards.get(aid, 0.0) for aid in env.agent_ids if aid in cumulative_rewards
        )
        opponent_team_reward = sum(
            cumulative_rewards.get(oid, 0.0)
            for oid in env.opponent_ids
            if oid in cumulative_rewards
        )
        agent_step_reward = sum(
            current_step_rewards.get(aid, 0.0)
            for aid in env.agent_ids
            if aid in current_step_rewards
        )
        opponent_step_reward = sum(
            current_step_rewards.get(oid, 0.0)
            for oid in env.opponent_ids
            if oid in current_step_rewards
        )

        status_lines.append(f"Blue Team: {agent_team_reward:+.1f} (step: {agent_step_reward:+.1f})")
        status_lines.append(
            f"Red Team: {opponent_team_reward:+.1f} (step: {opponent_step_reward:+.1f})"
        )
        if render_focus_id in active_aircraft_ids:
            focused = next(unit for unit in active_units if unit.id == render_focus_id)
            agent_id = unit_to_agent.get(focused.id, str(focused.id))
            status_lines.extend(
                [
                    "",
                    "FOCUS",
                    *_aircraft_label_text(
                        focused,
                        {"agent_name": str(agent_id).upper()},
                        detail="full",
                    ).splitlines(),
                ]
            )

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

        rgba = plotter.to_rgba(drawables)
        img.set_data(rgba)
        if save_video:
            _frame_buffer.append(rgba.copy())
            if episode_done:
                _flush_frame_buffer(video_output_file)

        if episode_done:
            observations, _ = env.reset()
            previous_aircraft.clear()
            action_states.clear()
            cumulative_rewards = {agent_id: 0.0 for agent_id in env.agents}
            current_step_rewards.clear()
        return img, hit_artist

    if save_video:
        video_output_file.parent.mkdir(parents=True, exist_ok=True)
        print("Video recording enabled - one file is saved for each completed episode.")
        print(f"Output directory: {video_output_file.parent}")

        def _on_close(_event):
            if _frame_buffer:
                print("Window closed before episode end - discarding partial video buffer.")
                _frame_buffer.clear()

        fig.canvas.mpl_connect("close_event", _on_close)

    # Live interactive view — frames are recorded and flushed at episode end
    # when save_video=True (see _flush_frame_buffer above).
    if render_offscreen:
        for frame_index in range(frames):
            update_frame(frame_index)
        if _frame_buffer:
            print("Frame limit reached before episode end - saving partial video buffer.")
            _flush_frame_buffer(video_output_file)
        plt.close(fig)
        return None

    anim = animation.FuncAnimation(fig, update_frame, frames=frames, interval=interval, blit=True)
    plt.show()
    return anim  # keep reference alive so GC doesn't destroy the animation
