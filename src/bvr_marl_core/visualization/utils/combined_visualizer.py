"""
Combined visualizer showing 2D simulation and attention heatmaps side by side.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.simulator import MapLimits
from bvr_marl_core.visualization.scenario_overlays import (
    build_scenario_overlay_drawables,
    build_scenario_status_lines,
    normalize_visualization_scenario,
    select_display_limits,
)
from bvr_marl_core.visualization.scenplotter import (
    PlotConfig,
    ScaleBar,
    ScenarioPlotter,
    StatusMessage,
    TopLeftMessage,
)
from bvr_marl_core.visualization.scenplotter.video_generation import (
    plot_aircraft,
    plot_countermeasure,
    plot_missile,
    update_status_message_queue,
)


class CombinedVisualizer:
    """Combined visualizer showing 2D sim and attention heatmaps."""

    def __init__(
        self,
        env=None,
        sim=None,
        scenario_name: str | None = None,
        force_line_overlay: bool = False,
        line_east_km: float | None = None,
        map_extents_mode: str | None = None,
        symbol_mode: str = "nato",
        dpi: int = 200,
        symbol_scale: float = 2.0,
        attention_extractor=None,
        num_agents=4,
        figsize=(20, 10),
    ):
        """Initialize the combined visualizer."""
        # Store params for run_simulation()
        self._env = env
        self._sim = sim
        self._scenario_name = normalize_visualization_scenario(scenario_name)
        self._force_line_overlay = bool(force_line_overlay)
        self._line_of_engagement_east_m = (
            None if line_east_km is None else float(line_east_km) * 1000.0
        )
        self._map_extents_mode = map_extents_mode
        self._symbol_mode = symbol_mode
        self._dpi = dpi
        self._symbol_scale = symbol_scale
        self._attention_extractor = attention_extractor

        self.num_agents = num_agents

        # Create figure with custom layout
        self.fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Left panel: 2D simulation (spans both rows)
        self.ax_sim = self.fig.add_subplot(gs[:, 0])
        self.ax_sim.axis("off")

        # Right panels: 2x2 grid for agent attention heatmaps
        self.ax_agents = []
        gs_right = gs[:, 1].subgridspec(2, 2, hspace=0.25, wspace=0.2)
        for i in range(min(4, num_agents)):
            row = i // 2
            col = i % 2
            ax = self.fig.add_subplot(gs_right[row, col])
            self.ax_agents.append(ax)

        # Attention visualization state
        self.colorbars = []
        # Map to keep agent IDs assigned to specific grid positions
        self.agent_to_position = {}

        # 2D simulation state
        self.plotter = None
        self.map_limits = None
        self.message_queue = []
        self.img_sim = None

    def init_2d_sim(
        self,
        map_limits: MapLimits | None = None,
        symbol_mode: str = "nato",
        dpi: int = 200,
        symbol_scale: float = 2.0,
    ):
        """Initialize the 2D simulation plotter."""
        if map_limits is None:
            # Default fallback for older call sites that do not provide explicit limits.
            self.map_limits = MapLimits(
                left_lon=-150.0 / 111.0,
                bottom_lat=-150.0 / 111.0,
                right_lon=150.0 / 111.0,
                top_lat=150.0 / 111.0,
            )
        else:
            self.map_limits = map_limits

        # Try to load symbol registry for the chosen mode
        symbol_registry = None
        if symbol_mode != "procedural":
            try:
                from bvr_marl_core.visualization.scenplotter import SymbolRegistry

                reg = SymbolRegistry(mode=symbol_mode)
                if reg.available_symbols():
                    symbol_registry = reg
            except Exception:
                pass

        # Initialize plotter with dark tactical palette
        plot_config = PlotConfig()
        plot_config.units_scale = 20.0
        plot_config.background_color = "#1a2535"
        plot_config.borders_color = "#8090a8"
        plot_config.coastlines_color = "#5a6e82"
        plot_config.grid_color = "#25364a"
        plot_config.symbol_mode = symbol_mode
        plot_config.symbol_size = max(32, round(32 * symbol_scale))
        plot_config.missile_symbol_size = max(20, round(20 * symbol_scale * 0.5))
        plot_config.symbol_scale = 1.0
        plot_config.missile_symbol_scale = 1.0
        plot_config.sprites_info_font_size = max(10, round(10 * symbol_scale))
        plot_config.sprites_info_spacing = max(26, round(26 * symbol_scale))

        self.plotter = ScenarioPlotter(
            map_extents=self.map_limits,
            dpi=dpi,
            config=plot_config,
            symbol_registry=symbol_registry,
        )

        # Initialise with correct dimensions to avoid layout jump on first frame
        self.img_sim = self.ax_sim.imshow(
            np.zeros((self.plotter.img_height, self.plotter.img_width, 4), dtype=np.uint8),
            interpolation="none",
        )

    def reset_episode(self):
        """Reset the visualizer state for a new episode."""
        # Clear agent position mapping so it gets re-initialized with new episode agents
        self.agent_to_position.clear()
        self.message_queue.clear()

    def update_2d_sim(self, env: BVRMultiAgentEnv, timestep: int):
        """Update the 2D simulation view."""
        if self.plotter is None:
            return

        # Collect drawable objects
        drawables = build_scenario_overlay_drawables(
            self.map_limits,
            self._scenario_name,
            force_line_overlay=self._force_line_overlay,
            line_of_engagement_east_m=self._line_of_engagement_east_m,
        )

        # Create reverse mapping from unit_id to agent_id
        unit_to_agent = {uid: aid for aid, uid in env.agent_to_unit_id.items()}

        # Plot units with agent names
        for unit in env.simulator.active_units.values():
            side = "blue" if unit.group == "agent" else "red"
            if getattr(unit, "unit_kind", None) == "aircraft":
                action_state = None
                if hasattr(env, "action_processor") and hasattr(env.action_processor, "state"):
                    action_state = env.action_processor.state.get(unit.id, None)

                # Add agent name to action_state for all aircraft
                if action_state is None:
                    action_state = {}
                # Get agent_id from unit_id and format to uppercase (a0 -> A0)
                agent_id = unit_to_agent.get(unit.id, str(unit.id))
                action_state["agent_name"] = agent_id.upper()

                drawables.extend(
                    plot_aircraft(
                        unit, side, env.simulator.trace_record_units, action_state=action_state
                    )
                )
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

        # Add status messages
        status_text = update_status_message_queue(
            env.simulator, [], self.message_queue, retention_time=5
        )
        if status_text:
            drawables.append(StatusMessage(text=status_text))

        # Add timestep info
        status_lines = [f"Step: {timestep}", f"Duration: {env.simulator.elapsed_time_s:.1f} s"]
        status_lines.extend(
            build_scenario_status_lines(
                self._scenario_name,
                force_line_overlay=self._force_line_overlay,
                line_of_engagement_east_m=self._line_of_engagement_east_m,
            )
        )
        drawables.append(TopLeftMessage(text="\n".join(status_lines)))

        # Add scale bar
        margin = 20
        scale_length = 50
        scale_bar = ScaleBar(
            length_km=scale_length,
            x=self.plotter.img_width
            - margin
            - (self.plotter._get_image_distance(scale_length) / 2),
            y=margin,
            line_width=2,
            edge_color=(1, 1, 1, 1),
            zorder=50,
        )
        drawables.append(scale_bar)

        self.img_sim.set_data(self.plotter.to_rgba(drawables))

    def update_attention(self, attention_data: dict[str, dict[str, np.ndarray]], timestep: int):
        """Update attention heatmaps for all agents."""
        # Initialize agent-to-position mapping on first call (sorted alphabetically)
        if not self.agent_to_position:
            agent_ids = sorted(attention_data.keys())
            for idx, agent_id in enumerate(agent_ids[: min(4, len(self.ax_agents))]):
                self.agent_to_position[agent_id] = idx

        # Update each agent's attention in their fixed position
        for agent_id, idx in self.agent_to_position.items():
            ax = self.ax_agents[idx]
            ax.clear()

            # Check if agent is still alive (present in attention_data)
            if agent_id in attention_data:
                # Get attention data
                attn = attention_data[agent_id]

                # Extract the 3 high-level attention values
                own_attn = attn.get("own_state", np.array([0.0]))[0]
                friendly_attn = attn.get("friendly", np.array([0.0]))[0]
                enemy_attn = attn.get("enemy", np.array([0.0]))[0]

                # Create single row with 3 attention values
                attention_values = np.array([[own_attn, friendly_attn, enemy_attn]])

                # Plot heatmap with red-to-green colormap
                im = ax.imshow(
                    attention_values,
                    aspect="auto",
                    cmap="RdYlGn",
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )

                # Add colorbar
                if idx >= len(self.colorbars):
                    cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Attention", fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                    self.colorbars.append(cbar)

                # Set labels
                ax.set_yticks([0])
                ax.set_yticklabels(["Attention"], fontsize=9)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(["Own", "Friendly", "Enemy"], fontsize=10, fontweight="bold")
                ax.set_title(f"{agent_id}", fontsize=11, fontweight="bold")

                # Add value text on each cell
                for col in range(3):
                    value = attention_values[0, col]
                    text_color = "white" if value < 0.5 else "black"
                    ax.text(
                        col,
                        0,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=12,
                        fontweight="bold",
                    )

                # Add grid
                ax.set_xticks(np.arange(3) - 0.5, minor=True)
                ax.set_yticks(np.arange(1) - 0.5, minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
            else:
                # Agent is dead - show a grayed-out placeholder
                attention_values = np.array([[0.0, 0.0, 0.0]])
                im = ax.imshow(
                    attention_values,
                    aspect="auto",
                    cmap="gray",
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )

                # Set labels
                ax.set_yticks([0])
                ax.set_yticklabels(["Attention"], fontsize=9)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(["Own", "Friendly", "Enemy"], fontsize=10, fontweight="bold")
                ax.set_title(f"{agent_id} (DEAD)", fontsize=11, fontweight="bold", color="red")

                # Add "DEAD" text
                ax.text(
                    1,
                    0,
                    "DEAD",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=14,
                    fontweight="bold",
                )

                # Add grid
                ax.set_xticks(np.arange(3) - 0.5, minor=True)
                ax.set_yticks(np.arange(1) - 0.5, minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    # Top-level simulation runner

    def run_simulation(
        self, model, num_frames: int = 200, interval: int = 100, save_video: bool = False
    ):
        """Run the full simulation loop with FuncAnimation."""
        from matplotlib import animation

        env = self._env
        observations, _ = env.reset()

        display_limits = select_display_limits(
            env,
            self._scenario_name,
            map_extents_mode=self._map_extents_mode,
        )

        self.init_2d_sim(
            map_limits=display_limits,
            symbol_mode=self._symbol_mode,
            dpi=self._dpi,
            symbol_scale=self._symbol_scale,
        )

        # Show placeholder in attention panels when no model is loaded
        if self._attention_extractor is None:
            agent_ids = sorted(env.agents)
            for idx, ax in enumerate(self.ax_agents):
                ax.set_facecolor("#1a1a1a")
                ax.set_xticks([])
                ax.set_yticks([])
                label = agent_ids[idx].upper() if idx < len(agent_ids) else f"Agent {idx}"
                ax.set_title(label, fontsize=11, fontweight="bold", color="white")
                ax.text(
                    0.5,
                    0.5,
                    "No model loaded\nLoad checkpoint\nfor attention data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="#888888",
                    style="italic",
                )
            self.fig.patch.set_facecolor("#1a2535")

        timestep = [0]

        def update_frame(_frame_index):
            nonlocal observations

            if not env.agents:
                observations, _ = env.reset()
                self.reset_episode()
                timestep[0] = 0

            actions = {
                agent_id: model.compute_single_action(observations[agent_id], agent_id)
                for agent_id in env.agents
                if agent_id in observations
            }
            observations, _, terminateds, truncateds, _ = env.step(actions)
            timestep[0] += 1

            if terminateds.get("__all__", False) or truncateds.get("__all__", False):
                observations, _ = env.reset()
                self.reset_episode()
                timestep[0] = 0

            self.update_2d_sim(env, timestep[0])

            if self._attention_extractor is not None:
                try:
                    attention_data = self._attention_extractor.extract(observations)
                    if attention_data:
                        self.update_attention(attention_data, timestep[0])
                except Exception:
                    pass

            return []

        anim = animation.FuncAnimation(
            self.fig, update_frame, frames=num_frames, interval=interval, blit=False
        )

        if save_video:
            from pathlib import Path

            out = Path("output/attention_viz.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            fps = max(1, 1000 // interval)
            try:
                writer = animation.FFMpegWriter(fps=fps)
                anim.save(str(out), writer=writer)
                print(f"Saved to {out}")
            except Exception as e:
                print(f"Video save failed: {e}")
        else:
            plt.tight_layout()
            plt.show()
