"""
2D simulation visualizer with RL command time-series side panel.

Right panel: one subplot per RL fighter agent, each showing
  Ps / n / phi  normalised to [0, 1] (matching the [0,1] network output range).
AWACS and other support assets are not tracked.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from bvr_marl_core.rl.environment.gym.bvr_multi_agent_env import BVRMultiAgentEnv
from bvr_marl_core.rl.environment.spaces.action_space.processors import (
    EnergyProcessor,
    LiftVectorProcessor,
)
from bvr_marl_core.simulator import MapLimits
from bvr_marl_core.visualization.scenario_overlays import (
    build_scenario_overlay_drawables,
    build_scenario_status_lines,
    normalize_visualization_scenario,
    select_display_limits,
)
from bvr_marl_core.visualization.scenplotter import (
    ScaleBar,
    StatusMessage,
    TopLeftMessage,
)
from bvr_marl_core.visualization.scenplotter.video_generation import (
    _try_create_symbol_registry,
    initialize_plotter,
    plot_aircraft,
    plot_countermeasure,
    plot_missile,
    plot_radar_cone,
    update_status_message_queue,
)

# ps_cmd_filtered  <- action_Ps in [0,1], centered so 0.5 means Ps=0.
#   We use a representative symmetric range; clamped to [0,1] after normalisation.
_PS_MIN, _PS_MAX = -150.0, 150.0  # m/s
# n_cmd_filtered   <- action_n in [0,1], centered so 0.5 means 1g.
_N_MIN, _N_MAX = -2.0, 9.0  # g
# phi_cmd_filtered <- action_phi in [0,1], centered so 0.5 means wings level.
_PHI_MIN, _PHI_MAX = -45.0, 45.0  # degrees

# Command display config
_CMDS = [
    ("ps", "Ps", "#4FC3F7"),  # cyan
    ("n", "n", "#81C784"),  # green
    ("phi", "φ", "#FFB74D"),  # orange
]


def _norm(value: float, lo: float, hi: float) -> float:
    """Clamp-normalise value into [0, 1]."""
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _normalize_cmd(cmd: str, value: float) -> float:
    if cmd == "ps":
        return EnergyProcessor.centered_ps_to_action(value, _PS_MIN, _PS_MAX)
    if cmd == "n":
        return LiftVectorProcessor.centered_load_factor_to_action(value, _N_MIN, _N_MAX)
    if cmd == "phi":
        return _norm(value, _PHI_MIN, _PHI_MAX)
    return value


class RLCommandsVisualizer:
    """2D map on the left; one command-plot per RL fighter agent on the right."""

    def __init__(
        self,
        env=None,
        sim=None,
        map_limits=None,
        scenario_name: str | None = None,
        force_line_overlay: bool = False,
        line_east_km: float | None = None,
        map_extents_mode: str | None = None,
        symbol_mode: str = "nato",
        dpi: int = 200,
        symbol_scale: float = 2.0,
        show_text: bool = True,
        num_agents: int = 4,
        max_history: int = 600,
        figsize: tuple = (22, 9),
    ):
        # Store env/sim params for run_simulation()
        self._env = env
        self._sim = sim
        self._map_limits = map_limits
        self._scenario_name = normalize_visualization_scenario(scenario_name)
        self._force_line_overlay = bool(force_line_overlay)
        self._line_of_engagement_east_m = (
            None if line_east_km is None else float(line_east_km) * 1000.0
        )
        self._map_extents_mode = map_extents_mode
        self._symbol_mode = symbol_mode
        self._dpi = dpi
        self._symbol_scale = symbol_scale
        self._show_text = bool(show_text)

        self.num_agents = num_agents
        self.max_history = max_history

        self.fig = plt.figure(figsize=figsize, facecolor="#1e1e1e")
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1.3, 1], wspace=0.35)

        # Left: 2D simulation map
        self.ax_sim = self.fig.add_subplot(gs[0, 0])
        self.ax_sim.axis("off")

        # Right: one subplot per fighter agent (created lazily on first frame)
        self._gs_right = gs[0, 1]
        self._agent_axes: dict[str, plt.Axes] = {}  # agent_id → Axes
        self._agent_order: list[str] = []  # insertion order
        self._subplots_created = False

        # 2D sim state
        self.plotter = None
        self.map_limits = None
        self.message_queue = []
        self.img_sim = None

        # History: {agent_id: {cmd: [values], 'time': [timestamps]}}
        self.history: dict = defaultdict(lambda: {"ps": [], "n": [], "phi": [], "time": []})

    # Subplot helpers

    def _create_agent_subplots(self, agent_ids: list[str]):
        """One-time creation of per-agent subplots after fighter list is known."""
        n = max(1, len(agent_ids))
        gs_right = self._gs_right.subgridspec(n, 1, hspace=0.55)
        for idx, aid in enumerate(agent_ids):
            ax = self.fig.add_subplot(gs_right[idx])
            self._style_ax(ax, aid)
            self._agent_axes[aid] = ax
        self._agent_order = list(agent_ids)
        self._subplots_created = True

    @staticmethod
    def _style_ax(ax: plt.Axes, agent_id: str):
        ax.set_facecolor("#2a2a2a")
        ax.set_xlim(left=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(agent_id, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.grid(True, color="#444444", linewidth=0.5, linestyle="--")
        ax.axhline(0.5, color="#666666", linewidth=0.5, linestyle=":")  # midpoint reference

    # Public API

    def init_2d_sim(
        self,
        map_limits: MapLimits,
        symbol_mode: str = "nato",
        dpi: int = 200,
        symbol_scale: float = 2.0,
    ):
        """Initialise the Cairo plotter for the left panel."""
        self.map_limits = map_limits
        symbol_registry = _try_create_symbol_registry(mode=symbol_mode)
        self.plotter, _ = initialize_plotter(
            map_limits=map_limits,
            symbol_registry=symbol_registry,
            symbol_mode=symbol_mode,
            dpi=dpi,
            symbol_scale=symbol_scale,
            show_text=self._show_text,
        )
        self.img_sim = self.ax_sim.imshow(
            np.zeros((self.plotter.img_height, self.plotter.img_width, 4), dtype=np.uint8),
            interpolation="none",
        )

    def reset_episode(self):
        """Clear history and redraw blank axes for a new episode."""
        self.history.clear()
        for aid, ax in self._agent_axes.items():
            ax.cla()
            self._style_ax(ax, aid)
        self.message_queue.clear()

    def update(self, env: BVRMultiAgentEnv, timestep: int):
        """Update map and command plots for one timestep."""
        self._update_2d_sim(env, timestep)
        self._update_rl_commands(env)
        self._redraw_command_plots()

    # Internal update steps

    def _is_fighter(self, unit) -> bool:
        return not getattr(unit, "is_support_asset", False)

    def _update_2d_sim(self, env: BVRMultiAgentEnv, timestep: int):
        if self.plotter is None:
            return

        drawables = build_scenario_overlay_drawables(
            self.map_limits,
            self._scenario_name,
            force_line_overlay=self._force_line_overlay,
            line_of_engagement_east_m=self._line_of_engagement_east_m,
        )
        unit_to_agent = {uid: aid for aid, uid in env.agent_to_unit_id.items()}

        # Pre-compute minimum enemy RCS per side for radar-cone scaling
        _enemy_min_rcs: dict[str, float | None] = {"blue": None, "red": None}
        for _u in env.simulator.active_units.values():
            if getattr(_u, "unit_kind", None) == "aircraft" and not getattr(
                _u, "is_support_asset", False
            ):
                _rcs = getattr(_u, "rcs", 10.0)
                _side_key = "blue" if _u.group == "agent" else "red"
                _enemy_key = "red" if _side_key == "blue" else "blue"
                cur = _enemy_min_rcs[_enemy_key]
                if cur is None or _rcs < cur:
                    _enemy_min_rcs[_enemy_key] = _rcs

        for unit in env.simulator.active_units.values():
            side = "blue" if unit.group == "agent" else "red"
            if getattr(unit, "unit_kind", None) == "aircraft":
                action_state = dict(
                    env.action_processor.state.get(unit.id, {})
                    if hasattr(env, "action_processor") and hasattr(env.action_processor, "state")
                    else {}
                )
                agent_id = unit_to_agent.get(unit.id, str(unit.id))
                action_state["agent_name"] = agent_id.upper()
                drawables.extend(
                    plot_aircraft(
                        unit,
                        side,
                        env.simulator.trace_record_units,
                        action_state=action_state,
                    )
                )
                drawables.extend(plot_radar_cone(unit, side, enemy_rcs=_enemy_min_rcs.get(side)))
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

        status_text = update_status_message_queue(
            env.simulator, [], self.message_queue, retention_time=5
        )
        if status_text:
            drawables.append(StatusMessage(text=status_text))
        status_lines = [f"Step: {timestep}", f"Duration: {env.simulator.elapsed_time_s:.1f} s"]
        status_lines.extend(
            build_scenario_status_lines(
                self._scenario_name,
                force_line_overlay=self._force_line_overlay,
                line_of_engagement_east_m=self._line_of_engagement_east_m,
            )
        )
        drawables.append(TopLeftMessage(text="\n".join(status_lines)))

        margin = 20
        scale_length = 50
        drawables.append(
            ScaleBar(
                length_km=scale_length,
                x=self.plotter.img_width
                - margin
                - (self.plotter._get_image_distance(scale_length) / 2),
                y=margin,
                line_width=2,
                edge_color=(1, 1, 1, 1),
                zorder=50,
            )
        )

        self.img_sim.set_data(self.plotter.to_rgba(drawables))

    def _update_rl_commands(self, env: BVRMultiAgentEnv):
        """Collect normalised command values for every RL fighter agent."""
        sim_time = env.simulator.elapsed_time_s
        unit_to_agent = {uid: aid for aid, uid in env.agent_to_unit_id.items()}
        action_states = (
            env.action_processor.state
            if hasattr(env, "action_processor") and hasattr(env.action_processor, "state")
            else {}
        )

        fighter_ids_seen = []
        for unit in env.simulator.active_units.values():
            if not getattr(unit, "unit_kind", None) == "aircraft" or not self._is_fighter(unit):
                continue
            agent_id = unit_to_agent.get(unit.id, str(unit.id)).upper()
            fighter_ids_seen.append(agent_id)

            state = action_states.get(unit.id, {})
            h = self.history[agent_id]
            h["time"].append(sim_time)
            h["ps"].append(_normalize_cmd("ps", state.get("ps_cmd_filtered", 0.0)))
            h["n"].append(_normalize_cmd("n", state.get("n_cmd_filtered", 1.0)))
            h["phi"].append(_normalize_cmd("phi", state.get("phi_cmd_filtered", 0.0)))

            # Rolling window
            if len(h["time"]) > self.max_history:
                for k in ("time", "ps", "n", "phi"):
                    h[k] = h[k][-self.max_history :]

        # Lazily create per-agent subplots on the first frame
        if not self._subplots_created and fighter_ids_seen:
            ordered = sorted(set(fighter_ids_seen))  # alphabetical: A0, A1, O0, O1
            self._create_agent_subplots(ordered)

    def _redraw_command_plots(self):
        """Redraw all per-agent subplots from history."""
        for agent_id in self._agent_order:
            ax = self._agent_axes[agent_id]
            h = self.history.get(agent_id)
            ax.cla()
            self._style_ax(ax, agent_id)

            if not h or not h["time"]:
                continue

            for cmd, label, color in _CMDS:
                ax.plot(h["time"], h[cmd], color=color, linewidth=1.3, label=label)

            ax.set_xlim(left=max(0.0, h["time"][0]), right=max(h["time"][-1], h["time"][0] + 1))
            ax.legend(
                loc="upper left",
                fontsize=7,
                framealpha=0.5,
                facecolor="#1e1e1e",
                edgecolor="#555555",
                labelcolor="white",
                ncol=3,
                handlelength=1.2,
            )

    # Top-level simulation runner

    def run_simulation(
        self, model, num_frames: int = 200, interval: int = 100, save_video: bool = False
    ):
        """Run the full simulation loop with FuncAnimation."""
        from matplotlib import animation

        env = self._env
        observations, _ = env.reset()

        # Resolve map limits from env if not provided at construction time
        map_limits = self._map_limits
        if map_limits is None:
            map_limits = select_display_limits(
                env,
                self._scenario_name,
                map_extents_mode=self._map_extents_mode,
            )

        self.init_2d_sim(map_limits, self._symbol_mode, self._dpi, self._symbol_scale)

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

            self.update(env, timestep[0])
            return []

        anim = animation.FuncAnimation(
            self.fig, update_frame, frames=num_frames, interval=interval, blit=False
        )

        if save_video:
            from pathlib import Path

            out = Path("output/rl_commands.mp4")
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
