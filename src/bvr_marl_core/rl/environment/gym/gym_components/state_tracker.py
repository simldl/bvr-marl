"""
State tracking for rewards and episode statistics.
"""

from __future__ import annotations

from bvr_marl_core.aircraft.systems.fire_veto import WASTED_VETO_CATEGORIES
from bvr_marl_core.domain.launch_geometry import TARGET_DESIGNATED_RATE
from bvr_marl_core.rl.environment.gym.gym_components.missile_diagnostics import (
    MissileDiagnosticsCollector,
)


class StateTracker:
    """Tracks state across steps for reward computation and episode statistics."""

    def __init__(self):
        # Position and missile tracking
        self.previous_positions: dict[str, tuple] = {}  # agent_id -> (lon, lat, alt)
        self.missiles_fired_last_step: dict[str, int] = {}  # agent_id -> count

        # Phase 2: Enhanced reward tracking
        self.prev_sqi: dict[str, float] = {}  # agent_id -> float
        self.no_lock_timer: dict[str, float] = {}  # agent_id -> float (seconds without lock)
        self.nez_steps: dict[str, int] = {}  # agent_id -> int (consecutive steps in NEZ)
        self.orbit_state: dict[
            str, dict
        ] = {}  # agent_id -> {dr_dt_history: deque, duration: float}
        self.time_with_lock: dict[str, float] = {}  # agent_id -> float (seconds with lock)
        self.had_lock_previous_step: dict[str, bool] = {}  # agent_id -> bool
        self.team_first_shot: dict[str, bool] = {}  # team -> bool
        self.prev_tactical_potential: dict[str, float] = {}  # agent_id -> float (V11 shaping)
        self.prev_energy_advantage: dict[str, float] = {}  # agent_id -> float (V11 shaping)

        # Episode statistics
        self.episode_missiles_fired: dict[str, int] = {}  # agent_id -> total missiles
        self.episode_kills: dict[str, int] = {}  # agent_id -> number of kills
        self.episode_missile_kills: dict[str, int] = {}  # agent_id -> kills credited to missiles
        self.episode_deaths: dict[str, int] = {}  # agent_id -> 0 or 1

        # Episode diagnostic metrics
        self.episode_valid_missile_shots: dict[str, int] = {}  # agent_id -> total valid shots
        self.episode_vetoed_missile_shots: dict[str, int] = {}  # agent_id -> total vetoed attempts
        # Veto attribution (subtotals of episode_vetoed_missile_shots):
        #   suppressed = trigger pressed but firing not permissible (cooldown, per-target
        #     cap, winchester) -> doctrine/safety, not a policy error.
        #   no_target  = trigger pressed while the selection action addressed an empty
        #     contact slot -> a policy error, tracked separately so it is visible and
        #     penalizable instead of hiding inside `suppressed`.
        #   wasted     = permissible trigger press rejected by geometry (FOV/range/lock)
        #     -> a policy error (fired on an invalid solution).
        self.episode_vetoed_missile_suppressed: dict[str, int] = {}
        self.episode_vetoed_missile_no_target: dict[str, int] = {}
        self.episode_vetoed_missile_wasted: dict[str, int] = {}
        # `wasted` split by the gate that rejected the launch (fov/range/lock).
        # agent_id -> gate -> count. These are CONDITIONS, not a partition: a press
        # taken out of FOV and beyond range increments both, so each subtotal is
        # BOUNDED BY episode_vetoed_missile_wasted rather than summing to it.
        self.episode_vetoed_missile_wasted_by_category: dict[str, dict[str, int]] = {}
        # Of the launched (valid) missiles, how many were inside vs outside the
        # aero NEZ/DLZ envelope. Used to score shot quality independent of trigger
        # discipline (vetoed presses).
        self.episode_in_envelope_missile_shots: dict[str, int] = {}
        self.episode_out_of_envelope_missile_shots: dict[str, int] = {}
        self.episode_lock_ok_count: dict[
            str, int
        ] = {}  # agent_id -> count of steps with radar lock
        self.episode_fov_ok_count: dict[
            str, int
        ] = {}  # agent_id -> count of steps with target in FOV
        self.episode_steps_count: dict[str, int] = {}  # agent_id -> total steps for this agent
        # Steps on which a viable firing solution existed, counted whether or not the
        # trigger was pulled. Denominator for the shot-conversion rate.
        self.episode_shot_opportunity_count: dict[str, int] = {}
        self.episode_fire_attempt_on_opportunity_count: dict[str, int] = {}
        # Answers "why did the shots we took not kill anything?" from inside the
        # campaign: launch geometry at trigger pull, and terminal outcome per
        # missile. See missile_diagnostics for why detonation_rate is the headline.
        self.missile_diagnostics = MissileDiagnosticsCollector()
        # Per-step sensor-chain state, summed per agent. `lock_rate` is the END of a
        # chain (detections -> tentative -> confirmed -> engageable -> lock); on its
        # own a collapse to 0.00 cannot say which link broke. See sensor_snapshot.
        self.episode_sensor_sums: dict[str, dict[str, float]] = {}

        # First-event timing (sim seconds; absent = event never occurred this episode)
        self.episode_first_shot_time: dict[str, float] = {}  # agent_id -> sim_time_s of 1st shot
        self.episode_first_kill_time: dict[str, float] = {}  # agent_id -> sim_time_s of 1st kill

        # Death cause tracking
        self.boundary_violators: set[str] = set()  # agent_ids that died from boundary violation
        self.kill_credited_victims: set[str] = set()
        """Victim agent IDs that have already produced a credited kill this episode."""

        # Event processing
        self.last_processed_event_count: int = 0

    def reset(self):
        """Reset all tracking state for new episode."""
        self.previous_positions.clear()
        self.missiles_fired_last_step.clear()

        self.prev_sqi.clear()
        self.no_lock_timer.clear()
        self.nez_steps.clear()
        self.orbit_state.clear()
        self.time_with_lock.clear()
        self.had_lock_previous_step.clear()
        self.team_first_shot.clear()
        self.prev_tactical_potential.clear()
        self.prev_energy_advantage.clear()

        self.episode_missiles_fired.clear()
        self.episode_kills.clear()
        self.episode_missile_kills.clear()
        self.episode_deaths.clear()

        self.episode_valid_missile_shots.clear()
        self.episode_vetoed_missile_shots.clear()
        self.episode_vetoed_missile_suppressed.clear()
        # Was omitted, so no-target vetoes leaked across episodes within a run.
        self.episode_vetoed_missile_no_target.clear()
        self.episode_vetoed_missile_wasted.clear()
        self.episode_vetoed_missile_wasted_by_category.clear()
        self.episode_in_envelope_missile_shots.clear()
        self.episode_out_of_envelope_missile_shots.clear()
        self.episode_lock_ok_count.clear()
        self.episode_fov_ok_count.clear()
        self.episode_steps_count.clear()
        self.episode_shot_opportunity_count.clear()
        self.episode_fire_attempt_on_opportunity_count.clear()
        self.missile_diagnostics.reset()
        self.episode_sensor_sums.clear()
        self.episode_first_shot_time.clear()
        self.episode_first_kill_time.clear()

        self.boundary_violators.clear()
        self.kill_credited_victims.clear()
        self.last_processed_event_count = 0

    def initialize_agent(self, agent_id: str, position: tuple = None):
        """Initialize tracking for a single agent."""
        if position:
            self.previous_positions[agent_id] = position
        self.missiles_fired_last_step[agent_id] = 0

        self.episode_missiles_fired[agent_id] = 0
        self.episode_kills[agent_id] = 0
        self.episode_missile_kills[agent_id] = 0
        self.episode_deaths[agent_id] = 0

        self.episode_valid_missile_shots[agent_id] = 0
        self.episode_vetoed_missile_shots[agent_id] = 0
        self.episode_vetoed_missile_suppressed[agent_id] = 0
        self.episode_vetoed_missile_no_target[agent_id] = 0
        self.episode_vetoed_missile_wasted[agent_id] = 0
        self.episode_vetoed_missile_wasted_by_category[agent_id] = {
            category: 0 for category in WASTED_VETO_CATEGORIES
        }
        self.episode_in_envelope_missile_shots[agent_id] = 0
        self.episode_out_of_envelope_missile_shots[agent_id] = 0
        self.episode_lock_ok_count[agent_id] = 0
        self.episode_fov_ok_count[agent_id] = 0
        self.episode_steps_count[agent_id] = 0

    def update_position(self, agent_id: str, position: tuple):
        """Update agent position for next step."""
        self.previous_positions[agent_id] = position

    def record_kill(self, agent_id: str, sim_time_s: float | None = None):
        """Record a kill for an agent."""
        self.episode_kills[agent_id] = self.episode_kills.get(agent_id, 0) + 1
        if sim_time_s is not None and agent_id not in self.episode_first_kill_time:
            self.episode_first_kill_time[agent_id] = sim_time_s

    def record_missile_kill(self, agent_id: str, sim_time_s: float | None = None):
        """Record a missile-caused kill for an agent."""
        self.record_kill(agent_id, sim_time_s)
        self.episode_missile_kills[agent_id] = self.episode_missile_kills.get(agent_id, 0) + 1

    def record_death(self, agent_id: str):
        """Record a death for an agent."""
        self.episode_deaths[agent_id] = 1

    def record_boundary_violation(self, agent_id: str):
        """Record that an agent died from boundary violation."""
        self.boundary_violators.add(agent_id)

    def update_missiles_fired(self, agent_id: str, count: int, sim_time_s: float | None = None):
        """Update missiles fired this step."""
        self.missiles_fired_last_step[agent_id] = count
        self.episode_missiles_fired[agent_id] = self.episode_missiles_fired.get(agent_id, 0) + count
        if count > 0 and sim_time_s is not None and agent_id not in self.episode_first_shot_time:
            self.episode_first_shot_time[agent_id] = sim_time_s

    def update_diagnostic_metrics(
        self,
        agent_id: str,
        valid_shots: int,
        vetoed_shots: int,
        lock_ok: bool,
        fov_ok: bool,
        in_envelope_shots: int = 0,
        out_of_envelope_shots: int = 0,
        suppressed_shots: int = 0,
        wasted_shots: int = 0,
        no_target_shots: int = 0,
        wasted_shots_by_category: dict[str, int] | None = None,
        shot_opportunity: int = 0,
        fire_attempt_on_opportunity: int = 0,
        launch_geometry: dict | None = None,
        sensor_state: dict | None = None,
        target_designated: float = 0.0,
    ):
        """Update episode-level diagnostic metrics."""
        # Launch diagnostics hang off `valid_shots`, which is the action space's own
        # `valid_missile_fires` signal and is the ONLY counter here that reliably
        # tracks launches. They were previously recorded from
        # step_processor._update_missiles_fired, which derives its count from
        # `len(unit.missiles)` -- the missiles LOADED. That number DECREASES on
        # launch, and the counter is `max(0, cur - prev)`, so it is structurally
        # always zero on the firing step. record_launch was never called once, and
        # all nine diagnostic columns wrote NaN for two full campaigns (v19, v20)
        # while the agent was demonstrably firing.
        if valid_shots > 0:
            geometry = launch_geometry if isinstance(launch_geometry, dict) else None
            for _ in range(int(valid_shots)):
                self.missile_diagnostics.record_launch(geometry)
        bucket = self.episode_sensor_sums.setdefault(agent_id, {})
        bucket[TARGET_DESIGNATED_RATE] = bucket.get(TARGET_DESIGNATED_RATE, 0.0) + float(
            target_designated
        )
        if sensor_state:
            for key, value in sensor_state.items():
                bucket[key] = bucket.get(key, 0.0) + float(value)
        self.episode_valid_missile_shots[agent_id] = (
            self.episode_valid_missile_shots.get(agent_id, 0) + valid_shots
        )
        self.episode_vetoed_missile_shots[agent_id] = (
            self.episode_vetoed_missile_shots.get(agent_id, 0) + vetoed_shots
        )
        self.episode_vetoed_missile_suppressed[agent_id] = (
            self.episode_vetoed_missile_suppressed.get(agent_id, 0) + suppressed_shots
        )
        self.episode_vetoed_missile_wasted[agent_id] = (
            self.episode_vetoed_missile_wasted.get(agent_id, 0) + wasted_shots
        )
        self.episode_vetoed_missile_no_target[agent_id] = (
            self.episode_vetoed_missile_no_target.get(agent_id, 0) + no_target_shots
        )
        if wasted_shots_by_category:
            per_category = self.episode_vetoed_missile_wasted_by_category.setdefault(
                agent_id, {category: 0 for category in WASTED_VETO_CATEGORIES}
            )
            for category, count in wasted_shots_by_category.items():
                per_category[category] = per_category.get(category, 0) + int(count)
        self.episode_in_envelope_missile_shots[agent_id] = (
            self.episode_in_envelope_missile_shots.get(agent_id, 0) + in_envelope_shots
        )
        self.episode_out_of_envelope_missile_shots[agent_id] = (
            self.episode_out_of_envelope_missile_shots.get(agent_id, 0) + out_of_envelope_shots
        )
        self.episode_lock_ok_count[agent_id] = self.episode_lock_ok_count.get(agent_id, 0) + (
            1 if lock_ok else 0
        )
        self.episode_fov_ok_count[agent_id] = self.episode_fov_ok_count.get(agent_id, 0) + (
            1 if fov_ok else 0
        )
        self.episode_steps_count[agent_id] = self.episode_steps_count.get(agent_id, 0) + 1
        self.episode_shot_opportunity_count[agent_id] = self.episode_shot_opportunity_count.get(
            agent_id, 0
        ) + int(shot_opportunity)
        self.episode_fire_attempt_on_opportunity_count[agent_id] = (
            self.episode_fire_attempt_on_opportunity_count.get(agent_id, 0)
            + int(fire_attempt_on_opportunity)
        )
