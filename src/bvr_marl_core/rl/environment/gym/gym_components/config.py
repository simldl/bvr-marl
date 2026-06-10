"""
Configuration parser and validator for BVR Multi-Agent Environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from bvr_marl_core.simulator import MapLimits


@dataclass
class AWACSConfigData:
    """Configuration for AWACS aircraft in scenarios."""

    # Whether to spawn AWACS for each team
    agent_awacs: bool = False
    opponent_awacs: bool = False

    # AWACS positioning (kept for backward compatibility)
    orbit_distance_km: float = 80.0
    orbit_altitude_m: float = 10000.0
    orbit_speed_mps: float = 200.0

    # Orbit pattern configuration
    orbit_pattern: str = "figure8"  # "racetrack", "figure8", "circle", "random", "random_all"
    orbit_leg_length_km: float = 40.0  # racetrack straight-leg length
    orbit_radius_km: float = 25.0  # figure-8 circle radius (also sets side-zone width)
    orbit_clockwise: bool = True

    # Fighter trailing: keep the AWACS a safe distance behind its own fighters
    # (away from the enemy) instead of orbiting a fixed side-zone point. The orbit
    # stays clamped inside the visible map so the AWACS remains on screen.
    trail_fighters: bool = True
    trail_standoff_km: float = 40.0

    # Lock FOV constraint (detection is still 360°, but locking requires facing target)
    lock_fov_deg: float = 60.0

    # When True, AWACS cannot be targeted by fighters and is shown with a different obs flag
    awacs_non_engageable: bool = True


@dataclass
class ScenarioConfigData:
    """Configuration for scenario selection and variation."""

    # list of geometry regimes to randomly sample from each episode
    # If empty, uses the single regime from geometry_config
    allowed_regimes: list[str] = field(default_factory=list)

    # Whether to randomize regime selection each episode
    randomize_regime: bool = True

    # AWACS configuration
    awacs_config: AWACSConfigData = field(default_factory=AWACSConfigData)


@dataclass
class GeometryConfigData:
    """Configuration for controlled scenario geometry."""

    use_controlled_geometry: bool = False
    regime: str = "hot_120km"
    formation_spread_m: float = 2000.0
    position_noise_m: float = 0.0
    altitude_noise_m: float = 0.0
    heading_noise_deg: float = 0.0


@dataclass
class MetricsConfigData:
    """Configuration for episode metrics collection."""

    enable_metrics: bool = False
    output_dir: str = "study_results"
    export_interval: int = 100


@dataclass
class BVREnvConfig:
    """Configuration for BVR Multi-Agent Environment."""

    # Agent configuration
    num_agents_per_team: int
    agent_ids: list[str] = field(default_factory=list)
    opponent_ids: list[str] = field(default_factory=list)
    all_agent_ids: list[str] = field(default_factory=list)
    possible_agents: list[str] = field(default_factory=list)
    aircraft_type_map: dict[str, str] = field(default_factory=dict)

    # Episode/time configuration
    max_steps: int = 600
    tick_secs: float = 1.0
    max_real_time_s: float = None

    # Termination configuration
    early_term_all_enemies_dead: bool = True
    early_term_mission_complete: bool = True
    early_term_no_missiles: bool = False

    # Training aids
    fix_radar_lock_after_first_obs: bool = False
    lock_fix_duration_s: float = 0.0
    force_locks_all_enemies: bool = False

    # Map configuration
    map_size_km: float = 1000.0
    map_width_km: float = 1000.0
    map_height_km: float = 1000.0
    max_alt: float = 25000.0
    map_limits: MapLimits = None  # Combat zone — agents destroyed if they leave this
    full_map_limits: MapLimits = None  # Full display area including AWACS side zones

    # Observation slots
    num_fm: int = 4
    num_ff: int = 2
    num_em: int = 4
    num_ef: int = 2
    num_pr: int = 2
    num_warn: int = 4
    own_state_dim: int = 21  # 22 minus SQI; SQI is evaluation-only

    # Tacview logging
    tacview_logfile: str = None

    # Neural wrapper config
    enable_missile_automation: bool = False
    missile_auto_sqi_threshold: float = 0.3
    missile_auto_max_per_target: int = 2
    missile_auto_long_cooldown_s: float = 10.0
    missile_fire_threshold: float = 0.5

    # Weapon system toggles
    enable_gun: bool = False

    # Datalink configuration
    datalink_mode: str = "full"

    # Geometry configuration for controlled scenarios
    geometry_config: GeometryConfigData = field(default_factory=GeometryConfigData)

    # Scenario configuration (AWACS, regime randomization)
    scenario_config: ScenarioConfigData = field(default_factory=ScenarioConfigData)

    # Metrics configuration for study
    metrics_config: MetricsConfigData = field(default_factory=MetricsConfigData)

    # Debugging
    debug: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> BVREnvConfig:
        """Create configuration from dictionary."""
        # Accept both keys for num_agents
        n = int(config.get("num_agents_per_team", config.get("num_agents_per_side", 2)))

        agent_ids = [f"A{i}" for i in range(n)]
        opponent_ids = [f"B{i}" for i in range(n)]
        all_agent_ids = agent_ids + opponent_ids

        # Aircraft class mapping
        grp_map = config.get("aircraft_types", {})
        aircraft_type_map = {
            aid: (grp_map.get("agent", None) if aid in agent_ids else grp_map.get("opponent", None))
            for aid in all_agent_ids
        }

        # Combat zone map limits (agents are destroyed if they cross these)
        map_size_km = float(config.get("map_size", 1000.0))
        map_width_km = float(config.get("map_width_km", config.get("map_width", map_size_km)))
        map_height_km = float(config.get("map_height_km", config.get("map_height", map_size_km)))
        half_width = (map_width_km / 2.0) / 111.0
        half_height = (map_height_km / 2.0) / 111.0
        max_alt_val = float(config.get("max_alt", 25000.0))
        map_limits = MapLimits(
            left_lon=-half_width,
            bottom_lat=-half_height,
            right_lon=half_width,
            top_lat=half_height,
            min_alt=0.0,
            max_alt=max_alt_val,
        )

        # Full display map limits: expand only when AWACS geometry actually
        # extends beyond the combat zone. If AWACS remain inside the combat map,
        # full/combat bounds stay identical so a 400 x 200 km scenario really
        # renders as 400 x 200 km.
        awacs_cfg_pre = config.get("scenario_config", {}).get("awacs_config", {})
        awacs_enabled = bool(
            awacs_cfg_pre.get("agent_awacs", False) or awacs_cfg_pre.get("opponent_awacs", False)
        )
        if awacs_enabled:
            awacs_radius_km = float(awacs_cfg_pre.get("orbit_radius_km", 25.0))
            awacs_distance_km = float(awacs_cfg_pre.get("orbit_distance_km", 80.0))
            awacs_required_half_width_km = max(
                map_width_km / 2.0,
                awacs_distance_km + awacs_radius_km,
            )
        else:
            awacs_required_half_width_km = map_width_km / 2.0
        half_ew = awacs_required_half_width_km / 111.0
        full_map_limits = MapLimits(
            left_lon=-half_ew,
            bottom_lat=-half_height,
            right_lon=half_ew,
            top_lat=half_height,
            min_alt=0.0,
            max_alt=max_alt_val,
        )

        # Simulation config
        sim_config = config.get("simulation_config", {})
        early_term = sim_config.get("early_termination", {})

        # Neural wrapper config
        neural_wrapper_config = config.get("neural_wrapper_config", {})

        # Weapon config
        weapon_config = config.get("weapon_config", {})

        # Geometry config
        geometry_cfg = config.get("geometry_config", {})
        geometry_config = GeometryConfigData(
            use_controlled_geometry=geometry_cfg.get("use_controlled_geometry", False),
            regime=geometry_cfg.get("regime", "hot_120km"),
            formation_spread_m=float(geometry_cfg.get("formation_spread_m", 2000.0)),
            position_noise_m=float(geometry_cfg.get("position_noise_m", 0.0)),
            altitude_noise_m=float(geometry_cfg.get("altitude_noise_m", 0.0)),
            heading_noise_deg=float(geometry_cfg.get("heading_noise_deg", 0.0)),
        )

        # Metrics config
        metrics_cfg = config.get("metrics_config", {})
        metrics_config = MetricsConfigData(
            enable_metrics=metrics_cfg.get("enable_metrics", False),
            output_dir=metrics_cfg.get("output_dir", "study_results"),
            export_interval=int(metrics_cfg.get("export_interval", 100)),
        )

        # Scenario config (AWACS and regime randomization)
        scenario_cfg = config.get("scenario_config", {})
        awacs_cfg = scenario_cfg.get("awacs_config", {})
        awacs_non_engageable = awacs_cfg.get("awacs_non_engageable", True)
        awacs_config = AWACSConfigData(
            agent_awacs=awacs_cfg.get("agent_awacs", False),
            opponent_awacs=awacs_cfg.get("opponent_awacs", False),
            orbit_distance_km=float(awacs_cfg.get("orbit_distance_km", 80.0)),
            orbit_altitude_m=float(awacs_cfg.get("orbit_altitude_m", 10000.0)),
            orbit_speed_mps=float(awacs_cfg.get("orbit_speed_mps", 200.0)),
            orbit_pattern=awacs_cfg.get("orbit_pattern", "figure8"),
            orbit_leg_length_km=float(awacs_cfg.get("orbit_leg_length_km", 40.0)),
            orbit_radius_km=float(awacs_cfg.get("orbit_radius_km", 25.0)),
            orbit_clockwise=awacs_cfg.get("orbit_clockwise", True),
            trail_fighters=bool(awacs_cfg.get("trail_fighters", True)),
            trail_standoff_km=float(awacs_cfg.get("trail_standoff_km", 40.0)),
            lock_fov_deg=float(awacs_cfg.get("lock_fov_deg", 60.0)),
            awacs_non_engageable=bool(awacs_non_engageable),
        )
        scenario_config = ScenarioConfigData(
            allowed_regimes=scenario_cfg.get("allowed_regimes", []),
            randomize_regime=scenario_cfg.get("randomize_regime", True),
            awacs_config=awacs_config,
        )

        return cls(
            num_agents_per_team=n,
            agent_ids=agent_ids,
            opponent_ids=opponent_ids,
            all_agent_ids=all_agent_ids,
            possible_agents=list(all_agent_ids),
            aircraft_type_map=aircraft_type_map,
            max_steps=int(config.get("max_steps", 600)),
            tick_secs=float(sim_config.get("tick_secs", 1.0)),
            max_real_time_s=sim_config.get("max_real_time_s", None),
            early_term_all_enemies_dead=early_term.get("all_enemies_dead", True),
            early_term_mission_complete=early_term.get("mission_complete", True),
            early_term_no_missiles=early_term.get("no_missiles_remaining", False),
            fix_radar_lock_after_first_obs=sim_config.get("fix_radar_lock_after_first_obs", False),
            lock_fix_duration_s=float(sim_config.get("lock_fix_duration_s", 0.0)),
            force_locks_all_enemies=sim_config.get("force_locks_all_enemies", False),
            map_size_km=map_width_km,
            map_width_km=map_width_km,
            map_height_km=map_height_km,
            max_alt=max_alt_val,
            map_limits=map_limits,
            full_map_limits=full_map_limits,
            num_fm=int(config.get("num_fm", 4)),
            num_ff=int(config.get("num_ff", 2)),
            num_em=int(config.get("num_em", 4)),
            # Auto-bump num_ef by 1 when opponent AWACS is present and non-engageable
            # so there is always a dedicated slot in the enemy observation for the AWACS.
            num_ef=int(config.get("num_enemy", 2))
            + (1 if awacs_cfg.get("opponent_awacs", False) and bool(awacs_non_engageable) else 0),
            num_pr=int(config.get("pr_slots", int(config.get("num_enemy", 2)))),
            num_warn=int(config.get("warn_sectors", 4)),
            own_state_dim=21,
            tacview_logfile=config.get("tacview_logfile", None),
            enable_missile_automation=neural_wrapper_config.get("enable_missile_automation", False),
            missile_auto_sqi_threshold=neural_wrapper_config.get("missile_auto_sqi_threshold", 0.3),
            missile_auto_max_per_target=neural_wrapper_config.get("missile_auto_max_per_target", 2),
            missile_auto_long_cooldown_s=neural_wrapper_config.get(
                "missile_auto_long_cooldown_s", 10.0
            ),
            missile_fire_threshold=float(neural_wrapper_config.get("missile_fire_threshold", 0.5)),
            enable_gun=bool(weapon_config.get("enable_gun", False)),
            datalink_mode=config.get("datalink_mode", "full"),
            geometry_config=geometry_config,
            scenario_config=scenario_config,
            metrics_config=metrics_config,
            debug=bool(config.get("debug", False)),
        )
