"""
Configuration parser and validator for BVR Multi-Agent Environment.
"""

from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field

from simulator.utils.map_limits import MapLimits


@dataclass
class BVREnvConfig:
    """Configuration for BVR Multi-Agent Environment."""

    # Agent configuration
    num_agents_per_team: int
    agent_ids: List[str] = field(default_factory=list)
    opponent_ids: List[str] = field(default_factory=list)
    all_agent_ids: List[str] = field(default_factory=list)
    possible_agents: List[str] = field(default_factory=list)
    aircraft_type_map: Dict[str, str] = field(default_factory=dict)

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
    force_locks_all_enemies: bool = False

    # Map configuration
    map_size_km: float = 1000.0
    max_alt: float = 25000.0
    map_limits: MapLimits = None

    # Observation slots
    num_fm: int = 4
    num_ff: int = 2
    num_em: int = 4
    num_ef: int = 2
    num_pr: int = 2
    num_warn: int = 4
    own_state_dim: int = 22  # Updated: reduced from 26 (removed DLZ range redundancy, added Ps)

    # Tacview logging
    tacview_logfile: str = None

    # Neural wrapper config
    enable_missile_automation: bool = False
    missile_auto_sqi_threshold: float = 0.3
    missile_auto_max_per_target: int = 2
    missile_auto_long_cooldown_s: float = 10.0

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

        # Map limits
        map_size_km = float(config.get("map_size", 1000.0))
        half = (map_size_km / 2.0) / 111.0
        map_limits = MapLimits(
            left_lon=-half, bottom_lat=-half, right_lon=half, top_lat=half,
            min_alt=0.0, max_alt=float(config.get("max_alt", 25000.0))
        )

        # Simulation config
        sim_config = config.get("simulation_config", {})
        early_term = sim_config.get("early_termination", {})

        # Neural wrapper config
        neural_wrapper_config = config.get('neural_wrapper_config', {})

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
            force_locks_all_enemies=sim_config.get("force_locks_all_enemies", False),

            map_size_km=map_size_km,
            max_alt=float(config.get("max_alt", 25000.0)),
            map_limits=map_limits,

            num_fm=int(config.get("num_fm", 4)),
            num_ff=int(config.get("num_ff", 2)),
            num_em=int(config.get("num_em", 4)),
            num_ef=int(config.get("num_enemy", 2)),
            num_pr=int(config.get("pr_slots", int(config.get("num_enemy", 2)))),
            num_warn=int(config.get("warn_sectors", 4)),
            own_state_dim=22,  # Updated

            tacview_logfile=config.get("tacview_logfile", None),

            enable_missile_automation=neural_wrapper_config.get('enable_missile_automation', False),
            missile_auto_sqi_threshold=neural_wrapper_config.get('missile_auto_sqi_threshold', 0.3),
            missile_auto_max_per_target=neural_wrapper_config.get('missile_auto_max_per_target', 2),
            missile_auto_long_cooldown_s=neural_wrapper_config.get('missile_auto_long_cooldown_s', 10.0),

            debug=bool(config.get("debug", False)),
        )
