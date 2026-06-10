"""Environment wrapper for tracking detailed reward breakdowns."""


class RewardTrackingEnvWrapper:
    """
    Wrapper for the environment that tracks detailed reward breakdowns.
    """

    def __init__(self, env):
        self.env = env
        self.reward_history = []
        self.current_timestep = 0

    def reset(self, **kwargs):
        """Reset the environment and clear reward history."""
        self.reward_history = []
        self.current_timestep = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        """Step the environment and track reward breakdowns."""
        # Perform the step
        observations, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Collect detailed reward breakdowns
        step_data = {
            "timestep": self.current_timestep,
            "sim_time": self.env.simulator.elapsed_time_s,
        }

        for agent_id in self.env.all_agent_ids:
            unit_id = self.env.agent_to_unit_id.get(agent_id)
            unit = self.env.simulator.active_units.get(unit_id) if unit_id else None

            # Get reward info from the step
            agent_reward = rewards.get(agent_id, 0.0)

            # Try to get detailed breakdown if unit still exists
            breakdown = {}
            if unit is not None:
                try:
                    # Reconstruct the reward context similar to how the environment does it
                    enemies = [
                        u
                        for u in self.env.simulator.active_units.values()
                        if hasattr(u, "group") and u.group != unit.group and hasattr(u, "position")
                    ]

                    targets = (
                        getattr(unit, "locked_targets", [])
                        if hasattr(unit, "locked_targets")
                        else []
                    )

                    incoming_missiles = [
                        m
                        for m in self.env.simulator.active_units.values()
                        if hasattr(m, "target") and m.target == unit
                    ]

                    # Get previous state info from env.state_tracker
                    previous_position = self.env.state_tracker.previous_positions.get(agent_id)

                    missiles_fired = self.env.state_tracker.missiles_fired_last_step.get(
                        agent_id, 0
                    )

                    # Extract previous altitude from previous_position tuple
                    prev_alt = previous_position[2] if previous_position else None

                    action_state = self.env.action_processor.agent_states.get(unit_id, {})

                    # Get Phase 2 tracking state from environment state_tracker
                    prev_sqi = self.env.state_tracker.prev_sqi.get(agent_id, 0.0)
                    nez_steps = self.env.state_tracker.nez_steps.get(agent_id, 0)
                    no_lock_duration_s = self.env.state_tracker.no_lock_timer.get(agent_id, 0.0)
                    orbit_state = self.env.state_tracker.orbit_state.get(agent_id, {})
                    tick_secs = self.env.tick_secs
                    time_with_lock_s = self.env.state_tracker.time_with_lock.get(agent_id, 0.0)

                    # Get detailed breakdown (matching env's call pattern with Phase 2 params)
                    breakdown = self.env.reward_calculator.get_reward_breakdown(
                        unit=unit,
                        map_limits=self.env.map_limits,
                        enemy_kill_count=0,  # Not tracked per-step in env
                        destroyed=False,
                        is_last_team=False,  # Env doesn't track this during normal steps
                        enemies=enemies,
                        targets=targets,
                        incoming_missiles=incoming_missiles,
                        previous_position=previous_position,
                        missiles_fired_this_step=missiles_fired,
                        previous_altitude=prev_alt,
                        action_state=action_state,
                        # Phase 2 tracking state
                        prev_sqi=prev_sqi,
                        nez_steps=nez_steps,
                        no_lock_duration_s=no_lock_duration_s,
                        orbit_state=orbit_state,
                        tick_secs=tick_secs,
                        time_with_lock_s=time_with_lock_s,
                    )
                except Exception as e:
                    # If breakdown calculation fails, just use total reward
                    print(f"Warning: Could not compute reward breakdown for {agent_id}: {e}")
                    breakdown = {"total": agent_reward}
            elif agent_id in rewards:
                # Agent was destroyed this step
                breakdown = {"destruction": rewards[agent_id]}

            step_data[agent_id] = {
                "total_reward": agent_reward,
                "breakdown": breakdown,
                "alive": unit is not None,
            }

        self.reward_history.append(step_data)
        self.current_timestep += 1

        return observations, rewards, terminateds, truncateds, infos

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self.env, name)
