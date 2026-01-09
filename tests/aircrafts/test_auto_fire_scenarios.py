"""
Test automated missile firing across multiple geometry scenarios.

Tests 5 different engagement scenarios to understand SQI threshold behavior:
1. Head-on, medium range (22km) - OPTIMAL
2. Beam aspect, medium range (22km) - GOOD
3. Tail chase, close range (15km) - MARGINAL
4. Head-on, long range (40km) - MARGINAL
5. Beam aspect, long range (40km) - POOR
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
from datetime import datetime
from simulator.simulator import Simulator
from simulator.core.units import Position
from aircrafts.types.eurofighter import Eurofighter
from missiles.fox3.amraam import AIM120_AMRAAM

class MapLimits:
    def __init__(self):
        self.left_lon = -1.0
        self.right_lon = 1.0
        self.bottom_lat = -1.0
        self.top_lat = 1.0
        self.min_alt = 0.0
        self.max_alt = 20000.0

def run_scenario(scenario_name, agent_config, enemy_config, threshold=0.3, max_steps=30):
    """
    Run a single test scenario.

    Args:
        scenario_name: Name of the scenario
        agent_config: (lat, lon, alt, heading, speed)
        enemy_config: (lat, lon, alt, heading, speed)
        threshold: SQI threshold for auto-fire
        max_steps: Maximum simulation steps

    Returns:
        (missiles_fired, initial_sqi, final_sqi, fired_at_steps)
    """
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)

    # Create simulator
    weapon_config = {
        'missile_hit_radius_m': 500.0,
        'gun_hit_radius_m': 5.0,
    }
    simulator = Simulator(
        utc_time=datetime.now(),
        tick_secs=1,
        random_seed=42,
        weapon_config=weapon_config
    )

    map_limits = MapLimits()

    # Unpack configs
    agent_lat, agent_lon, agent_alt, agent_hdg, agent_spd = agent_config
    enemy_lat, enemy_lon, enemy_alt, enemy_hdg, enemy_spd = enemy_config

    # Create agent aircraft
    agent_pos = Position(lat=agent_lat, lon=agent_lon, alt=agent_alt)
    agent = Eurofighter(
        agent_pos, agent_hdg, agent_spd, "BLUE", map_limits, 0.0, 20000.0
    )
    agent.missile_types = [AIM120_AMRAAM]
    agent_id = simulator.add_unit(agent)

    # Create enemy aircraft
    enemy_pos = Position(lat=enemy_lat, lon=enemy_lon, alt=enemy_alt)
    enemy = Eurofighter(
        enemy_pos, enemy_hdg, enemy_spd, "RED", map_limits, 0.0, 20000.0
    )
    enemy.missile_types = [AIM120_AMRAAM]
    enemy_id = simulator.add_unit(enemy)

    # Calculate initial range
    from simulator.utils.geodesics import geodetic_distance_km
    initial_range_km = geodetic_distance_km(
        agent_lat, agent_lon, agent_alt,
        enemy_lat, enemy_lon, enemy_alt
    )

    print(f"\n[SETUP]")
    print(f"  Agent: {agent_lat:.2f}°N, {agent_lon:.2f}°E, {agent_alt:.0f}m, hdg={agent_hdg:.0f}°, spd={agent_spd:.0f}m/s")
    print(f"  Enemy: {enemy_lat:.2f}°N, {enemy_lon:.2f}°E, {enemy_alt:.0f}m, hdg={enemy_hdg:.0f}°, spd={enemy_spd:.0f}m/s")
    print(f"  Initial range: {initial_range_km:.1f} km")

    # Configure auto-fire
    from reinforcement_learning.environment.spaces.action_space import ActionProcessor
    action_processor = ActionProcessor(simulator)
    action_processor.configure_automation(
        enable_missile_automation=True,
        missile_auto_sqi_threshold=threshold,
        missile_auto_max_per_target=2,
        missile_auto_long_cooldown_s=10.0
    )

    # Get initial SQI
    initial_sqi = None
    if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'get_sqi'):
        try:
            sqi_result = agent.metrics.get_sqi(enemy)
            if sqi_result.get('valid', False):
                initial_sqi = sqi_result.get('sqi', 0.0)
                print(f"  Initial SQI: {initial_sqi:.3f}")
            else:
                error = sqi_result.get('error', 'unknown')
                print(f"  Initial SQI invalid: {error}")
        except Exception as e:
            print(f"  Initial SQI check failed: {e}")

    # Run simulation
    missiles_fired = 0
    fired_at_steps = []
    action = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for step in range(1, max_steps + 1):
        # Apply action
        action_processor.apply(agent_id, action.copy())

        # Check for missile fires
        signals = action_processor.get_training_signals(agent_id)
        if signals['valid_missile_fires'] > 0:
            missiles_fired += signals['valid_missile_fires']
            fired_at_steps.append(step)

            # Get SQI at fire
            if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'get_sqi'):
                try:
                    sqi_result = agent.metrics.get_sqi(enemy)
                    if sqi_result.get('valid', False):
                        sqi_at_fire = sqi_result.get('sqi', 0.0)
                        print(f"  [STEP {step}] FIRED! SQI={sqi_at_fire:.3f}, total={missiles_fired}")
                except:
                    pass

        # Advance simulation
        simulator.do_tick()

        # Stop after max missiles fired
        if missiles_fired >= 2:
            break

    # Get final SQI
    final_sqi = None
    if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'get_sqi'):
        try:
            sqi_result = agent.metrics.get_sqi(enemy)
            if sqi_result.get('valid', False):
                final_sqi = sqi_result.get('sqi', 0.0)
        except:
            pass

    return missiles_fired, initial_sqi, final_sqi, fired_at_steps


def test_all_scenarios():
    """Test auto-fire across multiple engagement scenarios."""

    print("="*80)
    print("AUTO-FIRE GEOMETRY TEST - MULTIPLE SCENARIOS")
    print("="*80)
    print("\nTesting SQI threshold: 0.3")
    print("Max missiles per target: 2")
    print("Testing 5 different engagement geometries...")

    # Define scenarios
    # Format: (name, agent_config, enemy_config)
    # Config: (lat, lon, alt, heading_deg, speed_mps)

    scenarios = [
        (
            "1. HEAD-ON, MEDIUM RANGE (22km) - OPTIMAL",
            (0.0, 0.0, 10000.0, 90.0, 250.0),   # Agent: heading east
            (0.0, 0.2, 10000.0, 270.0, 250.0),  # Enemy: heading west, 22km away
        ),
        (
            "2. BEAM ASPECT, MEDIUM RANGE (22km) - GOOD",
            (0.0, 0.0, 10000.0, 90.0, 250.0),   # Agent: heading east
            (0.2, 0.0, 10000.0, 180.0, 250.0),  # Enemy: heading south (90° aspect), 22km away
        ),
        (
            "3. TAIL CHASE, CLOSE RANGE (15km) - MARGINAL",
            (0.0, 0.0, 10000.0, 90.0, 250.0),   # Agent: heading east
            (0.0, 0.135, 10000.0, 90.0, 200.0), # Enemy: heading east (same dir), 15km away
        ),
        (
            "4. HEAD-ON, LONG RANGE (40km) - MARGINAL",
            (0.0, 0.0, 10000.0, 90.0, 250.0),   # Agent: heading east
            (0.0, 0.36, 10000.0, 270.0, 250.0), # Enemy: heading west, 40km away
        ),
        (
            "5. BEAM ASPECT, LONG RANGE (40km) - POOR",
            (0.0, 0.0, 10000.0, 90.0, 250.0),   # Agent: heading east
            (0.36, 0.0, 10000.0, 180.0, 250.0), # Enemy: heading south, 40km away
        ),
    ]

    results = []

    for scenario_name, agent_config, enemy_config in scenarios:
        missiles_fired, initial_sqi, final_sqi, fired_at_steps = run_scenario(
            scenario_name, agent_config, enemy_config, threshold=0.3, max_steps=30
        )
        results.append({
            'name': scenario_name,
            'missiles_fired': missiles_fired,
            'initial_sqi': initial_sqi,
            'final_sqi': final_sqi,
            'fired_at_steps': fired_at_steps
        })

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - AUTO-FIRE BEHAVIOR ACROSS SCENARIOS")
    print("="*80)
    print(f"\n{'Scenario':<50} {'Initial SQI':<15} {'Missiles':<10} {'Fired Steps'}")
    print("-"*80)

    for r in results:
        name = r['name'][:48]
        sqi_str = f"{r['initial_sqi']:.3f}" if r['initial_sqi'] is not None else "N/A"
        missiles_str = f"{r['missiles_fired']}/2"
        steps_str = str(r['fired_at_steps']) if r['fired_at_steps'] else "None"

        # Add indicator
        if r['missiles_fired'] == 2:
            indicator = "✓"
        elif r['missiles_fired'] > 0:
            indicator = "~"
        else:
            indicator = "✗"

        print(f"{indicator} {name:<48} {sqi_str:<15} {missiles_str:<10} {steps_str}")

    print("\n" + "="*80)
    print("LEGEND")
    print("="*80)
    print("✓ = Fired max missiles (2) - Auto-fire triggered")
    print("~ = Fired some missiles (1) - Partial trigger")
    print("✗ = No missiles fired - SQI below threshold (0.3)")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Analyze SQI ranges
    sqi_values = [r['initial_sqi'] for r in results if r['initial_sqi'] is not None]
    fired_sqi = [r['initial_sqi'] for r in results if r['initial_sqi'] is not None and r['missiles_fired'] > 0]
    not_fired_sqi = [r['initial_sqi'] for r in results if r['initial_sqi'] is not None and r['missiles_fired'] == 0]

    if sqi_values:
        print(f"\nSQI range observed: {min(sqi_values):.3f} - {max(sqi_values):.3f}")

    if fired_sqi:
        print(f"SQI when firing occurred: {min(fired_sqi):.3f} - {max(fired_sqi):.3f}")

    if not_fired_sqi:
        print(f"SQI when NOT firing: {min(not_fired_sqi):.3f} - {max(not_fired_sqi):.3f}")

    # Check if threshold is appropriate
    print(f"\nThreshold setting: 0.3")
    scenarios_fired = sum(1 for r in results if r['missiles_fired'] > 0)
    print(f"Scenarios with auto-fire: {scenarios_fired}/{len(results)}")

    if scenarios_fired >= 3:
        print("\n✓ Threshold 0.3 appears GOOD - triggers in favorable geometries")
    elif scenarios_fired >= 2:
        print("\n~ Threshold 0.3 appears REASONABLE - selective firing")
    else:
        print("\n✗ Threshold 0.3 may be TOO HIGH - very conservative firing")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if fired_sqi and not_fired_sqi:
        boundary = (max(not_fired_sqi) + min(fired_sqi)) / 2
        print(f"\nObserved firing boundary: ~{boundary:.3f}")

        if boundary < 0.2:
            print("Consider LOWERING threshold to 0.2 for more aggressive engagement")
        elif boundary > 0.4:
            print("Consider RAISING threshold to 0.4-0.5 for more selective engagement")
        else:
            print("Current threshold 0.3 aligns well with natural SQI boundary")

    print("\nFor Phase 1 training (learning positioning):")
    print("  - Current 0.3 threshold should work well")
    print("  - Agents will fire in favorable geometries")
    print("  - 2-missile limit prevents excessive waste")

    print("\nFor Phase 2+ (manual shooting):")
    print("  - Consider 0.4-0.5 threshold to encourage better positioning")
    print("  - Or disable automation entirely")

    return results


if __name__ == "__main__":
    results = test_all_scenarios()
    sys.exit(0)
