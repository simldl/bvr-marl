"""
Controlled test for automated missile firing with forced good geometry.

This test:
1. Spawns agents in favorable engagement geometry (head-on, medium range)
2. Verifies SQI is calculated correctly
3. Confirms auto-fire triggers when SQI > threshold
4. Tests max missiles per target limit
5. Verifies long cooldown after limit reached
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

def test_auto_fire_controlled():
    """Test automated missile firing with controlled geometry."""

    print("="*80)
    print("CONTROLLED AUTO-FIRE TEST - FAVORABLE GEOMETRY")
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

    print("\n[SETUP] Creating agents in favorable geometry...")
    print("  Agent: 0°N, 0°E, 10000m, heading EAST (90°)")
    print("  Enemy: 0°N, 0.2°E (~22km), 10000m, heading WEST (270°)")
    print("  Configuration: HEAD-ON, medium range, co-altitude")

    # Map limits class
    class MapLimits:
        def __init__(self):
            self.left_lon = -1.0
            self.right_lon = 1.0
            self.bottom_lat = -1.0
            self.top_lat = 1.0
            self.min_alt = 0.0
            self.max_alt = 20000.0

    map_limits = MapLimits()

    # Create agent aircraft (facing east)
    agent_pos = Position(lat=0.0, lon=0.0, alt=10000.0)
    agent = Eurofighter(
        agent_pos,          # position
        90.0,               # yaw_deg (heading east)
        250.0,              # speed_mps
        "BLUE",             # group
        map_limits,         # map_limits
        0.0,                # min_alt_m
        20000.0             # max_alt_m
    )
    agent.missile_types = [AIM120_AMRAAM]
    agent_id = simulator.add_unit(agent)

    # Create enemy aircraft (facing west, ~22km away)
    enemy_pos = Position(lat=0.0, lon=0.2, alt=10000.0)  # ~22km east
    enemy = Eurofighter(
        enemy_pos,          # position
        270.0,              # yaw_deg (heading west toward agent)
        250.0,              # speed_mps
        "RED",              # group
        map_limits,         # map_limits
        0.0,                # min_alt_m
        20000.0             # max_alt_m
    )
    enemy.missile_types = [AIM120_AMRAAM]
    enemy_id = simulator.add_unit(enemy)

    print(f"\n[OK] Agent ID: {agent_id}, Enemy ID: {enemy_id}")

    # Configure auto-fire on agent
    from reinforcement_learning.environment.spaces.action_space import ActionProcessor
    action_processor = ActionProcessor(simulator)
    action_processor.configure_automation(
        enable_missile_automation=True,
        missile_auto_sqi_threshold=0.3,
        missile_auto_max_per_target=2,
        missile_auto_long_cooldown_s=10.0
    )

    print("\n[OK] Auto-fire configured:")
    print(f"  Enabled: True")
    print(f"  SQI threshold: 0.3")
    print(f"  Max missiles per target: 2")
    print(f"  Long cooldown: 10.0s")

    # Test scenario
    print("\n" + "="*80)
    print("RUNNING TEST - STEP BY STEP")
    print("="*80)

    missiles_fired = 0
    step_count = 0
    max_steps = 30

    # Action: neutral maneuver (Ps=0.5, n=0.5, φ=0.5)
    # action[4] = 0.0 initially (no manual fire), auto-fire will override
    action = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("\n[STEP 0] Initial positions set")

    # Check initial SQI
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
    else:
        print("  [WARNING] SQI calculation not available")

    while step_count < max_steps:
        step_count += 1

        # Apply action to agent
        action_processor.apply(agent_id, action.copy())

        # Check training signals
        signals = action_processor.get_training_signals(agent_id)

        if signals['valid_missile_fires'] > 0:
            missiles_fired += signals['valid_missile_fires']
            print(f"\n[STEP {step_count}] ✓ MISSILE FIRED!")
            print(f"  Total missiles fired: {missiles_fired}")

            # Get SQI at time of fire
            if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'get_sqi'):
                try:
                    sqi_result = agent.metrics.get_sqi(enemy)
                    if sqi_result.get('valid', False):
                        sqi_at_fire = sqi_result.get('sqi', 0.0)
                        print(f"  SQI at fire: {sqi_at_fire:.3f}")
                        if sqi_at_fire >= 0.3:
                            print(f"  [OK] SQI above threshold (0.3)")
                        else:
                            print(f"  [ERROR] SQI below threshold! Should not have fired!")
                    else:
                        error = sqi_result.get('error', 'unknown')
                        print(f"  SQI invalid: {error}")
                except Exception as e:
                    print(f"  Could not get SQI: {e}")

            # Check missiles-per-target tracking
            missiles_at_target = action_processor.missile_auto.missiles_per_target.get(enemy_id, 0)
            print(f"  Missiles at target {enemy_id}: {missiles_at_target}/2")

            # Check cooldown
            agent_state = action_processor.agent_states.get(agent_id, {})
            cooldown = agent_state.get('missile_cooldown_left_s', 0)
            print(f"  Cooldown set: {cooldown:.1f}s")

            if missiles_at_target >= 2:
                if cooldown == 10.0:
                    print(f"  [OK] Long cooldown (10s) applied after 2nd missile")
                else:
                    print(f"  [ERROR] Expected long cooldown (10s), got {cooldown:.1f}s")

        elif signals['vetoed_missile_attempts'] > 0:
            # Auto-fire tried but was vetoed
            if step_count <= 5:  # Only log first few vetos
                print(f"[STEP {step_count}] Auto-fire vetoed")

        # Advance simulation
        simulator.do_tick()

        # Stop if we've fired max missiles and waited through cooldown
        if missiles_fired >= 2 and step_count > 15:
            print(f"\n[STEP {step_count}] Test complete - fired {missiles_fired} missiles")
            break

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Steps completed: {step_count}")
    print(f"Missiles fired: {missiles_fired}")

    # Verify results
    success = True

    if missiles_fired == 0:
        print("\n[FAILED] No missiles fired!")
        print("  Possible causes:")
        print("  - SQI not calculated correctly")
        print("  - Auto-fire logic not triggering")
        print("  - Geometry not favorable enough")
        success = False
    elif missiles_fired < 2:
        print(f"\n[PARTIAL] Only {missiles_fired}/2 missiles fired")
        print("  Check cooldown and max missiles per target logic")
        success = False
    elif missiles_fired == 2:
        print("\n[SUCCESS] Correct number of missiles fired (2)")

        # Verify no more missiles after limit
        missiles_at_target = action_processor.missile_auto.missiles_per_target.get(enemy_id, 0)
        if missiles_at_target == 2:
            print("[OK] Missiles-per-target tracking correct (2/2)")
        else:
            print(f"[ERROR] Missiles-per-target mismatch: {missiles_at_target}/2")
            success = False
    else:
        print(f"\n[FAILED] Too many missiles fired ({missiles_fired})!")
        print("  Max missiles per target limit not enforced!")
        success = False

    # Test SQI directly
    print("\n" + "="*80)
    print("SQI VERIFICATION")
    print("="*80)

    if hasattr(agent, 'metrics') and hasattr(agent.metrics, 'get_sqi'):
        try:
            sqi_result = agent.metrics.get_sqi(enemy)

            if sqi_result.get('valid', False):
                final_sqi = sqi_result.get('sqi', 0.0)
                print(f"Final SQI: {final_sqi:.3f}")
                print(f"Threshold: 0.3")

                if final_sqi >= 0.3:
                    print("[OK] SQI above threshold - should trigger auto-fire")
                else:
                    print("[INFO] SQI below threshold - auto-fire correctly vetoed")
            else:
                error = sqi_result.get('error', 'unknown')
                print(f"[ERROR] SQI calculation invalid: {error}")
                success = False
        except Exception as e:
            print(f"[ERROR] SQI calculation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False
    else:
        print("[ERROR] No SQI calculation method found!")
        print("  Agent attributes:")
        print(f"  - has 'metrics': {hasattr(agent, 'metrics')}")
        if hasattr(agent, 'metrics'):
            print(f"  - metrics.get_sqi: {hasattr(agent.metrics, 'get_sqi')}")
        success = False

    print("\n" + "="*80)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("="*80)

    return success

if __name__ == "__main__":
    success = test_auto_fire_controlled()
    sys.exit(0 if success else 1)
