# Fully Scripted Tactical Controller

A comprehensive, fully automated air-to-air combat system implementing advanced BVR (Beyond Visual Range) and WVR (Within Visual Range) tactics. This system provides complete flight control automation while leveraging the semi-automated helper for countermeasures and target selection.

## 🎯 Key Features

### Advanced BVR Tactics
- **Crank Maneuvers**: Maintain radar lock while reducing closure rate
- **Launch-and-Leave (Skate)**: Fire and immediately exit to avoid retaliation
- **Launch-and-Decide (Banzai)**: Fire and continue approach for follow-up engagement
- **Notch/Beam**: Defeat Doppler radar and incoming missiles
- **F-pole Management**: Tactical decisions based on missile active seeker range

### Energy Management
- **Specific Energy Optimization**: Balance kinetic and potential energy
- **Energy Climbs/Dives**: Trade altitude for speed and vice versa
- **Tactical Altitude Control**: Maintain energy advantage over opponents
- **Turn Performance Calculation**: Optimize maneuvering based on energy state

### Engagement Geometry
- **Aspect Angle Analysis**: Head-on, beam, and tail aspect considerations
- **SQI Integration**: Target-Interceptor-Predicted-Waypoint calculations
- **Attack Vector Optimization**: Choose optimal approach angles
- **Lead Pursuit**: Calculate intercept geometry for guns and missiles

### Range Management
- **NEZ Calculations**: No-Escape-Zone inner and outer boundaries
- **MAR Assessment**: Minimum Abort Range for defensive planning
- **F-pole/A-pole**: Critical missile guidance transition points
- **DOR Planning**: Desired Out Range for tactical disengagement

## 🏗️ Architecture

### Behavior Tree System
```
TacticalRoot (Selector)
├── EmergencyDefense (Sequence)
│   ├── IncomingMissile (Condition)
│   └── DefensiveManeuvers (Action)
├── BVRCombat (Sequence)
│   ├── HasBVRTarget (Condition)
│   └── BVRTactics (Selector)
│       ├── Crank (Sequence)
│       ├── LaunchAndLeave (Sequence)
│       ├── LaunchAndDecide (Sequence)
│       └── Approach (Sequence)
├── WVRCombat (Sequence)
│   └── WVRTactics (Selector)
└── SearchPatrol (Sequence)
```

### Tactical Modules

#### BVRTactics
- `execute_crank_maneuver()` - 60-70° off-nose positioning
- `execute_launch_and_leave()` - Missile-and-exit tactics
- `execute_launch_and_decide()` - Continued engagement tactics
- `execute_notch_maneuver()` - 90° perpendicular to threat
- `execute_beam_maneuver()` - Perpendicular to radar threat

#### EnergyManager
- `calculate_specific_energy()` - Total energy state assessment
- `assess_energy_advantage()` - Relative energy comparison
- `execute_energy_climb()` - Trade speed for altitude
- `execute_energy_dive()` - Trade altitude for speed
- `calculate_turn_performance()` - G-force and turn rate optimization

#### GeometryCalculator
- `calculate_aspect_angle()` - Target aspect relative to us
- `calculate_antenna_train_angle()` - Bearing to target from our nose
- `calculate_closure_rate()` - Rate of range change
- `calculate_engagement_envelope()` - Tactical geometry assessment

#### RangeCalculator
- `calculate_nez_ranges()` - No-Escape-Zone boundaries
- `calculate_f_pole_range()` - Missile active seeker transition
- `calculate_launch_envelope()` - Complete engagement envelope
- `calculate_mar_range()` - Minimum abort distance

## 🚀 Quick Start

### Basic Usage

```python
from automation.scripted_control import FullTacticalController
from automation.strategies import BalancedStrategy

# Create fully scripted controller
config = BalancedStrategy.create_config()
controller = FullTacticalController(aircraft, config)

# In simulation loop
action = controller.get_action(simulator, dt)
aircraft.apply_rl_action(action, simulator)

# Monitor status
status = controller.get_status()
print(f"Current maneuver: {status['current_maneuver']}")
print(f"Tactical state: {status['tactical_state']}")
```

### Advanced Configuration

```python
from automation.strategies import DefensiveStrategy, AggressiveStrategy

# Defensive configuration (early countermeasures, threat-focused)
defensive_config = DefensiveStrategy.create_config()
defensive_controller = FullTacticalController(aircraft, defensive_config)

# Aggressive configuration (late countermeasures, engagement-focused)  
aggressive_config = AggressiveStrategy.create_config()
aggressive_controller = FullTacticalController(aircraft, aggressive_config)
```

## 📊 Tactical Decision Logic

### BVR Engagement Priorities

1. **Emergency Defense** (Highest Priority)
   - Incoming missile within 3km → Immediate defensive maneuvers
   - Deploy countermeasures and execute evasive maneuvers

2. **BVR Combat** (High Priority)
   - Target range > 5km → BVR tactics
   - Choose between crank, launch-and-leave, or launch-and-decide
   - Based on threat count, energy state, and tactical advantage

3. **WVR Combat** (Medium Priority)
   - Target range ≤ 5km → WVR tactics
   - Gun engagement, short-range missiles, or BFM maneuvers

4. **Search/Patrol** (Default)
   - No immediate threats → Maintain energy and search

### Range-Based Tactical Decisions

| Range | Primary Tactic | Secondary Options |
|-------|---------------|------------------|
| >40km | Direct Approach | Energy Climb |
| 20-40km | Crank/Offset Approach | Launch-and-Decide |
| 10-20km | Launch-and-Leave | Launch-and-Decide |
| 5-10km | WVR Transition | Short-Range Missiles |
| <5km | BFM/Guns | IR Missiles |

### Aspect Angle Considerations

- **Head-on (0-30°)**: Offset approach for better missile kinematics
- **Oblique (30-60°)**: Direct engagement with energy management
- **Beam (60-120°)**: Flanking approach for tactical advantage
- **Tail (120-180°)**: Direct stern attack for maximum effectiveness

## 🎮 Control Integration

### Action Space Mapping
```python
action[0] = throttle        # Fully automated (0.0-1.0)
action[1] = yaw_change      # Fully automated (0.0-1.0)
action[2] = pitch_change    # Fully automated (0.0-1.0)
action[3] = target_select   # Semi-automated (from AutoHelper)
action[4] = missile_fire    # Fully automated (0.0/1.0)
action[5] = flares         # Semi-automated (from AutoHelper)
action[6] = chaff          # Semi-automated (from AutoHelper)
action[7] = ecm            # Semi-automated (from AutoHelper)
action[8] = decoys         # Semi-automated (from AutoHelper)
```

### Flight Control Translation
- **Yaw**: Desired heading → normalized turn rate (0.5 = no turn)
- **Pitch**: Desired pitch angle → normalized pitch rate (0.5 = level)
- **Throttle**: Direct mapping with tactical adjustments

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd automation/scripted_control
python test_scripted_control.py
```

Tests validate:
- Behavior tree execution
- Tactical decision logic
- Geometry calculations
- Range assessments
- Energy management
- Complete system integration

## 📈 Performance Metrics

### Tactical Effectiveness
- **Target Selection**: Automated threat prioritization
- **Engagement Timing**: Optimal launch window detection
- **Defensive Response**: <1s reaction to missile threats
- **Energy Efficiency**: Maintains optimal altitude/speed profile

### Flight Performance
- **Turn Optimization**: G-force limited maneuvering
- **Energy Management**: ±10% of optimal energy state
- **Altitude Control**: Maintains tactical altitude advantage
- **Speed Control**: Optimized for combat effectiveness

## 🔧 Customization

### Custom Tactical Behaviors

```python
def custom_maneuver(self, context):
    target = context.get('target')
    if target:
        # Custom tactical logic
        context['desired_yaw'] = calculate_custom_bearing(target)
        context['desired_pitch'] = calculate_custom_pitch(target)
        return NodeStatus.SUCCESS
    return NodeStatus.FAILURE

# Add to behavior tree
custom_node = ActionNode("CustomManeuver", custom_maneuver)
```

### Custom Energy Profiles

```python
# Modify energy manager preferences
controller.energy_manager.preferred_altitude_m = 15000  # Higher altitude
controller.energy_manager.preferred_speed_mps = 350    # Different speed
```

### Custom Range Parameters

```python
# Adjust missile parameters
controller.range_calc.default_missile_params.update({
    'max_range_m': 80000,     # Longer range missiles
    'nez_range_m': 50000,     # Extended NEZ
    'active_range_m': 15000   # Better active seekers
})
```

## 🚨 Safety Features

### Boundary Protection
- Automatic map boundary avoidance
- Minimum altitude enforcement
- Speed envelope protection
- G-force limitation

### Threat Response
- Immediate defensive response to critical threats
- Automatic countermeasure deployment
- Emergency evasive maneuvers
- Multiple threat prioritization

### System Monitoring
- Continuous performance monitoring
- Tactical state tracking
- Decision logging
- Error detection and recovery

## 🔮 Future Enhancements

### Planned Features
- **Multi-Aircraft Coordination**: Formation flying and cooperative engagement
- **Advanced Threat Assessment**: Machine learning threat classification
- **Dynamic Tactics**: Adaptive behavior based on opponent patterns
- **Mission-Specific Profiles**: CAP, SEAD, Interdiction optimized behaviors

### Research Areas
- **Reinforcement Learning Integration**: Hybrid scripted-learned behaviors
- **Real-time Adaptation**: Dynamic parameter tuning based on performance
- **Adversarial Testing**: Red team validation and improvement

## 📚 Technical References

Based on real-world air combat tactics and procedures:
- F/A-18 NATOPS Flight Manual tactical procedures
- BVR engagement doctrine from modern fighter operations
- Energy management principles from fighter pilot training
- Threat reaction procedures from operational manuals

The system implements proven tactical concepts adapted for simulation environments while maintaining realistic flight dynamics and combat decision-making processes.