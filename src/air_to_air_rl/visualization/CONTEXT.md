# Visualization Module - Context

## Purpose
Provides real-time visualization and analysis tools for air combat simulations. Enables debugging, training monitoring, and scenario analysis through graphical displays.

## Key Components

### `2d_live_view.py`
- **Real-time display**: Live simulation visualization
- **Multi-platform view**: Aircraft, missiles, and radar contacts
- **Tactical overlays**: NEZ zones, engagement envelopes
- **Interactive controls**: Pause, zoom, pan functionality

### `scenplotter/` Subfolder
Scenario plotting and analysis tools:
- **Track visualization**: Aircraft and missile trajectories
- **Engagement analysis**: Shot opportunities and outcomes
- **Performance metrics**: Statistical displays and charts
- **Replay functionality**: Post-mission analysis

## Visualization Features

### Real-Time Display
```python
# Live view capabilities
- Aircraft positions and headings
- Missile trajectories and guidance
- Radar detection cones and tracks
- Engagement zones (NEZ, DLZ)
- Tactical information overlays
```

### Multi-Layer Rendering
- **Geographic layer**: Map boundaries and terrain
- **Unit layer**: Aircraft and missile positions
- **Sensor layer**: Radar coverage and detections
- **Tactical layer**: Engagement zones and threats
- **Information layer**: Text overlays and status

### Interactive Features
- **Zoom and pan**: Detailed area examination
- **Layer toggles**: Show/hide different information
- **Time controls**: Pause, slow-motion, fast-forward
- **Selection tools**: Click for detailed unit information

## Plotting Capabilities

### Trajectory Analysis
- **Ground tracks**: 2D flight path visualization
- **Altitude profiles**: 3D trajectory representation
- **Energy state**: Speed and altitude over time
- **Maneuver analysis**: G-loading and turn performance

### Engagement Visualization
- **Shot geometry**: Launch conditions and intercept
- **Miss distance**: Closest point of approach
- **Countermeasure effects**: ECM and chaff deployment
- **Multi-target scenarios**: Complex engagement analysis

### Performance Metrics
- **Kill ratios**: Win/loss statistics
- **Survival curves**: Agent longevity analysis
- **Tactical scores**: SQI and positioning metrics
- **Learning curves**: Training progress visualization

## Technical Implementation

### Graphics Backend
- **Matplotlib**: Primary plotting library
- **Real-time updates**: Efficient frame rendering
- **Vector graphics**: Scalable display elements
- **Color coding**: Intuitive friend/foe identification

### Data Integration
- **Simulation interface**: Direct access to simulator state
- **Event logging**: Recording critical events
- **State history**: Tracking temporal data
- **Export capabilities**: Save plots and animations

### Performance Optimization
- **Selective updates**: Only redraw changed elements
- **Level-of-detail**: Simplify distant objects
- **Memory management**: Efficient data structures
- **Frame rate control**: Smooth animation playback

## Analysis Tools

### Statistical Analysis
- **Outcome distributions**: Monte Carlo results
- **Parameter sensitivity**: Factor analysis
- **Correlation studies**: Multi-variable relationships
- **Trend analysis**: Performance over time

### Debugging Support
- **Step-through mode**: Frame-by-frame analysis
- **Variable inspection**: Real-time value display
- **Event markers**: Critical decision points
- **Error visualization**: Problem identification

### Training Visualization
- **Reward progression**: Learning curve display
- **Policy evolution**: Behavioral change over time
- **Exploration patterns**: Agent search behavior
- **Convergence analysis**: Training stability assessment

## Integration Points

### With Simulator
- **State polling**: Regular simulator queries
- **Event subscription**: Notification of critical events
- **Control interface**: Pause/resume simulation
- **Data export**: Save simulation results

### With RL Training
- **Training monitoring**: Real-time learning progress
- **Episode visualization**: Individual training runs
- **Performance benchmarking**: Comparative analysis
- **Hyperparameter effects**: Parameter sensitivity

### With Analysis Tools
- **Data pipeline**: Formatted output for analysis
- **Report generation**: Automated documentation
- **Presentation mode**: Clean displays for briefings
- **Animation export**: Video generation capability

## Usage Scenarios

### Development and Debugging
- **Algorithm verification**: Visual confirmation of behavior
- **Physics validation**: Realistic flight characteristics
- **Control system testing**: Response to inputs
- **Edge case analysis**: Boundary condition handling

### Training Monitoring
- **Agent behavior**: Real-time policy observation
- **Learning progress**: Skill development tracking
- **Environment interaction**: Agent-world dynamics
- **Performance bottlenecks**: Training issue identification

### Scenario Analysis
- **Tactical assessment**: Mission effectiveness
- **What-if analysis**: Alternative scenario exploration
- **Comparative studies**: Different algorithm comparison
- **Presentation material**: Results visualization

## Configuration Options
- **Display settings**: Colors, sizes, update rates
- **Layer control**: Selectable information overlays
- **Export formats**: PNG, SVG, MP4 video
- **Performance tuning**: Frame rate and quality balance

## Recent Enhancements
- **Improved performance**: Faster rendering pipeline
- **Enhanced interactivity**: Better user controls
- **Richer displays**: More tactical information
- **Better integration**: Seamless simulator connection