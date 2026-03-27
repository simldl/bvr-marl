# Mock module for aircraft testing
from .aircraft import (
    MockAircraft,
    MockControl,
    MockCountermeasures,
    MockGun,
    MockMapLimits,
    MockNEZCalculator,
    MockPhysics,
    MockPosition,
    MockRadar,
    MockSensor,
    MockVelocity,
    MockWeaponSystem,
)
from .missiles import (
    MockAIM7,
    MockAIM9,
    MockAIM120,
    MockMissile,
    MockMissileEngine,
    MockMissileGuidance,
    MockMissilePhysics,
    MockMissileRadar,
)
from .simulator import (
    MockBenchmark,
    MockEnvironment,
    MockScenario,
    MockSimulator,
    create_basic_engagement_scenario,
    create_bvr_scenario,
    create_multi_threat_scenario,
)

__all__ = [
    # Aircraft mocks
    "MockPosition",
    "MockVelocity",
    "MockMapLimits",
    "MockPhysics",
    "MockRadar",
    "MockSensor",
    "MockGun",
    "MockWeaponSystem",
    "MockCountermeasures",
    "MockNEZCalculator",
    "MockControl",
    "MockAircraft",
    # Missile mocks
    "MockMissileRadar",
    "MockMissileGuidance",
    "MockMissileEngine",
    "MockMissilePhysics",
    "MockMissile",
    "MockAIM120",
    "MockAIM9",
    "MockAIM7",
    # Simulator mocks
    "MockSimulator",
    "MockScenario",
    "MockEnvironment",
    "MockBenchmark",
    "create_basic_engagement_scenario",
    "create_bvr_scenario",
    "create_multi_threat_scenario",
]
