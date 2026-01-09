import numpy as np
from missiles.core.movement import MissileMovement

class DummyMissile:
    def __init__(self):
        from simulator.core.helpers import Position
        self.position = Position(1,2,3)
        self.yaw_deg = 0
        self.pitch_deg = 0
        self.roll_deg = 0
        self.desired_yaw_deg = 5
        self.desired_pitch_deg = 2
        self.speed = 200
        self.max_speed_mps = 300
        self.map_limits = type("Limits", (), {
            "bottom_lat": 0, "top_lat": 10, "left_lon": 0, "right_lon": 10,
            "min_alt": 0, "max_alt": 10000
        })()

class DummyPhysics:
    def compute_movement(self, *a, **k):
        # return plausible values, including out-of-bound speed/pos
        return (12, 13, 9000, 350, 15, 8, 3)

def test_movement_update_and_clamping():
    missile = DummyMissile()
    physics = DummyPhysics()
    movement = MissileMovement(missile, physics)
    movement.update(1.0)
    # Should be clamped:
    assert 0 <= missile.position.lat <= 10
    assert 0 <= missile.position.lon <= 10
    assert 0 <= missile.position.alt <= 10000
    # Speed must be clamped
    assert missile.speed == 300
    # Heading should be updated
    assert missile.yaw_deg == 15
    assert missile.pitch_deg == 8
    assert missile.roll_deg == 3
