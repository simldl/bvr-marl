import numpy as np
import pytest
from reinforcement_learning.environment.gym.spawn_utils import spawn_position, spawn_unit, MapBoundaryChecker

class DummyMapLimits:
    left_lon = -1
    right_lon = 1
    bottom_lat = -1
    top_lat = 1
    min_alt = 1000
    max_alt = 12000

class DummyAircraft:
    def __init__(self, pos, yaw, speed, group, map_limits, min_alt, max_alt):
        self.position = pos
        self.yaw_deg = yaw
        self.speed = speed
        self.group = group

class DummySim:
    def __init__(self): self.units = []; self.calls = []
    def add_unit(self, unit): self.units.append(unit); return len(self.units)
    def record_unit_trace(self, uid): self.calls.append(uid)

class DummyEnv:
    def __init__(self):
        self.aircraft_type_map = {"A": DummyAircraft, "B": DummyAircraft}
        self.map_size_km = 10
        self.map_limits = DummyMapLimits()
        self.simulator = DummySim()
        self.config = {"default_speed": 300}

def test_spawn_position_agent_and_opponent():
    limits = DummyMapLimits()
    pos_agent = spawn_position("agent", 10, limits)
    pos_op = spawn_position("opponent", 10, limits)
    assert limits.left_lon <= pos_agent.lon <= limits.right_lon
    assert limits.min_alt <= pos_agent.alt <= limits.max_alt
    assert limits.left_lon <= pos_op.lon <= limits.right_lon

def test_spawn_unit_registers():
    env = DummyEnv()
    uid = spawn_unit(env, "A", "agent")
    assert uid == 1
    assert isinstance(env.simulator.units[0], DummyAircraft)

def test_map_boundary_checker():
    u = type("U", (), {"position": type("P", (), {"lon": 0, "lat": 0})()})()
    limits = DummyMapLimits()
    assert MapBoundaryChecker.within_bounds(u, limits)
    u.position.lon = 2
    assert not MapBoundaryChecker.within_bounds(u, limits)
