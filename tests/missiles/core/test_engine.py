import pytest

from air_to_air_rl.missiles.core.engine import MissileEngine


class DummySim:
    def __init__(self):
        self.removed = []

    def remove_unit(self, id):
        self.removed.append(id)


class DummyMissile:
    def __init__(self):
        self.id = "missile01"
        self.physics = type("Phys", (), {"constant_engine_F": 0})()
        self.phase_manager = type("PM", (), {"get_thrust_kN": lambda self_: 42.0})()
        self.speed = 200.0  # Initial speed above minimum threshold
        self.position = type("Pos", (), {"alt": 5000.0})()  # Above ground


def test_engine_fuel_and_thrust():
    missile = DummyMissile()
    engine = MissileEngine(missile, initial_fuel_s=5.0)
    sim = DummySim()

    # First update - fuel decreases, engine produces thrust
    engine.update(1.0, sim)
    assert engine.fuel_s == 4.0  # Check fuel_s directly
    assert missile.physics.constant_engine_F == 42_000
    assert engine.fuel_s > 0.0  # Still has fuel
    assert "missile01" not in sim.removed  # Should not be removed yet

    # Exhaust all fuel - but missile still has sufficient energy
    engine.update(5.0, sim)
    assert engine.fuel_s == 0.0  # Fuel is empty
    assert missile.physics.constant_engine_F == 0  # No thrust when empty
    assert "missile01" not in sim.removed  # Should not be removed yet (has energy)

    # Energy-based removal test - missile loses speed
    missile.speed = 30.0  # Below minimum effective speed (50 m/s)
    assert engine.should_remove_missile_on_energy(missile)  # Should indicate removal


def test_engine_energy_removal():
    """Test energy-based removal conditions"""
    missile = DummyMissile()
    engine = MissileEngine(missile, initial_fuel_s=0.0)  # No fuel

    # Test 1: High speed missile should not be removed
    missile.speed = 200.0
    missile.position.alt = 1000.0
    assert not engine.should_remove_missile_on_energy(missile)

    # Test 2: Low speed missile should be removed
    missile.speed = 30.0  # Below 50 m/s threshold
    assert engine.should_remove_missile_on_energy(missile)

    # Test 3: Missile that hit ground should be removed
    missile.speed = 200.0  # Good speed
    missile.position.alt = -200.0  # Below ground
    assert engine.should_remove_missile_on_energy(missile)

    # Test 4: Missile with fuel can still indicate energy-based removal
    # (The engine checks speed/altitude independently of fuel status)
    engine.fuel_s = 1.0  # Has fuel
    missile.speed = 10.0  # Very low speed
    missile.position.alt = -500.0  # Way below ground
    # should_remove_missile_on_energy only checks energy, not fuel
    assert engine.should_remove_missile_on_energy(missile)
