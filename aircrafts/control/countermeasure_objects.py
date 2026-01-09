"""
Simple countermeasure objects that persist in the simulation for hit calculation.
Flares, chaff, and decoys stay in the simulation for multiple seconds.
"""
import numpy as np
from simulator.core.units import Unit


class Countermeasure(Unit):
    """Simple countermeasure that persists for multiple seconds in the simulation."""

    def __init__(self, parent_aircraft, cm_type: str, lifetime_s: float = 10.0):
        """
        Initialize a countermeasure.

        Args:
            parent_aircraft: Aircraft that deployed this countermeasure
            cm_type: Type of countermeasure ('flare', 'chaff', 'decoy', 'ecm')
            lifetime_s: How long the countermeasure persists (default: 10 seconds)
        """
        # Get required fields for Unit initialization
        name = f"{cm_type.upper()}_{parent_aircraft.name}_{id(self)}"
        position = parent_aircraft.position.copy()
        yaw_deg = getattr(parent_aircraft, 'yaw_deg', 0.0)
        speed = 0.0  # Countermeasures are stationary

        # Initialize Unit with required fields
        super().__init__(name=name, position=position, yaw_deg=yaw_deg, speed=speed)

        self.parent_aircraft = parent_aircraft
        self.cm_type = cm_type
        self.is_countermeasure = True
        self.is_missile = False
        self.is_destroyed = False

        # Lifetime tracking
        self.lifetime_s = lifetime_s
        self.age_s = 0.0

        # Additional attitude
        self.pitch_deg = getattr(parent_aircraft, 'pitch_deg', 0.0)
        self.roll_deg = getattr(parent_aircraft, 'roll_deg', 0.0)

        # Simple properties
        self.group = parent_aircraft.group
        self.should_be_removed = False

    def update(self, dt_s: float, sim) -> list:
        """
        Update countermeasure - just track age and remove when expired.

        Args:
            dt_s: Time step in seconds
            sim: Simulator instance

        Returns:
            Empty list (no events)
        """
        self.age_s += dt_s

        # Remove when lifetime expires
        if self.age_s >= self.lifetime_s:
            self.should_be_removed = True
            self.is_destroyed = True

        return []

    def substep_update(self, dt_s: float, sim) -> list:
        """Substep update (same as regular update for countermeasures)."""
        return self.update(dt_s, sim)


class Flare(Countermeasure):
    """IR countermeasure - persists for 5 seconds."""
    def __init__(self, parent_aircraft):
        super().__init__(parent_aircraft, 'flare', lifetime_s=5.0)


class Chaff(Countermeasure):
    """Radar countermeasure - persists for 8 seconds."""
    def __init__(self, parent_aircraft):
        super().__init__(parent_aircraft, 'chaff', lifetime_s=8.0)


class Decoy(Countermeasure):
    """Active decoy - persists for 15 seconds."""
    def __init__(self, parent_aircraft):
        super().__init__(parent_aircraft, 'decoy', lifetime_s=15.0)


# Factory function for easy creation
def create_countermeasure(cm_type: str, parent_aircraft, custom_params: dict = None):
    """
    Factory function to create simple countermeasures.

    Args:
        cm_type: 'flare', 'chaff', or 'decoy'
        parent_aircraft: Aircraft deploying the countermeasure
        custom_params: Ignored (for backward compatibility)

    Returns:
        Countermeasure instance that persists for several seconds
    """
    cm_type = cm_type.lower()

    if cm_type == 'flare':
        return Flare(parent_aircraft)
    elif cm_type == 'chaff':
        return Chaff(parent_aircraft)
    elif cm_type == 'decoy':
        return Decoy(parent_aircraft)
    else:
        raise ValueError(f"Unknown countermeasure type: {cm_type}")
