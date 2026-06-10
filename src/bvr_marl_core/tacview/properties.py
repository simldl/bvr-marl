class TacviewPropertyDict(dict):
    """
    A property dict for Tacview that extracts properties from arbitrary unit objects.
    Supports simple mapping and per-unit-type modification.
    """

    def __init__(self, unit):
        super().__init__()
        self["Name"] = getattr(unit, "name", "Unknown")
        self["Type"] = getattr(unit, "type", unit.__class__.__name__)
        self["Color"] = self._get_color(unit)

        self["Roll"] = getattr(unit, "roll_deg", 0.0)
        self["Pitch"] = getattr(unit, "pitch_deg", 0.0)
        self["Yaw"] = getattr(unit, "yaw_deg", 0.0)
        self["HDG"] = getattr(unit, "yaw_deg", 0.0)
        self["IAS"] = getattr(unit, "speed", 0.0)
        self["DataLink"] = 0

        # Radar cone visualization removed - no radar properties logged

    def _get_color(self, unit):
        if hasattr(unit, "group"):
            g = str(unit.group).lower()
            if g in ["blue", "a", "0", "agent"]:
                return "Blue"
            elif g in ["red", "b", "1", "opponent"]:
                return "Red"
        if getattr(unit, "is_missile", False):
            if hasattr(unit, "source") and hasattr(unit.source, "group"):
                return self._get_color(unit.source)
            return "Orange"
        return "Green"
