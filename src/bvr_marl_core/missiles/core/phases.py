from copy import deepcopy
from typing import ClassVar


class MissilePhaseManager:
    DEFAULT_PHASES: ClassVar[dict[str, dict[str, float]]] = {
        "boost": {"duration_s": 20.0, "thrust_kN": 50.0},
        "middle": {"duration_s": 30.0, "thrust_kN": 45.0},
        "terminal": {"duration_s": 50.0, "thrust_kN": 0.0},
    }

    def __init__(
        self, flight_phases: dict = None, life_time_s: float = None, motor_burn_s: float = None
    ):
        if flight_phases is None:
            if motor_burn_s is None:
                raise ValueError("motor_burn_s required for default phases.")
            flight_phases = deepcopy(self.DEFAULT_PHASES)
            boost = float(flight_phases["boost"]["duration_s"])
            flight_phases["middle"]["duration_s"] = max(0.0, float(motor_burn_s) - boost)

        self.flight_phases = flight_phases
        self.current_phase = "boost"
        self.motor_burn_s = (
            float(motor_burn_s)
            if motor_burn_s is not None
            else (
                float(self.flight_phases["boost"]["duration_s"])
                + float(self.flight_phases["middle"]["duration_s"])
            )
        )

    def update(self, elapsed_time: float):
        """Set phase based on elapsed time since launch; terminal begins at burnout."""
        boost = float(self.flight_phases["boost"]["duration_s"])
        if elapsed_time < boost:
            self.current_phase = "boost"
        elif elapsed_time < self.motor_burn_s:
            self.current_phase = "middle"
        else:
            self.current_phase = "terminal"

    def get_thrust_kN(self) -> float:
        """Return thrust for current phase (0 in terminal)."""
        if self.current_phase == "terminal":
            return 0.0
        return float(self.flight_phases[self.current_phase]["thrust_kN"])
