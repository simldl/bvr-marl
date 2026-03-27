class BaseGuidanceMode:
    def __init__(self, missile):
        self.missile = missile

    def compute(
        self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
    ):
        raise NotImplementedError
