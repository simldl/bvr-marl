from missiles.guidance.base import BaseGuidanceMode
from radar.core.utils import _angles_dist

class FovCaptureGuidance(BaseGuidanceMode):
    YAW_GAIN = 0.4
    PITCH_GAIN = 0.4
    def compute(self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs):
        az, el, _ = _angles_dist(missile_position, current_yaw_deg, current_pitch_deg, target_position)
        return current_yaw_deg + self.YAW_GAIN * az, current_pitch_deg + self.PITCH_GAIN * el