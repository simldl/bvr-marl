from bvr_marl_core.missiles.guidance.direct import DirectPursuitGuidance
from bvr_marl_core.missiles.guidance.lead import LeadInterceptGuidance
from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance


class TerminalGuidance:
    def __init__(self, missile):
        self.direct_guidance = DirectPursuitGuidance(missile)
        self.pn = PnPropNavGuidance(missile)
        self.missile = missile

    def compute(
        self, current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
    ):
        # Use direct pursuit for terminal guidance until PN is fixed
        return self.direct_guidance.compute(
            current_yaw_deg, current_pitch_deg, missile_position, target_position, tick_secs
        )
