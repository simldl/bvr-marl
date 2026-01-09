"""
Core semi-automated helper system that manages all aircraft actions except:
- Throttle control
- Yaw (heading) control  
- Pitch control
- Missile firing decision
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from automation.systems.threat_assessment import ThreatAssessment
from automation.systems.target_manager import TargetManager
from automation.systems.countermeasure_controller import CountermeasureController


class AutoHelperConfig:
    """Configuration for the AutoHelper system."""
    
    def __init__(self):
        # Target selection parameters
        self.target_priority_weights = {
            'range': 0.3,          # Closer targets have higher priority
            'threat_level': 0.4,   # More dangerous targets prioritized
            'angle': 0.2,          # Targets in better firing position
            'lock_quality': 0.1    # Quality of radar lock
        }
        
        # Countermeasure deployment parameters
        self.threat_response_thresholds = {
            'incoming_missile_distance_m': 1300,    # Deploy countermeasures when missile within 1.3km
            'radar_lock_time_s': 300.0,              # switched off (until improved counter measures )
            'multiple_threats': 2,                  # Number of threats to trigger defensive mode
        }
        
        # Automation behavior settings
        self.automation_level = 'balanced'  # 'defensive', 'balanced', 'aggressive'
        self.enable_countermeasures = True
        self.enable_target_selection = True
        self.enable_radar_management = True


class AutoHelper:
    """
    Semi-automated helper that intelligently manages aircraft systems.
    
    Handles everything except throttle, yaw, pitch, and missile firing decisions.
    """
    
    def __init__(self, aircraft, config: Optional[AutoHelperConfig] = None):
        self.aircraft = aircraft
        self.config = config or AutoHelperConfig()
        
        # Initialize subsystems
        self.threat_assessment = ThreatAssessment(aircraft, self.config)
        self.target_manager = TargetManager(aircraft, self.config)
        self.countermeasure_controller = CountermeasureController(aircraft, self.config)
        
        # State tracking
        self.last_update_time = 0.0
        self.current_threats = []
        self.selected_target = None
        
    def update(self, simulator, dt: float) -> Dict[str, Any]:
        """
        Update the automation system and return recommended actions.
        
        Args:
            simulator: The simulation environment
            dt: Time delta since last update
            
        Returns:
            Dict containing automated action recommendations
        """
        self.last_update_time += dt
        
        # Update threat assessment
        self.current_threats = self.threat_assessment.assess_threats(simulator)
        
        # Initialize action recommendations
        actions = {
            'target_selection': 0.0,  # For action[3]
            'countermeasures': {
                'flares': False,  # For action[6]
                'chaff':  False,  # For action[7]
                'ecm':    False,  # For action[8]
                'decoys': False,  # For action[9]
            }
        }
        
        # Target selection and management
        if self.config.enable_target_selection:
            target_action = self.target_manager.select_optimal_target(
                simulator, self.current_threats
            )
            actions['target_selection'] = target_action
            
        # Countermeasure deployment
        if self.config.enable_countermeasures:
            cm_actions = self.countermeasure_controller.evaluate_countermeasures(
                self.current_threats, simulator
            )
            actions['countermeasures'].update(cm_actions)

        self.selected_target = getattr(self.target_manager, "current_target", None)  

        return actions
        
    def get_action_values(self, simulator, dt: float, manual_controls: Dict[str, float]) -> np.ndarray:
        """
        Build full 10-D action vector (energy + lift-vector space):
        [0]=P_s (specific energy rate cmd, scaled to [0,1])
        [1]=n   (normal load factor cmd, scaled to [0,1])
        [2]=φ   (bank angle cmd, scaled to [0,1])
        [3]=target selector (continuous bin selector in [0,1])
        [4]=missile fire (trigger in [0,1])
        [5]=gun fire     (trigger in [0,1])
        [6]=flares (auto)
        [7]=chaff  (auto)
        [8]=ecm    (auto)
        [9]=decoys (auto)
        Notes:
        - The ActionProcessor (energy mode) maps [0]=P_s, [1]=n, [2]=φ to physics.
        - We keep weapon triggers (4,5) under manual/NN control by default.
        - Target selection (3) & countermeasures (6..9) are provided by the automation layer.
        """
        auto_actions = self.update(simulator, dt)

        # Initialize vector
        action = np.zeros(10, dtype=float)

        # ---- Manual / NN channels (energy + lift-vector) ----
        # Use multiple key aliases for convenience/backward-compat.
        Ps = manual_controls.get('Ps', 
            manual_controls.get('Ps_cmd', 
            manual_controls.get('energy', 0.5)))
        n  = manual_controls.get('n', 
            manual_controls.get('n_cmd', 0.5))
        phi = manual_controls.get('phi', 
            manual_controls.get('phi_cmd', 0.5))

        action[0] = float(np.clip(Ps,  0.0, 1.0))   # P_s command
        action[1] = float(np.clip(n,   0.0, 1.0))   # load factor command
        action[2] = float(np.clip(phi, 0.0, 1.0))   # bank angle command

        # Weapons: leave under manual/NN (you can also route through automation if desired)
        action[4] = float(np.clip(manual_controls.get('missile_fire', 0.0), 0.0, 1.0))
        action[5] = float(np.clip(manual_controls.get('gun_fire',     0.0), 0.0, 1.0))

        # ---- Automated systems ----
        # Target selection from AutoHelper
        action[3] = float(np.clip(auto_actions['target_selection'], 0.0, 1.0))

        # Countermeasures from AutoHelper
        cm = auto_actions['countermeasures']
        action[6] = 1.0 if cm.get('flares', False) else 0.0
        action[7] = 1.0 if cm.get('chaff',  False) else 0.0
        action[8] = 1.0 if cm.get('ecm',    False) else 0.0
        action[9] = 1.0 if cm.get('decoys', False) else 0.0

        return action
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the automation system."""
        return {
            'active_threats': len(self.current_threats),
            'selected_target': getattr(self.selected_target, 'id', None),
            'automation_level': self.config.automation_level,
            'countermeasures_enabled': self.config.enable_countermeasures,
            'target_selection_enabled': self.config.enable_target_selection,
            'threat_summary': self.threat_assessment.get_threat_summary() if hasattr(self, 'threat_assessment') else None
        }
        
    def set_automation_level(self, level: str):
        """
        Set the automation aggressiveness level.
        
        Args:
            level: 'defensive', 'balanced', or 'aggressive'
        """
        if level in ['defensive', 'balanced', 'aggressive']:
            self.config.automation_level = level
            # Update subsystem configurations
            self.threat_assessment.update_config(self.config)
            self.target_manager.update_config(self.config)
            self.countermeasure_controller.update_config(self.config)