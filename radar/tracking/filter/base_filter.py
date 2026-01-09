import numpy as np


class BaseKFFilter:
    """Base interface for Kalman filter implementations."""
    
    def predict(self, dt: float): 
        """Predict step of the filter."""
        ...
    
    def update(self, z: np.ndarray): 
        """Update step of the filter."""
        ...
    
    def get_state(self) -> np.ndarray: 
        """Get current state estimate."""
        ...
    
    def get_covariance(self) -> np.ndarray: 
        """Get current covariance estimate."""
        ...
    
    def get_velocity(self) -> np.ndarray: 
        """Get current velocity estimate."""
        ...
    
    def set_state(self, x: np.ndarray, P: np.ndarray | None = None): 
        """Set state and optionally covariance."""
        ...