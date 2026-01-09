"""
Centralized observation space dimension constants.

This module defines all dimension constants for the structured observation space
to ensure consistency across builders, encoders, and tests.

All constants follow the naming convention from the paper:
- K_* : Number of entity slots
- d_* : Feature dimension per entity
- b_* : Dimension of binary/auxiliary features
"""

# =============================================================================
# ENTITY SLOT COUNTS (K_*)
# =============================================================================

# Friendly entities
K_FF = 2  # Number of friendly fighter slots
K_FM = 4  # Number of friendly missile slots

# Enemy entities
K_EF = 2  # Number of enemy fighter slots
K_EM = 4  # Number of enemy missile slots

# Threat detection slots
K_PR = 2  # Number of passive radar detection slots
K_WARN = 4  # Number of warning sectors for missile warning system


# =============================================================================
# FEATURE DIMENSIONS PER ENTITY (d_*)
# =============================================================================

# Ownship state
d_OWN = 22
"""Ownship state dimension (22 features):
[0-5]:   Position/orientation (lat, lon, alt, yaw, speed, pitch)
[6]:     Remaining missiles (normalized)
[7]:     Ps (specific excess power, normalized)
[8]:     SQI (shot quality index) [0-1]
[9]:     DLZ zone encoded [0-1] (R1=0.25, R2=0.5, R3=0.75, R4=1.0)
[10-14]: Envelope scalars (s_factor, c_factor) and radar features
[15-17]: Radar features (max_range, FOV, power normalized)
[18]:    Own RCS normalized
[19]:    Radar lock quality [0-1]
[20-21]: Boundary distances normalized (min of 4 directions)
"""

# Friendly fighters
d_FF = 6
"""Friendly fighter feature dimension per slot (6 features):
[0-2]: Relative position (dx, dy, dz) in ENU frame [m]
[3-5]: Relative velocity (dvx, dvy, dvz) [m/s]
"""

# Enemy fighters
d_EF = 9
"""Enemy fighter feature dimension per slot (9 features):
[0-2]: Relative position (dx, dy, dz) in ENU frame [m]
[3-5]: Relative velocity (dvx, dvy, dvz) [m/s]
[6]:   Track confidence [0-1]
[7]:   NEZ active (active radar NEZ) [0-1 normalized]
[8]:   NEZ passive (passive radar NEZ) [0-1 normalized]
"""

# Friendly missiles
d_FM = 8
"""Friendly missile feature dimension per slot (8 features):
[0-2]: Relative position (dx, dy, dz) in ENU frame [m]
[3-5]: Relative velocity (dvx, dvy, dvz) [m/s]
[6]:   Phase encoded [0-1] (boost=0.33, mid=0.67, terminal=1.0)
[7]:   Seeker lock state [0/1]
"""

# Enemy missiles
d_EM = 7
"""Enemy missile feature dimension per slot (7 features):
[0-2]: Relative position (dx, dy, dz) in ENU frame [m]
[3-5]: Relative velocity (dvx, dvy, dvz) [m/s]
[6]:   Time-to-impact estimate (normalized) [0-1]
"""

# Passive radar detections
d_PR = 3
"""Passive radar detection dimension per slot (3 features):
[0]: Relative angle [deg, normalized]
[1]: Distance [m, normalized]
[2]: Age [s, normalized]
"""


# =============================================================================
# AUXILIARY FEATURE DIMENSIONS (b_*)
# =============================================================================

b_MWS = 1 + K_WARN
"""Missile warning system feature dimension (1 + K_WARN):
[0]:        Warning flag (normalized count)
[1:1+K_WARN]: Sector one-hot encoding per warning
"""

b_PR = K_PR * d_PR
"""Total passive radar feature dimension"""


# =============================================================================
# TOTAL OBSERVATION DIMENSION
# =============================================================================

def get_total_obs_dim() -> int:
    """
    Calculate total flattened observation dimension.

    Returns:
        Total number of scalar features in flattened observation
    """
    own_block = d_OWN

    ff_block = K_FF * d_FF + K_FF  # features + masks
    fm_block = K_FM * d_FM + K_FM  # features + masks

    ef_block = K_EF * d_EF + K_EF  # features + masks (NEZ now included in d_EF)
    em_block = K_EM * d_EM + K_EM  # features + masks

    # Additional info arrays
    fm_target_indices = K_FM + K_FM  # indices + masks
    ff_lock_indices = K_FF + K_FF    # indices + masks

    # Threat cues
    mws_block = 1 + K_EM * (1 + K_WARN) + K_EM  # flag + dirs + masks
    pr_block = K_PR * d_PR + K_PR               # detections + masks

    total = (own_block + ff_block + fm_block + ef_block + em_block +
             fm_target_indices + ff_lock_indices + mws_block + pr_block)

    return total


# =============================================================================
# VALIDATION
# =============================================================================

def validate_constants():
    """Validate that all constants are sensible."""
    assert K_FF > 0, "Must have at least 1 friendly fighter slot"
    assert K_FM > 0, "Must have at least 1 friendly missile slot"
    assert K_EF > 0, "Must have at least 1 enemy fighter slot"
    assert K_EM > 0, "Must have at least 1 enemy missile slot"

    assert d_OWN >= 20, "Ownship state should have at least 20 features"
    assert d_FF == 6, "Friendly fighter should be 6D relative state"
    assert d_EF >= 7, "Enemy fighter should have at least 7 features (state + extras)"
    assert d_FM >= 6, "Friendly missile should have at least 6D relative state"
    assert d_EM >= 6, "Enemy missile should have at least 6D relative state"

    print(f"Observation space constants validated:")
    print(f"  Ownship: {d_OWN} dims")
    print(f"  Friendly fighters: {K_FF} slots × {d_FF} dims = {K_FF * d_FF} dims")
    print(f"  Friendly missiles: {K_FM} slots × {d_FM} dims = {K_FM * d_FM} dims")
    print(f"  Enemy fighters: {K_EF} slots × {d_EF} dims = {K_EF * d_EF} dims")
    print(f"  Enemy missiles: {K_EM} slots × {d_EM} dims = {K_EM * d_EM} dims")
    print(f"  Total observation dimension: {get_total_obs_dim()}")


if __name__ == "__main__":
    validate_constants()
