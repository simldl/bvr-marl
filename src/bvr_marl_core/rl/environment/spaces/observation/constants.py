"""
Centralized observation-space dimension constants.

The observation is a set of per-entity **token** blocks. Every entity token is
self-contained: its features are followed by a trailing **mask column** (1.0 =
real entity, 0.0 = padded slot), so the network encoder recovers an attention
padding mask directly from the token rather than from a separate parallel array.
Relational cues (which enemy a friendly missile/fighter is guiding on) are folded
into the owning entity's token as compact scalars, so token widths do not depend
on the roster size — the space stays robust as the number of agents grows.

Naming:
- K_* : number of entity slots (capacity)
- d_* : feature dimension per entity token, INCLUDING the trailing mask column
"""

# --- Entity slot capacities ---
K_FF = 2  # Friendly fighter slots
K_FM = 4  # Friendly missile slots
K_EF = 2  # Enemy fighter slots
K_EM = 4  # Enemy missile slots
K_PR = 2  # Passive-radar (ESM) detection slots
K_MWS = 4  # Missile-warning threat slots

# Missile-warning per-token feature block: [urgency, sin(bearing), cos(bearing)].
WARN_FEATURE_DIM = 3


# --- Ownship state (single vector, no mask column) ---
d_OWN = 16
"""Ownship state dimension (16 features, all ~[-1, 1]) — see OwnStateBuilder.build:
[0-1]:   Heading sin(yaw), cos(yaw)
[2]:     Speed / max_speed
[3]:     Pitch / 90deg
[4]:     Remaining missiles (normalized)
[5]:     Ps (specific excess power, normalized)
[6]:     Radar lock quality on the selected target [0-1]
[7]:     Episode time remaining [0-1]
[8]:     Can-fire-now flag [0/1]
[9-11]:  Horizontal boundary: signed h-pos, signed v-pos, closing rate to nearest edge
[12-13]: Altitude boundary: signed altitude, closing rate to nearest floor/ceiling
[14]:    Fuel fraction remaining [0-1] (1.0 when fuel is not modelled)
[15]:    On-station endurance / station time, normalized by ~1 h [0-1]

REMOVED vs the previous 19-dim layout (redundant / poorly conditioned):
- envelope scalars s_factor, c_factor (TIPWI internals, also implicit in the
  enemy-fighter shot cues)
- in-envelope flag (redundant with the enemy DLZ zone + can-fire flag)
"""


OWN_IDX_CAN_FIRE = 8
"""Index of the can-fire-now flag within the ownship vector (see ``d_OWN`` above).

Named because the policy side needs it: it is the per-step "a launch would succeed
right now" signal (lock, FOV, gimbal, range, inventory, cooldown all satisfied), which
is exactly the mask for restricting the fire action's gradient to steps where the
shooting decision is real. Reading it by magic number from a flattened observation is
how that silently breaks the next time the ownship layout changes."""


# --- Active-sensing (EMCON) ownship extension ---
# When ``env.emcon_action_enabled`` is set, the ownship vector is widened with the
# agent's own radar-emission state so the policy can reason about WHEN it is sensing:
#   [radar_emitting flag, emission duty cycle so far, normalized time since last
#    on/off toggle].
# Width is fixed whenever EMCON is on (default off keeps d_OWN=16 unchanged).
OWN_EMCON_EXTRA = 3
d_OWN_EMCON = d_OWN + OWN_EMCON_EXTRA  # 19
# Reference (steps) for normalizing "time since last radar toggle" into [0, 1].
OWN_EMCON_TOGGLE_REF_STEPS = 30.0


def own_state_dim(emcon_action_enabled: bool) -> int:
    """Ownship vector width: base (16) or EMCON-extended (19)."""
    return d_OWN_EMCON if emcon_action_enabled else d_OWN


# --- Friendly missiles ---
FM_IDX_TARGET_KNOWN = 8
FM_IDX_TARGET_SLOT = 9
FM_IDX_MASK = 10
d_FM = 11
"""Friendly missile token (11 = 10 features + mask):
[0-2]: Relative position (forward, right, up), normalized
[3-5]: Relative velocity, normalized
[6]:   Phase encoded [0-1] (boost=0.33, mid=0.67, terminal=1.0)
[7]:   Seeker lock state [0/1]
[8]:   Has-target flag [0/1] — this missile is guiding on a known enemy-fighter slot
[9]:   Target enemy-fighter slot index, normalized to [0,1] (0 when none)
[10]:  Mask [0/1]
"""


# --- Friendly fighters ---
FF_IDX_LOCK_KNOWN = 6
FF_IDX_LOCK_SLOT = 7
FF_IDX_MASK = 8
d_FF = 9
"""Friendly fighter token (9 = 8 features + mask):
[0-2]: Relative position (forward, right, up), normalized
[3-5]: Relative velocity, normalized
[6]:   Has-lock flag [0/1] — this fighter holds a lock on a known enemy-fighter slot
[7]:   Locked enemy-fighter slot index, normalized to [0,1] (0 when none)
[8]:   Mask [0/1]
"""


# --- Enemy missiles ---
EM_IDX_TTI = 6
EM_IDX_MASK = 7
d_EM = 8
"""Enemy missile token (8 = 7 features + mask):
[0-2]: Relative position, normalized
[3-5]: Relative velocity, normalized
[6]:   Time-to-impact estimate, normalized [0-1] (0 = imminent)
[7]:   Mask [0/1]
"""


# --- Enemy fighters ---
EF_IDX_POS = slice(0, 3)
EF_IDX_VEL = slice(3, 6)
EF_IDX_CONFIDENCE = 6
EF_IDX_NEZ = 7  # merged active/passive NEZ (max), normalized by engagement range
EF_IDX_DLZ_R_MIN = 8
EF_IDX_DLZ_R_TR = 9  # NEZ / no-escape outer edge
EF_IDX_SQI = 10
EF_IDX_DLZ_ZONE = 11  # ordinal (R1=0.25, R2=0.50, R3=0.75, R4=1.00; 0=unknown)
EF_IDX_IS_SUPPORT = 12
EF_IDX_RID_KNOWN = 13
EF_IDX_RID_AIRFRAME = 14
EF_IDX_RID_WEAPON = 15
EF_IDX_RID_MISSILES_LEFT = 16
EF_IDX_BDA = 17
EF_IDX_MASK = 18
d_EF = 19
"""Enemy fighter token (19 = 18 features + mask):
[0-2]:  Relative position (forward, right, up), normalized
[3-5]:  Relative velocity, normalized
[6]:    Track confidence [0-1]
[7]:    NEZ range (max of active/passive), normalized by the engagement-range scale
[8]:    DLZ r_min normalized by the engagement-range scale
[9]:    DLZ r_tr (no-escape / NEZ-out edge) normalized by the engagement-range scale
[10]:   Shot quality index (SQI) [0-1]
[11]:   DLZ zone ordinal (R1=0.25 .. R4=1.00; 0=unknown)
[12]:   is_support_asset flag (1.0 = AWACS/support asset)
[13]:   RID known flag [0/1] — 1.0 once a weapons-quality lock resolves identity
[14]:   RID airframe type id, normalized [0-1] (0 = unknown)
[15]:   RID primary weapon type id, normalized [0-1] (0 = unknown)
[16]:   RID enemy missiles remaining, normalized [0-1] (0 = unknown)
[17]:   BDA flag [0/1] — 1.0 once this contact's kill has been confirmed
[18]:   Mask [0/1]

REMOVED vs the previous 22-dim layout (correlated shot-geometry features):
- DLZ r_pi and r_aero edges (intermediate/aero-max; the r_min/r_tr bracket + zone
  already localize the shot envelope)
- F-pole / supported-shot window flag (redundant with zone == R3)
- separate passive NEZ (merged with active into a single NEZ range)
"""


# --- Enemy-fighter token extension point ---
# An extension package may widen the enemy-fighter token with additional feature
# columns, appended BEFORE the mask column. The extra width is declared through
# ``env.ef_extra_dim`` so the observation space and the builder agree; the values
# are produced by an ``EnemyInfoBuilder`` subclass. Core itself adds nothing.


def ef_token_dim(extra_dim: int = 0) -> int:
    """Enemy-fighter token width: base (19) plus any extension columns."""
    return d_EF + int(extra_dim)


# Observation schema version. Bump whenever the token layout changes so old
# checkpoints fail loudly instead of silently reading shifted indices.
OBS_SCHEMA_VERSION = "2"


# --- Missile-warning threat tokens ---
MWS_IDX_MASK = WARN_FEATURE_DIM
d_MWS = WARN_FEATURE_DIM + 1
"""Missile-warning token (4 = [urgency, sin(bearing), cos(bearing)] + mask)."""


# --- Passive radar (ESM) detections ---
PR_IDX_MASK = 3
d_PR = 4
"""Passive radar token (4 = [rel_angle, distance, age] + mask)."""


def get_total_obs_dim() -> int:
    """Total flattened observation dimension (own + all entity token blocks)."""
    return (
        d_OWN + K_FM * d_FM + K_FF * d_FF + K_EM * d_EM + K_EF * d_EF + K_MWS * d_MWS + K_PR * d_PR
    )


def validate_constants():
    """Validate that all constants are sensible."""
    assert K_FF > 0 and K_FM > 0 and K_EF > 0 and K_EM > 0
    assert d_OWN >= 8, "Ownship state should have at least 8 features"
    for name, d in (("FF", d_FF), ("FM", d_FM), ("EM", d_EM), ("EF", d_EF)):
        assert d >= 2, f"{name} token must carry features + a mask column"

    print("Observation space constants validated:")
    print(f"  Ownship: {d_OWN} dims")
    print(f"  Friendly fighters: {K_FF} x {d_FF}   missiles: {K_FM} x {d_FM}")
    print(f"  Enemy fighters:    {K_EF} x {d_EF}   missiles: {K_EM} x {d_EM}")
    print(f"  Warnings: {K_MWS} x {d_MWS}   passive radar: {K_PR} x {d_PR}")
    print(f"  Total observation dimension: {get_total_obs_dim()}")


if __name__ == "__main__":
    validate_constants()
