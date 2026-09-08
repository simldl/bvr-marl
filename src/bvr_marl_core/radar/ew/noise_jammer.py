"""Noise (barrage) jamming as a range-denial effect.

Model (see 04_Radar.tex, EW section). A self-protection noise jammer co-located
with a target raises the victim radar's in-band noise floor. Because the skin echo
falls off as R^{-4} while the one-way jamming falls off only as R^{-2}, the
jam-to-signal ratio grows with range: far away the jammer buries the skin return's
*range* information (the victim still sees a strong return on the correct bearing —
a strobe — but cannot measure range), while close in the echo overpowers the jammer
and the radar "burns through", recovering range.

Balancing realism and control, a jammer is specified by its nominal burn-through
range against a *reference* radar and target (``burn_through_km`` vs the reference
below). The actual burn-through range then scales with the physical jam-to-signal
geometry: equating skin echo S ~ P_t G^2 sigma / R^4 and one-way jam J ~ ERP G / R^2
gives R_bt ~ sqrt(P_t G sigma / ERP), so a more powerful victim radar or a larger
target RCS burns through farther. We therefore scale the configured reference range
by sqrt( (P_t G sigma) / (P_t G sigma)_ref ). Missile seekers are not jam-susceptible
and always keep range.
"""

import math

# Reference victim radar / target the nominal burn-through range is quoted against:
# a legacy-class fighter radar (18 kW, 36 dB) viewing a sigma = 1 m^2 aspect. Against
# a stronger radar or larger RCS the burn-through range grows as sqrt(P_t G sigma).
_REF_TX_POWER_W = 18e3
_REF_GAIN_LIN = 10.0 ** (36.0 / 10.0)
_REF_SIGMA = 1.0
_REF_PGS = _REF_TX_POWER_W * _REF_GAIN_LIN * _REF_SIGMA


def burn_through_range_m(victim_radar, burn_through_km: float, sigma_eff: float) -> float:
    """Range (m) inside which ``victim_radar`` burns through the jammer.

    Beyond this range the jammer denies range measurement; at or inside it the skin
    echo dominates and range is recovered. ``burn_through_km`` is the jammer's
    nominal burn-through range against the reference radar/target; it is scaled here
    by the actual jam-to-signal geometry. Returns 0.0 when there is no jammer.
    """
    if burn_through_km <= 0.0 or sigma_eff <= 0.0:
        return 0.0
    p_t = float(getattr(victim_radar, "tx_power_w", 0.0))
    gain_lin = 10.0 ** (float(getattr(victim_radar, "antenna_gain_db", 0.0)) / 10.0)
    pgs = p_t * gain_lin * float(sigma_eff)
    scale = math.sqrt(max(0.0, pgs / _REF_PGS))
    return float(burn_through_km) * 1000.0 * scale
