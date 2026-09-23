"""
Hydraulic relations linking the remotely sensed quantities to power.

  satellite width -> discharge Q     (at-a-station hydraulic geometry)
  DEM drop        -> hydraulic head H
  Q and H         -> power P = rho * g * Q * H * eta

w = a * Q^b is the at-a-station hydraulic geometry relation of Leopold and
Maddock (1953), which describes how width varies with discharge at a
single cross-section. The width exponent b is small (typically 0.0-0.3),
so inverting the relation to obtain Q from w amplifies width errors: to
first order, a relative width error e produces a relative discharge error
of about e/b. This sensitivity was one of the main limitations of the
approach.
"""

import numpy as np

RHO = 1000.0    # water density, kg/m3
G = 9.81        # gravitational acceleration, m/s2


def fit_hydraulic_geometry(width, discharge):
    """Fit w = a * Q^b by least squares on log(w) against log(Q).

    width (m) and discharge (m3/s) must be paired observations from the
    same dates. Returns a dict with a, b, b_stderr, r2 and n. A value of b
    well outside 0.05-0.35 usually indicates erroneous widths or
    mismatched dates.
    """
    width = np.asarray(width, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    ok = np.isfinite(width) & np.isfinite(discharge) & (width > 0) & (discharge > 0)
    if ok.sum() < 3:
        raise ValueError(
            f"need at least 3 paired width/discharge points, got {ok.sum()}"
        )
    x = np.log(discharge[ok])
    y = np.log(width[ok])
    n = len(x)

    b, log_a = np.polyfit(x, y, 1)
    resid = y - (b * x + log_a)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Standard error of b, the gradient of the log-log regression
    sxx = float(np.sum((x - x.mean()) ** 2))
    b_stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if n > 2 and sxx > 0 else np.nan

    return {"a": float(np.exp(log_a)), "b": float(b),
            "b_stderr": b_stderr, "r2": float(r2), "n": int(n)}


def discharge_from_width(width, a, b):
    """Invert the hydraulic geometry fit: Q = (w / a)^(1/b).

    Valid only for reaches with geometry similar to the section used for
    the fit.
    """
    width = np.asarray(width, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(width > 0, (width / a) ** (1.0 / b), np.nan)


def manning_velocity(hydraulic_radius, slope, n_manning=0.035):
    """Mean velocity (m/s) from Manning's equation, V = (1/n) R^(2/3) S^(1/2).

    n_manning is the roughness coefficient, typically 0.025-0.06 for
    natural channels. Not used by any of the scripts.
    """
    r = np.asarray(hydraulic_radius, dtype=float)
    s = np.asarray(slope, dtype=float)
    s = np.maximum(s, 0.0)
    return (1.0 / n_manning) * r ** (2.0 / 3.0) * np.sqrt(s)


def flow_duration_curve(daily_discharge):
    """Flow duration curve from a daily discharge record.

    Returns (exceedance_percent, discharge), sorted in descending order of
    discharge. The discharge at 30% exceedance is equalled or exceeded on
    30% of days.
    """
    q = np.asarray(daily_discharge, dtype=float)
    q = q[np.isfinite(q)]
    if len(q) == 0:
        return np.array([]), np.array([])
    q_sorted = np.sort(q)[::-1]
    ranks = np.arange(1, len(q_sorted) + 1)
    exceedance = 100.0 * ranks / (len(q_sorted) + 1)
    return exceedance, q_sorted


def design_flow(daily_discharge, exceedance_pct=30.0):
    """Discharge exceeded exceedance_pct percent of the time (default Q30)."""
    exc, q = flow_duration_curve(daily_discharge)
    if len(q) == 0:
        return np.nan
    return float(np.interp(exceedance_pct, exc, q))


def hydropower(discharge, head, efficiency=0.80):
    """Hydraulic power P = rho * g * Q * H * eta, in W.

    efficiency is the combined turbine, generator and penstock efficiency.
    head is the gross head from the DEM; no head losses are subtracted.
    """
    q = np.asarray(discharge, dtype=float)
    h = np.asarray(head, dtype=float)
    return RHO * G * q * h * efficiency


def power_kw(discharge, head, efficiency=0.80):
    """Hydraulic power in kW (see hydropower)."""
    return hydropower(discharge, head, efficiency) / 1000.0


def annual_energy_mwh(power_watts, capacity_factor=0.5):
    """Approximate annual energy yield (MWh/yr) for a constant capacity factor."""
    return np.asarray(power_watts, dtype=float) * 8760.0 * capacity_factor / 1e6
