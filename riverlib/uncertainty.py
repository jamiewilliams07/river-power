"""
Uncertainty in the power estimate and its attribution to each input.

Monte Carlo error propagation: the inputs are sampled from their assumed
distributions, power is computed for every draw, and the spread of the
result is summarised. Each input is then held fixed in turn to measure
how much the variance falls. Monte Carlo was used instead of first-order
(linearised) error propagation because Q = (w/a)^(1/b) is strongly
non-linear (b ~ 0.15 gives 1/b ~ 7), so a linear approximation is
inadequate.
"""

import numpy as np

from .hydro import discharge_from_width, hydropower


def propagate(width_m, head_m, a, b,
              width_sd=10.0, head_sd=5.0, b_sd=0.02,
              efficiency=0.80, efficiency_sd=0.05,
              n_draws=5000, seed=0):
    """Monte Carlo distribution of power for a single reach.

    width_m and head_m (m) are the central estimates; a and b come from
    the hydraulic geometry fit w = a*Q^b. The input standard deviations
    are width_sd (m, about one pixel per bank), head_sd (m, the DEM
    vertical error, about 5 m) and b_sd.

    Returns a dict with all draws, the median, 5th and 95th percentile
    power in kW, and the mean and coefficient of variation.
    """
    rng = np.random.default_rng(seed)

    w = rng.normal(width_m, width_sd, n_draws)
    h = rng.normal(head_m, head_sd, n_draws)
    b_draw = rng.normal(b, b_sd, n_draws)
    eta = np.clip(rng.normal(efficiency, efficiency_sd, n_draws), 0.05, 0.95)

    # Width and head must be non-negative, and b is bounded away from zero
    # because 1/b diverges as b approaches 0
    w = np.maximum(w, 1.0)
    h = np.maximum(h, 0.0)
    b_draw = np.maximum(b_draw, 0.01)

    q = discharge_from_width(w, a, b_draw)
    p = hydropower(q, h, eta) / 1000.0  # kW

    finite = np.isfinite(p)
    p_ok = p[finite]

    return {
        "power_kw": p,
        "discharge": q,
        "draws": {"width": w, "head": h, "b": b_draw, "efficiency": eta},
        "median_kw": float(np.median(p_ok)) if p_ok.size else np.nan,
        "p05_kw": float(np.percentile(p_ok, 5)) if p_ok.size else np.nan,
        "p95_kw": float(np.percentile(p_ok, 95)) if p_ok.size else np.nan,
        "mean_kw": float(np.mean(p_ok)) if p_ok.size else np.nan,
        "cv": float(np.std(p_ok) / np.mean(p_ok)) if p_ok.size else np.nan,
    }


def variance_decomposition(width_m, head_m, a, b,
                           width_sd=10.0, head_sd=5.0, b_sd=0.02,
                           efficiency=0.80, efficiency_sd=0.05,
                           n_draws=5000, seed=0):
    """Fraction of the variance of log(power) attributable to each input.

    Each input is fixed at its central value in turn (standard deviation
    set to zero) and the fractional reduction in variance is recorded. The
    fractions need not sum to one; the remainder is reported as
    'interaction'. log(power) is used because the power distribution is
    strongly right-skewed. All runs share the same seed, so the other
    inputs receive identical draws in every run.
    """
    inputs = ("width", "head", "b", "efficiency")
    central = {"width": width_m, "head": head_m,
               "b": b, "efficiency": efficiency}
    spread = {"width": width_sd, "head": head_sd,
              "b": b_sd, "efficiency": efficiency_sd}

    def run(frozen=None):
        sd = dict(spread)
        if frozen is not None:
            sd[frozen] = 0.0
        res = propagate(
            central["width"], central["head"], a, central["b"],
            width_sd=sd["width"], head_sd=sd["head"], b_sd=sd["b"],
            efficiency=central["efficiency"],
            efficiency_sd=sd["efficiency"],
            n_draws=n_draws, seed=seed,
        )
        p = res["power_kw"]
        p = p[np.isfinite(p) & (p > 0)]
        return float(np.var(np.log(p))) if p.size else np.nan

    total = run(None)
    if not np.isfinite(total) or total <= 0:
        return {k: np.nan for k in inputs}

    out = {}
    for name in inputs:
        reduced = run(name)
        out[name] = float(max(0.0, (total - reduced) / total))

    out["interaction"] = float(1.0 - sum(out[k] for k in inputs))
    out["total_log_variance"] = total
    return out


def dominant_input(decomposition):
    """Return the input with the largest variance share, or None."""
    candidates = {k: v for k, v in decomposition.items()
                  if k in ("width", "head", "b", "efficiency")
                  and np.isfinite(v)}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)
