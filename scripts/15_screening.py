"""
15_screening.py - SWOT-only discharge, mean velocity, annual energy and shortlist.

Usage:
    python3 scripts/15_screening.py   (run 14_power.py first)

The final stage of the analysis, in four parts. Default parameters are in
DEFAULTS and can be overridden under screening: in swot_config.yaml.

A. SWOT-only discharge. The rating curve from 13_validate.py converts SWOT
   WSE to discharge at the gauge reach and at reaches upstream of the
   Clearwater confluence, where it is calibrated against main-stem minus
   tributary discharge (07DA001 minus 07CD001). Power is linear in Q, so
   the percentage difference between SWOT-derived and gauged design flow on
   coincident dates approximates the error of a SWOT-only power estimate.
   Accuracy degrades near the confluence: the median absolute percentage
   error is about 22% and 12% at the two nearest reaches, compared with
   single digits further upstream. A backwater test, correlating the
   rating error with the Clearwater's share of total discharge, was
   inconclusive.

B. Mean velocity. Manning's equation for a wide rectangular channel,
   Q = (1/n) W D^(5/3) S^(1/2), is solved for depth D at the design flow
   using SWOT width W and slope S; mean velocity follows from continuity.
   No measured roughness coefficient is available for this river, so
   n = 0.035 is used, with 0.025 and 0.045 as bounds.

C. Annual energy. The daily discharge record is applied to a run-of-river
   plant sized for the design flow. An environmental flow of 10% of mean
   discharge is always left in the river, and the turbine does not operate
   when the available flow is below the minimum turbine flow (15% of the
   design flow). Outputs are annual energy (GWh/yr) and capacity factor
   (annual energy divided by installed capacity x 8760 h).

D. Screening. A reach passes if its head gradient meets the minimum and
   its SWOT head estimate is reliable (enough observations and a limited
   10th-90th percentile spread). Passing reaches are ranked by annual
   energy per km, and the top three form the shortlist.

Inputs:
    swot_config.yaml, outputs/reach_power.csv, data/swot_clean.csv,
    data/gauge_<station>.csv

Outputs:
    outputs/screening.csv, outputs/screening_summary.json,
    outputs/upstream_calibration.csv, figures/result5_flow_duration.png,
    figures/result6_velocity.png, figures/result7_screening.png
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Module names that begin with a digit cannot be imported with an import
# statement, so load the earlier scripts by file path
VAL = _load("validate", "13_validate.py")
PWR = _load("power", "14_power.py")
RHO, G = 1000.0, 9.81

# Defaults; each can be overridden in swot_config.yaml under screening:
DEFAULTS = {
    "env_flow_fraction_of_mean": 0.10,   # environmental flow as a fraction of mean discharge
    "min_turbine_fraction": 0.15,        # minimum turbine flow as a fraction of design flow
    "manning_n": [0.025, 0.035, 0.045],  # low / central / high roughness (assumed, not measured)
    "min_head_gradient_m_per_km": 0.5,   # below this the reach is "too flat"
    "min_readings": 30,                  # minimum number of clean overpasses
    "max_drop_spread": 0.30,             # maximum (p90 - p10) / median of the SWOT head
    "shortlist_size": 3,
    "hydrokinetic_reference_ms": 1.5,    # hydrokinetic reference velocity (velocity figure only)
}

# Figure colours
BLUE, LIGHT, ORANGE, GREY = "#1f6fb4", "#6baed6", "#d95f02", "#bdbdbd"


def daily_series(path):
    """Read a gauge CSV as a daily discharge Series indexed by date."""
    g = pd.read_csv(path)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return g.dropna(subset=["discharge_m3s"]).set_index("date")["discharge_m3s"]


def calibrate(swot, reach, qdaily, offset_h):
    """Fit the rating curve Q = a(h - h0)^b at one reach with the
    13_validate.py functions.

    Returns the fit, the leave-one-out predictions and metrics, and the
    SWOT-derived discharge for every overpass, or None if there are fewer
    than 8 matched dates or no admissible fit.
    """
    s = swot[swot["reach_id"] == reach].copy()
    if s.empty:
        return None
    # Same UTC to local-date shift as in 13_validate.py
    s["date"] = (s["time"] + pd.Timedelta(hours=offset_h)).dt.date
    m = s.merge(qdaily.rename("q_gauge"), left_on="date", right_index=True, how="inner")
    m = m.dropna(subset=["wse", "q_gauge"])
    m = m[m["q_gauge"] > 0].sort_values("time").reset_index(drop=True)
    if len(m) < 8:
        return None
    h, q = m["wse"].values, m["q_gauge"].values
    fit = VAL.fit_rating(h, q)
    if fit is None:
        return None
    m["q_swot_loo"] = VAL.leave_one_out(h, q, VAL.fit_rating, VAL.predict_rating)
    s["q_swot"] = VAL.predict_rating(fit, s["wse"].values)
    return {"reach": reach, "fit": fit, "matched": m, "all": s,
            "metrics": VAL.metrics(q, m["q_swot_loo"].values)}


def manning(Q, W, S, n):
    """Manning's equation for a wide rectangular channel (hydraulic radius
    approximately equal to depth).

    Q = (1/n) W D^(5/3) S^(1/2), solved for depth D. Returns depth and
    mean velocity.
    """
    D = (Q * n / (W * np.sqrt(S))) ** 0.6   # exponent 0.6 = 3/5
    # Continuity: V = Q / A with A = W D
    return D, Q / (W * D)


def annual_energy(q, head_m, q_design, eta, env_frac, min_frac):
    """Annual energy from the daily discharge record.

    Returns mean annual energy (GWh/yr), capacity factor, the share of
    energy produced in May-October, and the number of complete years used.
    """
    q = q.dropna()
    s = pd.Series(q.values, index=pd.to_datetime(q.index))
    counts = s.groupby(s.index.year).count()
    # Complete years only (at least 330 days); a partial year would bias the annual mean low
    full_years = counts[counts >= 330].index
    s = s[s.index.year.isin(full_years)]
    if s.empty or head_m <= 0:
        return np.nan, np.nan, np.nan, 0
    # Reserve the environmental flow first; the turbine takes the remainder up to the design flow
    q_env = env_frac * s.mean()
    qt = np.minimum(s - q_env, q_design).clip(lower=0)
    # Below the minimum turbine flow the unit cannot operate, so output is zero
    qt[qt < min_frac * q_design] = 0.0
    e_day_gwh = RHO * G * qt * head_m * eta * 24 / 1e9       # W x 24 h -> GWh
    per_year = e_day_gwh.groupby(e_day_gwh.index.year).sum()
    e = float(per_year.mean())
    p_inst_gw = RHO * G * q_design * head_m * eta / 1e9
    cf = e / (p_inst_gw * 8760) if p_inst_gw > 0 else np.nan
    # May-October approximates the open-water season; the river is
    # typically ice-covered from about November to April
    open_share = float(e_day_gwh[e_day_gwh.index.month.isin(range(5, 11))].sum()
                       / e_day_gwh.sum()) if e_day_gwh.sum() > 0 else np.nan
    return e, cf, open_share, len(full_years)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--utc-offset-h", type=float, default=-7.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sc = {**DEFAULTS, **(cfg.get("screening") or {})}   # config values override DEFAULTS
    eta = float(cfg.get("efficiency", 0.8))
    exc = float(cfg.get("design_exceedance_pct", 30))

    if not os.path.exists("outputs/reach_power.csv"):
        sys.exit("outputs/reach_power.csv not found; run 14_power.py first.")
    res = pd.read_csv("outputs/reach_power.csv", dtype={"reach_id": str})
    swot = pd.read_csv("data/swot_clean.csv", dtype={"reach_id": str})
    swot["time"] = pd.to_datetime(swot["time"], utc=True)

    main_stn = cfg["gauges"]["main"]
    trib_stn = cfg["gauges"].get("tributary")
    q_main = daily_series(f"data/gauge_{main_stn}.csv")
    upstream = {str(r) for r in (cfg.get("upstream_of_tributary") or [])}
    q_up, q_t = None, None
    if upstream and trib_stn and os.path.exists(f"data/gauge_{trib_stn}.csv"):
        q_t = daily_series(f"data/gauge_{trib_stn}.csv")
        # Discharge upstream of the Clearwater = main-stem minus tributary
        # discharge on the same day, clipped at zero because the difference
        # of two gauge records can be slightly negative
        both = pd.concat([q_main, q_t], axis=1, join="inner").dropna()
        q_up = (both.iloc[:, 0] - both.iloc[:, 1]).clip(lower=0)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    summary = {"exceedance_pct": exc, "efficiency": eta, "settings": sc}

    # ======== A. SWOT-only discharge
    print("A. Design flow from SWOT vs gauge (coincident overpass dates)")

    # Calibrate one reach and compare Q30 from SWOT-derived and gauged
    # discharge. Both must use the same dates, otherwise differences in
    # sampling would be mixed into the comparison
    def evaluate(role, reach, qd):
        c = calibrate(swot, reach, qd, args.utc_offset_h)
        if c is None:
            return None
        m = c["matched"]
        q30_g = PWR.design_flow(m["q_gauge"], exc)
        q30_s = PWR.design_flow(m["q_swot_loo"], exc)
        row = {"role": role, "reach_id": reach, "n_matched": int(len(m)),
               "n_all": int(c["all"]["q_swot"].notna().sum()),
               "dist_out_km": float(res.loc[res.reach_id == reach, "dist_out_km"].iloc[0])
               if (res.reach_id == reach).any() else np.nan,
               "loo_median_err_pct": c["metrics"]["median_abs_pct_err"],
               "loo_nse": c["metrics"]["nse"],
               "q30_gauge_same_dates": q30_g, "q30_swot_same_dates": q30_s,
               "q30_diff_pct": 100 * (q30_s - q30_g) / q30_g,
               "q30_swot_all_dates": PWR.design_flow(c["all"]["q_swot"], exc),
               "rating": c["fit"], "backwater_r": np.nan}
        return {**row, "_c": c}

    cals = []
    g = evaluate("gauge section", str(cfg.get("gauge_reach")), q_main)
    if g:
        cals.append(g)

    # Reaches upstream of the Clearwater are calibrated against main-stem
    # minus tributary discharge. Backwater hypothesis: high Clearwater
    # discharge may raise the Athabasca stage immediately upstream of the
    # confluence, so that SWOT WSE is high for a given discharge. If so, the
    # rating error should increase with the Clearwater's share of total
    # discharge.
    # Note: this test was inconclusive (see the module docstring)
    up_cals = []
    if q_up is not None:
        share = (q_t / q_main).dropna()
        cand = res[res["reach_id"].isin(upstream) & (res["n_readings"] >= 20)]
        # Nearest the confluence first
        for rid in cand.sort_values("dist_out_km")["reach_id"]:
            c = evaluate("upstream section", rid, q_up)
            if c is None:
                continue
            m = c["_c"]["matched"]
            sh = m["date"].map(share)
            rel = m["q_swot_loo"] / m["q_gauge"] - 1
            ok = sh.notna() & rel.notna()
            if ok.sum() >= 8 and sh[ok].std() > 0:
                c["backwater_r"] = float(np.corrcoef(sh[ok], rel[ok])[0, 1])
            up_cals.append(c)

    if up_cals:
        tab = pd.DataFrame([{k: v for k, v in c.items() if k not in ("_c", "rating")}
                            for c in up_cals])
        tab.to_csv("outputs/upstream_calibration.csv", index=False)
        print("  Upstream reaches (calibrated against main-stem minus tributary discharge), "
              "nearest the confluence first:")
        for c in up_cals:
            print(f"    {c['reach_id']} ({c['dist_out_km']:.0f} km): n={c['n_matched']}, "
                  f"median abs. error {c['loo_median_err_pct']:.1f}%, Q{exc:.0f} difference "
                  f"{c['q30_diff_pct']:+.1f}%, backwater r = {c['backwater_r']:+.2f}")
        # The first reach is at the confluence. The best reach is selected
        # from those further upstream (or from all of them if there is only one)
        adjacent = up_cals[0]
        adjacent["role"] = "at the Clearwater confluence"
        others = up_cals[1:] or up_cals
        best = min(others, key=lambda c: c["loo_median_err_pct"])
        if best is not adjacent:
            best = {**best, "role": "above the Clearwater (best section)"}
            cals.append(best)
        cals.append(adjacent)
        errs = [c["loo_median_err_pct"] for c in up_cals]
        diffs = [abs(c["q30_diff_pct"]) for c in up_cals]
        summary["upstream_calibration"] = {
            "n_sections": len(up_cals),
            "median_err_pct": float(np.median(errs)),
            "median_abs_q30_diff_pct": float(np.median(diffs)),
            "rest_median_err_pct": float(np.median([c["loo_median_err_pct"] for c in others])),
            "rest_median_abs_q30_diff_pct": float(np.median([abs(c["q30_diff_pct"]) for c in others])),
            "rest_median_backwater_r": float(np.nanmedian([c["backwater_r"] for c in others]))
            if any(np.isfinite(c["backwater_r"]) for c in others) else None,
            "adjacent": {k: v for k, v in adjacent.items() if k != "_c"},
            "best": {k: v for k, v in best.items() if k != "_c"},
        }
        print(f"  Median over upstream reaches: abs. error {np.median(errs):.1f}%, "
              f"|Q{exc:.0f} difference| {np.median(diffs):.1f}%")
        # If backwater is responsible, the correlation should be strongest at
        # the confluence and weaken upstream. A similar correlation at every
        # reach points instead to the main-stem minus tributary discharge
        # itself (the errors of two gauges combined). The thresholds
        # |r| >= 0.4 and a difference in |r| of 0.2 are my own choice
        r_adj = adjacent["backwater_r"]
        r_rest = summary["upstream_calibration"]["rest_median_backwater_r"]
        verdict = "none"
        if np.isfinite(r_adj) and abs(r_adj) >= 0.4:
            if r_rest is None or abs(r_adj) - abs(r_rest) >= 0.2:
                verdict = "backwater"
            else:
                verdict = "gauge_difference"
        summary["upstream_calibration"]["error_link"] = verdict
        rr = f"{r_rest:+.2f}" if r_rest is not None else "n/a"
        msg = {"backwater": f"strongest at the confluence (r = {r_adj:+.2f} vs {rr} "
                            "elsewhere), consistent with backwater from the Clearwater.",
               "gauge_difference": f"similar at every reach (r = {r_adj:+.2f} at the confluence, "
                                   f"{rr} elsewhere), more likely due to the main-stem "
                                   "minus tributary discharge than to backwater.",
               "none": "no clear relationship with Clearwater discharge."}[verdict]
        print(f"  Relationship between rating error and Clearwater share of discharge: {msg}")

    for c in cals:
        print(f"  {c['role']} ({c['reach_id']}): n={c['n_matched']}, median abs. error "
              f"{c['loo_median_err_pct']:.1f}%, Q{exc:.0f} gauge {c['q30_gauge_same_dates']:.0f} "
              f"vs SWOT {c['q30_swot_same_dates']:.0f} m3/s ({c['q30_diff_pct']:+.1f}%)")
    summary["calibrations"] = [{k: v for k, v in c.items() if k != "_c"} for c in cals]
    summary["design_flow_long_record"] = {
        "main": PWR.design_flow(q_main.values, exc),
        "above_tributary": PWR.design_flow(q_up.values, exc) if q_up is not None else None,
        "record_start": str(min(q_main.index)), "record_end": str(max(q_main.index)),
    }
    if cals:
        summary["max_power_diff_pct"] = max(abs(c["q30_diff_pct"]) for c in cals)

    # ======== B. Mean velocity from Manning's equation
    print("\nB. Mean velocity at design flow (Manning's equation, SWOT width and slope)")
    widths = swot.groupby("reach_id")["width"].median()
    # TODO: replace the assumed n with a measured or published roughness
    # coefficient for the Athabasca
    n_lo, n_mid, n_hi = sorted(sc["manning_n"])
    vel = []
    for r in res.itertuples():
        W = float(widths.get(r.reach_id, np.nan))
        S = r.slope_cm_per_km / 1e5
        Q = r.design_flow_m3s
        if not (np.isfinite(W) and W > 0 and S > 0 and Q > 0):
            vel.append((np.nan,) * 6)
            continue
        D_mid, V_mid = manning(Q, W, S, n_mid)
        _, V_fast = manning(Q, W, S, n_lo)     # lower n (smoother bed) gives higher velocity
        _, V_slow = manning(Q, W, S, n_hi)
        # Froude number as a check: Fr > 1 indicates supercritical flow
        froude = V_mid / np.sqrt(G * D_mid)
        vel.append((W, D_mid, V_mid, V_slow, V_fast, froude))
    vel = pd.DataFrame(vel, columns=["width_m", "depth_m", "velocity_ms",
                                     "velocity_low_ms", "velocity_high_ms", "froude"])
    res = pd.concat([res.reset_index(drop=True), vel], axis=1)
    # Kinetic power flux per unit flow area (0.5 rho V^3), for comparison
    # with hydrokinetic turbines
    res["hydrokinetic_kW_per_m2"] = 0.5 * RHO * res["velocity_ms"] ** 3 / 1000
    for r in res.itertuples():
        if np.isfinite(r.velocity_ms):
            print(f"  {r.reach_id}: width {r.width_m:.0f} m, depth {r.depth_m:.1f} m, "
                  f"mean velocity {r.velocity_ms:.2f} m/s "
                  f"(n range {r.velocity_low_ms:.2f}-{r.velocity_high_ms:.2f})")
    if (res["froude"] > 1).any():
        print("  Note: Froude number > 1 (supercritical flow) at some reaches; "
              "the Manning estimate is less reliable there.")

    # ======== C. Annual energy from the daily discharge record
    print(f"\nC. Annual energy (environmental flow {sc['env_flow_fraction_of_mean']:.0%} "
          f"of mean discharge, minimum turbine flow {sc['min_turbine_fraction']:.0%} of design flow)")
    energy = []
    for r in res.itertuples():
        # Upstream reaches use the main-stem minus tributary series
        q = q_up if (r.reach_id in upstream and q_up is not None) else q_main
        e, cf, share, ny = annual_energy(q, r.drop_m, r.design_flow_m3s, eta,
                                         sc["env_flow_fraction_of_mean"],
                                         sc["min_turbine_fraction"])
        energy.append((e, cf, share, ny))
    en = pd.DataFrame(energy, columns=["energy_GWh_per_yr", "capacity_factor",
                                       "open_water_share", "years_used"])
    res = pd.concat([res, en], axis=1)
    # Normalise by reach length so that reaches of different length can be compared
    res["energy_GWh_per_km"] = res["energy_GWh_per_yr"] / res["length_km"]
    print(f"  {int(res['years_used'].max())} complete years of daily discharge used. "
          f"Capacity factor {res['capacity_factor'].min():.2f}-{res['capacity_factor'].max():.2f} "
          "(governed by the flow regime rather than the head, so similar at all reaches).")
    print(f"  Total over all reaches: {res['energy_GWh_per_yr'].sum():.0f} GWh/yr; "
          f"{res['open_water_share'].median():.0%} of it in May-October.")

    # ======== D. Screening and shortlist
    res["head_gradient_m_per_km"] = res["drop_m"] / res["length_km"]
    # Relative spread of the per-overpass SWOT head: (p90 - p10) / median
    res["drop_spread"] = (res["drop_p90_m"] - res["drop_p10_m"]) / res["drop_m"]
    res["pass_gradient"] = res["head_gradient_m_per_km"] >= sc["min_head_gradient_m_per_km"]
    res["pass_confidence"] = ((res["n_readings"] >= sc["min_readings"]) &
                              (res["drop_spread"] <= sc["max_drop_spread"]))
    res["passes"] = res["pass_gradient"] & res["pass_confidence"]
    ranked = res[res["passes"]].sort_values("energy_GWh_per_km", ascending=False)
    short_ids = list(ranked["reach_id"].head(int(sc["shortlist_size"])))
    res["shortlisted"] = res["reach_id"].isin(short_ids)
    res["rank"] = res["reach_id"].map({r: i + 1 for i, r in enumerate(ranked["reach_id"])})

    # Reason for pass or fail, written to the screen_note column
    def why(r):
        out = []
        if not r.pass_gradient:
            out.append("too flat")
        # With too few observations the spread is not meaningful, so report only one reason
        if r.n_readings < sc["min_readings"]:
            out.append("too few readings")
        elif r.drop_spread > sc["max_drop_spread"]:
            out.append("drop too uncertain")
        return ", ".join(out) or "passes"
    res["screen_note"] = [why(r) for r in res.itertuples()]
    res.to_csv("outputs/screening.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nD. Screening (upstream first)")
    show = res[["reach_id", "dist_out_km", "head_gradient_m_per_km", "drop_m",
                "power_MW", "energy_GWh_per_yr", "capacity_factor", "velocity_ms",
                "screen_note", "rank"]].copy()
    print(show.round(2).to_string(index=False))
    print("\nShortlist (candidates for site investigation):")
    for i, rid in enumerate(short_ids, 1):
        r = res[res.reach_id == rid].iloc[0]
        print(f"  {i}. {rid} at {r.dist_out_km:.0f} km: {r.drop_m:.1f} m head over "
              f"{r.length_km:.1f} km, {r.power_MW:.0f} MW installed capacity, "
              f"{r.energy_GWh_per_yr:.0f} GWh/yr, capacity factor {r.capacity_factor:.2f}, "
              f"mean velocity {r.velocity_ms:.1f} m/s")
    # Check whether the shortlisted reaches are contiguous (same gap test as
    # the head cross-check in 14_power.py)
    if len(short_ids) > 1:
        d = res[res.reach_id.isin(short_ids)].sort_values("dist_out_km")
        gaps = np.diff(d["dist_out_km"].values) - (d["length_km"].values[:-1] + d["length_km"].values[1:]) / 2
        if np.all(np.abs(gaps) < 3):
            print("  These reaches are contiguous and should be assessed as one stretch "
                  "(several low-head plants or a single larger scheme).")

    # Summary JSON (read by report/make_summary.py)
    up_mask = res["reach_id"].isin(upstream)
    summary["velocity"] = {
        "n_values": [n_lo, n_mid, n_hi],
        "upstream_range_ms": [float(res.loc[up_mask, "velocity_ms"].min()),
                              float(res.loc[up_mask, "velocity_ms"].max())] if up_mask.any() else None,
        "downstream_range_ms": [float(res.loc[~up_mask, "velocity_ms"].min()),
                                float(res.loc[~up_mask, "velocity_ms"].max())] if (~up_mask).any() else None,
        "max_froude": float(res["froude"].max()),
    }
    summary["energy"] = {
        "total_GWh_per_yr": float(res["energy_GWh_per_yr"].sum()),
        "upstream_GWh_per_yr": float(res.loc[up_mask, "energy_GWh_per_yr"].sum()),
        "median_capacity_factor": float(res["capacity_factor"].median()),
        "years_used": int(res["years_used"].max()),
    }
    summary["n_sections"] = int(len(res))
    summary["n_pass"] = int(res["passes"].sum())
    summary["shortlist"] = [
        {k: (float(v) if isinstance(v, (np.floating, float)) else v)
         for k, v in res[res.reach_id == rid].iloc[0][
             ["reach_id", "dist_out_km", "length_km", "drop_m", "head_gradient_m_per_km",
              "power_MW", "energy_GWh_per_yr", "capacity_factor", "velocity_ms",
              "hydrokinetic_kW_per_m2"]].items()}
        for rid in short_ids]
    with open("outputs/screening_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ======== Figures
    # Result 5: flow-duration curves on the overpass dates, SWOT vs gauge
    if cals:
        fig, axes = plt.subplots(1, len(cals), figsize=(4.4 * len(cals), 3.8), squeeze=False)
        for ax, c in zip(axes[0], cals):
            m = c["_c"]["matched"]
            for col, lab, sty in (("q_gauge", "gauge", dict(color="k", lw=1.6)),
                                  ("q_swot_loo", "SWOT (leave-one-out)",
                                   dict(color=BLUE, lw=1.6, ls="--"))):
                q = np.sort(m[col].dropna().values)[::-1]
                ex = 100 * np.arange(1, len(q) + 1) / (len(q) + 1)
                ax.plot(ex, q, label=lab, **sty)
            ax.axvline(exc, color=GREY, lw=1, ls=":")
            ax.set_title(f"{c['role'][0].upper() + c['role'][1:]}\n({c['reach_id']}, "
                         f"median abs. error {c['loo_median_err_pct']:.0f}%)\n"
                         f"design flow: gauge {c['q30_gauge_same_dates']:.0f}, SWOT "
                         f"{c['q30_swot_same_dates']:.0f} m³/s ({c['q30_diff_pct']:+.1f}%)",
                         fontsize=9)
            ax.set_xlabel("exceedance (% of overpass dates)")
            ax.set_ylabel("discharge (m³/s)")
            ax.legend(fontsize=8)
        fig.suptitle("Flow-duration curves: SWOT vs gauge (coincident overpass dates)", fontsize=10)
        fig.tight_layout()
        fig.savefig("figures/result5_flow_duration.png", dpi=150)
        plt.close(fig)

    x = np.arange(len(res))
    labels = [f"{d:.0f}" for d in res["dist_out_km"]]
    n_up = int(up_mask.sum())

    # Dashed divider between reaches upstream and downstream of the confluence
    def divider(ax):
        if 0 < n_up < len(res):
            ax.axvline(n_up - 0.5, color="#777777", ls="--", lw=1)

    # Result 6: mean velocity; error bars span the range of n
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [BLUE if u else LIGHT for u in up_mask]
    err = [res["velocity_ms"] - res["velocity_low_ms"], res["velocity_high_ms"] - res["velocity_ms"]]
    ax.bar(x, res["velocity_ms"], color=colors, yerr=err, capsize=3, error_kw={"lw": 0.8})
    ax.set_ylim(0, max(sc["hydrokinetic_reference_ms"] * 1.2, float(res["velocity_high_ms"].max()) * 1.1))
    ax.axhline(sc["hydrokinetic_reference_ms"], color=ORANGE, lw=1, ls=":")
    ax.text(len(res) - 0.5, sc["hydrokinetic_reference_ms"],
            f"{sc['hydrokinetic_reference_ms']:g} m/s: approximate minimum for hydrokinetic turbines",
            ha="right", va="bottom", fontsize=7, color=ORANGE)
    divider(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_xlabel("reach (km from river mouth; upstream at left)")
    ax.set_ylabel("mean velocity at design flow (m/s)")
    ax.set_title(f"Mean velocity at design flow (Manning's equation, n = {n_mid}; "
                 f"error bars: n = {n_lo}-{n_hi})", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/result6_velocity.png", dpi=150)
    plt.close(fig)

    # Result 7: annual energy per reach, shortlisted reaches in orange
    fig, ax = plt.subplots(figsize=(8, 4.3))
    col = [ORANGE if s else (BLUE if p else GREY)
           for s, p in zip(res["shortlisted"], res["passes"])]
    ax.bar(x, res["energy_GWh_per_yr"], color=col)
    for i, r in enumerate(res.itertuples()):
        if r.shortlisted:
            ax.text(i, r.energy_GWh_per_yr, f"#{int(r.rank)}", ha="center", va="bottom",
                    fontsize=8, color=ORANGE, fontweight="bold")
    divider(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_xlabel("reach (km from river mouth; upstream at left)")
    ax.set_ylabel("annual energy (GWh/yr)")
    ax.set_title("Annual energy per reach and shortlist", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="shortlisted"),
                       Patch(color=BLUE, label="passes screening criteria"),
                       Patch(color=GREY, label="fails (too flat or uncertain head)")],
              fontsize=8, loc="upper right")
    fig.text(0.01, 0.01,
             f"Criteria: head gradient >= {sc['min_head_gradient_m_per_km']} m/km; >= {sc['min_readings']} "
             f"clean overpasses; SWOT head spread <= {sc['max_drop_spread']:.0%}. Ranked by annual energy per km.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/result7_screening.png", dpi=150)
    plt.close(fig)

    print("\nWrote outputs/screening.csv, outputs/screening_summary.json, "
          "figures/result5_flow_duration.png, result6_velocity.png, result7_screening.png")


if __name__ == "__main__":
    main()
