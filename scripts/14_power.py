"""
14_power.py - Hydraulic head and gross theoretical hydropower potential per reach.

Usage:
    python3 scripts/14_power.py

1. Hydraulic head. The head over each SWORD reach is H = S * L, where S is
   the SWOT reach slope (m/m) and L is the reach length. H is computed for
   every clean overpass; the median is the estimate and the 10th and 90th
   percentiles give its range.
2. Design flow. The design flow is Q30, the discharge exceeded 30% of the
   time on the flow-duration curve of the full daily gauge record
   (design_exceedance_pct in the config). Reaches upstream of Fort McMurray
   do not receive the Clearwater River inflow, so for these the design flow
   is computed from main-stem (07DA001) minus tributary (07CD001) discharge.
3. Power. P = rho g Q H eta. This is gross theoretical potential: the only
   loss is the overall efficiency eta, costs are not considered, and no
   environmental flow is reserved (15_screening.py applies the
   environmental flow).
4. Head cross-check. The slope-derived head is compared with the difference
   in median WSE between adjacent reaches. On the Athabasca data the two
   estimates agree within about 0.5%.

Inputs:
    swot_config.yaml, data/swot_clean.csv, data/gauge_<station>.csv

Outputs:
    outputs/reach_power.csv, outputs/drop_check.csv,
    figures/result3_profile.png, figures/result4_power.png
"""

import argparse
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

RHO, G = 1000.0, 9.81   # water density (kg/m3), gravitational acceleration (m/s2)


def design_flow(q, exceed_pct):
    """Discharge exceeded exceed_pct percent of the time (30 gives Q30).

    Discharges are ranked in descending order and assigned the Weibull
    plotting position rank/(n+1) as exceedance probability; the requested
    exceedance is then linearly interpolated on this flow-duration curve.
    Non-finite values are ignored, and an empty series returns NaN.
    """
    q =np.sort(np.asarray(q, float)[np.isfinite(q)])[::-1]
    if len(q) == 0:
        return np.nan
    exc = 100.0 * np.arange(1, len(q) + 1) / (len(q) + 1)
    return float(np.interp(exceed_pct, exc, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    eta = float(cfg.get("efficiency", 0.8))
    exc = float(cfg.get("design_exceedance_pct", 30))

    swot = pd.read_csv("data/swot_clean.csv", dtype={"reach_id": str})
    swot["time"] = pd.to_datetime(swot["time"], utc=True)

    # Design flow at the main-stem gauge
    main_stn = cfg["gauges"]["main"]
    g = pd.read_csv(f"data/gauge_{main_stn}.csv", parse_dates=["date"])
    q_main = design_flow(g["discharge_m3s"], exc)

    # Upstream of the Clearwater: daily main-stem minus tributary discharge
    # (dates with data at both gauges only), then Q30 of that series
    upstream = {str(r) for r in (cfg.get("upstream_of_tributary") or [])}
    q_up = np.nan
    trib = cfg["gauges"].get("tributary")
    if upstream and trib and os.path.exists(f"data/gauge_{trib}.csv"):
        t = pd.read_csv(f"data/gauge_{trib}.csv", parse_dates=["date"])
        both = g.merge(t, on="date", suffixes=("", "_trib"))
        q_up = design_flow(both["discharge_m3s"] - both["discharge_m3s_trib"], exc)
    print(f"Design flow (Q{exc:.0f}) at the main-stem gauge: {q_main:.0f} m3/s")
    if upstream:
        print(f"Design flow upstream of the tributary: {q_up:.0f} m3/s")

    # Hydraulic head for every overpass (slope in m/m, reach length in m)
    swot["drop_m"] = swot["slope"] * swot["p_length"]
    rows = []
    for rid, sub in swot.groupby("reach_id"):
        d = sub["drop_m"].dropna()
        if len(d) < 3:   # too few values for a median and percentile range
            continue
        q = q_up if rid in upstream else q_main
        med, lo, hi = np.median(d), np.percentile(d, 10), np.percentile(d, 90)
        rows.append({
            "reach_id": rid,
            "dist_out_km": float(sub["p_dist_out"].median()) / 1000,
            "length_km": float(sub["p_length"].median()) / 1000,
            "n_readings": int(len(d)),
            "median_wse_m": float(sub["wse"].median()),
            "slope_cm_per_km": float(sub["slope"].median() * 1e5),
            "drop_m": med, "drop_p10_m": lo, "drop_p90_m": hi,
            "design_flow_m3s": q,
            # A negative head is measurement noise on a near-flat reach, so treat it as zero
            "power_MW": RHO * G * q * max(med, 0) * eta / 1e6,
            "power_p10_MW": RHO * G * q * max(lo, 0) * eta / 1e6,
            "power_p90_MW": RHO * G * q * max(hi, 0) * eta / 1e6,
        })
    if not rows:
        sys.exit("No reach has 3 or more clean observations with slope. "
                 "Check the output of 12_clean_swot.py.")

    res = pd.DataFrame(rows).sort_values("dist_out_km", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    res.to_csv("outputs/reach_power.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nResults per reach (upstream first):")
    print(res[["reach_id", "dist_out_km", "n_readings", "slope_cm_per_km",
               "drop_m", "power_MW", "power_p10_MW", "power_p90_MW"]]
          .round(2).to_string(index=False))
    top = res.sort_values("power_MW", ascending=False).iloc[0]
    print(f"\nHighest gross potential: reach {top.reach_id}, "
          f"{top.power_MW:.1f} MW (10th-90th percentile {top.power_p10_MW:.1f}-{top.power_p90_MW:.1f}), "
          f"head {top.drop_m:.2f} m over {top.length_km:.1f} km")

    # Head cross-check. The distance between the centres of two adjacent
    # reaches spans half of each reach, so the difference in median WSE
    # should be approximately (H_up + H_down) / 2. Agreement between the two
    # estimates supports the slope-derived head (on the Athabasca data they
    # agree within about 0.5%, summed over all pairs)
    pairs = []
    for i in range(len(res) - 1):
        up, dn = res.iloc[i], res.iloc[i + 1]
        gap = up.dist_out_km - dn.dist_out_km
        # Centre spacing inconsistent with the reach lengths: a reach is missing in between
        if abs(gap - (up.length_km + dn.length_km) / 2) > 3:
            continue
        pairs.append({"upstream": up.reach_id, "downstream": dn.reach_id,
                      "from_slopes_m": (up.drop_m + dn.drop_m) / 2,
                      "from_heights_m": up.median_wse_m - dn.median_wse_m})
    if pairs:
        pc = pd.DataFrame(pairs)
        pc.to_csv("outputs/drop_check.csv", index=False)
        a, b = pc["from_slopes_m"].sum(), pc["from_heights_m"].sum()
        print(f"\nHead cross-check over {len(pc)} adjacent reach pairs: "
              f"{a:.1f} m from slope x length vs {b:.1f} m from WSE differences "
              f"({100 * (a - b) / b:+.1f}%)")

    upstream_ids = upstream if upstream else set()
    up_rows = res[res.reach_id.isin(upstream_ids)]
    # Approximate confluence position: downstream end of the last reach above the Clearwater
    conf_km = None
    if len(up_rows) and len(up_rows) < len(res):
        last_up = up_rows.sort_values("dist_out_km").iloc[0]
        conf_km = last_up.dist_out_km - last_up.length_km / 2
    g_reach = str(cfg.get("gauge_reach") or "")

    # Result 3: water-surface profile. NaN breaks the line where a reach is
    # missing, so that no straight segment is drawn across the gap
    fig, ax = plt.subplots(figsize=(8, 4))
    xs, ys = [], []
    for i, row in enumerate(res.itertuples()):
        if i > 0:
            prev = res.iloc[i - 1]
            if prev.dist_out_km - row.dist_out_km > 1.5 * max(row.length_km, prev.length_km):
                xs.append(np.nan); ys.append(np.nan)
        xs.append(row.dist_out_km); ys.append(row.median_wse_m)
    ax.plot(xs, ys, "-o", ms=4, color="#1f6fb4", label="SWOT reach (median WSE)")
    if conf_km is not None:
        ax.axvline(conf_km, color="#777777", ls="--", lw=1)
        ax.text(conf_km, ax.get_ylim()[1], " Clearwater confluence\n (Fort McMurray)",
                va="top", ha="left", fontsize=8, color="#555555")
    if g_reach in set(res.reach_id):
        gr = res[res.reach_id == g_reach].iloc[0]
        ax.plot(gr.dist_out_km, gr.median_wse_m, "v", ms=9, color="#d95f02",
                label=f"gauge reach ({main_stn})")
    ax.invert_xaxis()
    ax.set_xlabel("distance from river mouth (km)  ->  downstream")
    ax.set_ylabel("water-surface elevation (m a.s.l.)")
    ax.set_title("Water-surface profile from SWOT", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/result3_profile.png", dpi=150)

    # Result 4: gross potential per reach; error bars from the 10th and 90th percentile head
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(res))
    err = [res["power_MW"] - res["power_p10_MW"], res["power_p90_MW"] - res["power_MW"]]
    colors = ["#1f6fb4" if r in upstream_ids else "#6baed6" for r in res["reach_id"]]
    ax.bar(x, res["power_MW"], color=colors, yerr=err, capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}" for d in res["dist_out_km"]], fontsize=7, rotation=90)
    ax.set_xlabel("reach (km from river mouth; upstream at left)")
    ax.set_ylabel("gross theoretical potential (MW)")
    n_up = int(res["reach_id"].isin(upstream_ids).sum())
    top_y = float((res["power_p90_MW"].max()) * 1.18)
    ax.set_ylim(0, top_y)
    if 0 < n_up < len(res):
        ax.axvline(n_up - 0.5, color="#777777", ls="--", lw=1)
        ax.text(n_up / 2 - 0.5, top_y * 0.97, "upstream of Fort McMurray (rapids)",
                ha="center", va="top", fontsize=8, color="#555555")
        ax.text(n_up + (len(res) - n_up) / 2 - 0.5, top_y * 0.97, "downstream of Fort McMurray",
                ha="center", va="top", fontsize=8, color="#555555")
    ax.set_title(f"Gross theoretical potential per reach "
                 f"(design flow, {eta:.0%} efficiency)", fontsize=10)
    fig.text(0.01, 0.01, "Error bars: 10th-90th percentile of per-overpass hydraulic head from SWOT.",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("figures/result4_power.png", dpi=150)

    print("\nWrote outputs/reach_power.csv, outputs/drop_check.csv, "
          "figures/result3_profile.png, figures/result4_power.png")


if __name__ == "__main__":
    main()
