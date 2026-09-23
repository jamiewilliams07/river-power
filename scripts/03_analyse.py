"""
Estimate hydropower for each candidate reach from width and slope, with
its uncertainty.

    python scripts/03_analyse.py --config config.yaml

Requires data/gauge_daily.csv with a date column (YYYY-MM-DD) and a
discharge_m3s column. Writes outputs/reach_power.csv and
outputs/variance_decomposition.csv, and saves figures to figures/.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riverlib import hydro as H
from riverlib import uncertainty as U

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not installed:  pip install pyyaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    widths = pd.read_csv("outputs/widths.csv")
    widths["date"] = pd.to_datetime(widths["date"], format="%Y%m%d")

    gauge = pd.read_csv("data/gauge_daily.csv")
    gauge["date"] = pd.to_datetime(gauge["date"])

    # Fit w = a*Q^b using widths within gauge_window_m of the gauge
    gauge_chainage = float(cfg["gauge_chainage_m"])
    tol = float(cfg.get("gauge_window_m", 300))
    at_gauge = widths[np.abs(widths.chainage_m - gauge_chainage) <= tol]

    paired = (at_gauge.groupby("date")["width_m"].median()
              .reset_index()
              .merge(gauge, on="date", how="inner"))

    print(f"Paired width/discharge observations: {len(paired)}")
    if len(paired) < 3:
        sys.exit("At least 3 same-day pairs are required. Add image dates or "
                 "check that image and gauge dates coincide.")

    fit = H.fit_hydraulic_geometry(paired.width_m.values,
                                   paired.discharge_m3s.values)
    print(f"  w = {fit['a']:.2f} * Q^{fit['b']:.3f}   "
          f"(b stderr {fit['b_stderr']:.3f}, r2 {fit['r2']:.3f}, n {fit['n']})")

    # Signal-to-noise check: the width change over the observed flow range
    # must be well above the width noise, otherwise the fitted b reflects noise
    dW = paired.width_m.max() - paired.width_m.min()
    noise = float(cfg.get("width_noise_m", 10.0))
    snr = dW / noise
    print(f"  width range {dW:.1f} m vs noise {noise:.1f} m -> SNR {snr:.1f}")
    if snr < 3:
        print("  WARNING: SNR below 3. The fitted b is unreliable "
              "at this site with this sensor.")
    if not (0.02 < fit["b"] < 0.4):
        print(f"  WARNING: b = {fit['b']:.3f} is outside the typical range "
              "for rivers (~0.05-0.35). Check that dates match and that the "
              "section is not at a bridge or lined bank.")

    # Design flow (Q30 unless set otherwise in the config)
    q_design = H.design_flow(gauge.discharge_m3s.values,
                             float(cfg.get("design_exceedance_pct", 30)))
    print(f"  design flow Q{cfg.get('design_exceedance_pct', 30):.0f} "
          f"= {q_design:.1f} m3/s (mean {gauge.discharge_m3s.mean():.1f})")

    # Power for each candidate reach
    slope_df = pd.read_csv("outputs/slope.csv")
    reach_len = float(cfg.get("candidate_reach_m", 2000))
    med_width = widths.groupby("chainage_m")["width_m"].median()

    results = []
    starts = np.arange(slope_df.chainage_m.min(),
                       slope_df.chainage_m.max() - reach_len, reach_len)
    for start in starts:
        end = start + reach_len
        seg = slope_df[(slope_df.chainage_m >= start) &
                       (slope_df.chainage_m <= end)]
        if len(seg) < 3 or not np.isfinite(seg.elevation_m).any():
            continue
        # TODO: the drop uses only the two end points. It should use a
        # least-squares fit, as reach_drop() in slope.py does, so that a
        # single noisy DEM pixel cannot determine it
        drop = seg.elevation_m.iloc[0] - seg.elevation_m.iloc[-1]
        if not np.isfinite(drop) or drop <= 0:
            continue

        w_sel = med_width[(med_width.index >= start) & (med_width.index <= end)]
        if len(w_sel) == 0:
            continue
        w = float(w_sel.median())

        q = float(H.discharge_from_width(w, fit["a"], fit["b"]))
        # Cap at the maximum gauged discharge; the fit is unconstrained beyond
        # the observed range
        q = min(q, float(gauge.discharge_m3s.max()))
        q_used = min(q, q_design)

        p = H.power_kw(q_used, drop, float(cfg.get("efficiency", 0.8)))
        results.append({
            "start_m": start, "end_m": end, "width_m": w,
            "drop_m": drop, "slope_pct": 100 * drop / reach_len,
            "discharge_m3s": q_used, "power_kw": p,
        })

    reaches = pd.DataFrame(results).sort_values("power_kw", ascending=False)
    reaches.to_csv("outputs/reach_power.csv", index=False)
    print(f"\nwrote outputs/reach_power.csv ({len(reaches)} candidate reaches)")
    if len(reaches):
        top = reaches.iloc[0]
        print(f"  best reach: {top.start_m:.0f}-{top.end_m:.0f} m, "
              f"drop {top.drop_m:.1f} m, {top.power_kw:.0f} kW")

    # Uncertainty analysis for the highest-power reach only
    if len(reaches):
        top = reaches.iloc[0]
        res = U.propagate(
            width_m=top.width_m, head_m=top.drop_m,
            a=fit["a"], b=fit["b"],
            width_sd=noise,
            head_sd=float(cfg.get("dem_vertical_sd_m", 5.0)),
            b_sd=fit["b_stderr"] if np.isfinite(fit["b_stderr"]) else 0.02,
            n_draws=20000, seed=7,
        )
        print(f"\n  best reach power: median {res['median_kw']:.0f} kW, "
              f"90% interval [{res['p05_kw']:.0f}, {res['p95_kw']:.0f}] kW")

        dec = U.variance_decomposition(
            width_m=top.width_m, head_m=top.drop_m,
            a=fit["a"], b=fit["b"],
            width_sd=noise,
            head_sd=float(cfg.get("dem_vertical_sd_m", 5.0)),
            b_sd=fit["b_stderr"] if np.isfinite(fit["b_stderr"]) else 0.02,
            n_draws=20000, seed=7,
        )
        pd.DataFrame([dec]).to_csv("outputs/variance_decomposition.csv",
                                   index=False)
        print("\n  variance in log(power) attributable to:")
        for k in ("width", "head", "b", "efficiency", "interaction"):
            print(f"    {k:<12} {100*dec[k]:5.1f}%")
        print(f"  -> dominant: {U.dominant_input(dec)}")
        print("\n  The dominant input indicates what to improve first: finer "
              "imagery, a more accurate DEM, or more gauge-matched dates.")

    # Figures
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sensor, grp in widths.groupby("sensor"):
        prof = grp.groupby("chainage_m")["width_m"].median()
        ax[0].plot(prof.index / 1000, prof.values, label=sensor, lw=1.2)
    ax[0].set_xlabel("chainage (km)")
    ax[0].set_ylabel("width (m)")
    ax[0].set_title("Channel width along reach, by sensor")
    ax[0].legend()

    ax[1].scatter(paired.discharge_m3s, paired.width_m, s=28, zorder=3)
    qq = np.linspace(paired.discharge_m3s.min(), paired.discharge_m3s.max(), 50)
    ax[1].plot(qq, fit["a"] * qq ** fit["b"], color="k", lw=1.2,
               label=f"w = {fit['a']:.1f} Q^{fit['b']:.3f}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("gauge discharge (m3/s)")
    ax[1].set_ylabel("satellite width (m)")
    ax[1].set_title(f"Width-discharge fit at the gauge (r2={fit['r2']:.2f})")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures/width_and_fit.png", dpi=150)

    exc, q = H.flow_duration_curve(gauge.discharge_m3s.values)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(exc, q, lw=1.4)
    ax.axvline(float(cfg.get("design_exceedance_pct", 30)),
               ls="--", color="k", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("exceedance (%)")
    ax.set_ylabel("discharge (m3/s)")
    ax.set_title("Flow duration curve")
    fig.tight_layout()
    fig.savefig("figures/flow_duration.png", dpi=150)

    if len(reaches):
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [dec[k] for k in ("width", "head", "b", "efficiency")]
        ax.barh(["width", "head", "exponent b", "efficiency"], vals)
        ax.set_xlabel("fraction of variance in log(power)")
        ax.set_title("Share of power uncertainty by input")
        fig.tight_layout()
        fig.savefig("figures/variance_decomposition.png", dpi=150)

    print("\nwrote figures/")


if __name__ == "__main__":
    main()
