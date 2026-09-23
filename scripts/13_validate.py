"""
13_validate.py - Validation of SWOT reach observations against gauge 07DA001.

Usage:
    python3 scripts/13_validate.py [--reach ID] [--version C|D]

    --reach    reach to validate (default: gauge_reach in swot_config.yaml)
    --version  use observations from one SWOT product version only (C or D)

Before SWOT observations are used for hydropower screening, they are
validated against the Water Survey of Canada gauge on the Athabasca River
at Fort McMurray (07DA001). Each overpass is matched to the gauge daily
record on the same local date.

1. Water level. The SWOT water-surface elevation (WSE) anomaly is compared
   with the gauge stage anomaly. Anomalies (departures from the respective
   means) are used because gauge stage is referenced to a local datum.
2. Discharge from WSE. A rating curve Q = a(h - h0)^b is fitted between
   SWOT WSE and gauged discharge and evaluated out of sample in two ways:
   leave-one-out cross-validation, and a chronological split (fit on the
   first 70% of dates, predict the last 30%).
3. Discharge from width. Width is tested as an alternative predictor. It
   varies little with discharge on this river, so it is assessed by
   correlation before any fit is attempted (see the comment in main()).

Note: the first validation at reach 82285000481 was invalid because
Version D reused that reach ID for a different location, so observations
from two locations were combined. The validation was repeated after the ID
mapping was fixed (11b_find_version_d_ids.py and 12_clean_swot.py now
handle this).

Inputs:
    swot_config.yaml, data/swot_clean.csv, data/gauge_<station>.csv

Outputs (<reach> becomes <reach>_v<version> when --version is set):
    outputs/validation_matched_<reach>.csv
    outputs/validation_summary_<reach>.csv
    figures/result1_height_vs_level_<reach>.png
    figures/result2_rating_curve_<reach>.png
    figures/result2_predicted_vs_gauged_<reach>.png
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


# ---- Curve fitting

# Bounds on the rating-curve exponent b. Without them the fit converged to
# b = 20.7 at one reach, which reflects the fit following noise rather than
# a hydraulic control, so b is constrained to 0.8-5.
B_MIN, B_MAX = 0.8, 5.0


def fit_rating(h, q):
    """Fit the rating curve Q = a(h - h0)^b.

    For each trial h0 on a grid, fits a straight line to log(Q) against
    log(h - h0) by least squares, and keeps the h0 with the smallest
    residual sum of squares. h0 is the stage of zero flow, so the grid lies
    entirely below the lowest observed WSE. Fits with b outside
    B_MIN-B_MAX are rejected; returns None if no trial h0 is admissible.
    """
    h, q = np.asarray(h, float), np.asarray(q, float)
    span = max(np.ptp(h), 0.5)
    best = None
    for h0 in np.linspace(h.min() - 10 * span - 5, h.min() - 0.02, 600):
        x = np.log(h - h0)
        y = np.log(q)
        b, loga = np.polyfit(x, y, 1)
        sse = np.sum((y - (b * x + loga)) ** 2)
        if B_MIN <= b <= B_MAX and (best is None or sse < best[0]):
            best = (sse, np.exp(loga), b, h0)
    if best is None:
        return None
    _, a, b, h0 = best
    return {"a": a, "b": b, "h0": h0}


def predict_rating(p, h):
    h = np.asarray(h, float)
    # The rating curve is undefined at or below h0, so return NaN there
    with np.errstate(invalid="ignore"):
        return np.where(h > p["h0"], p["a"] * (h - p["h0"]) ** p["b"], np.nan)


def fit_width(w, q):
    """Fit the power law w = a Q^b (width as a function of discharge).

    Inverted for prediction as Q = (w/a)^(1/b).
    """
    b, loga = np.polyfit(np.log(q), np.log(w), 1)
    return {"a": np.exp(loga), "b": b}


def predict_width(p, w):
    # b <= 0 implies width decreasing with discharge, so the inversion is not usable
    if p["b"] <= 0:
        return np.full(len(w), np.nan)
    return (np.asarray(w, float) / p["a"]) ** (1.0 / p["b"])


def metrics(obs, pred):
    """Median absolute percentage error, Nash-Sutcliffe efficiency (NSE)
    and percentage bias.

    NSE = 1 - sum((P - O)^2) / sum((O - mean(O))^2): 1 is a perfect fit and
    0 means the predictions are no better than the mean of the observations.
    Pairs with non-finite values or non-positive observed discharge are
    excluded; with fewer than 3 valid pairs the metrics are NaN.
    """
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0)
    o, p = obs[ok], pred[ok]
    if len(o) < 3:
        return {"n": int(len(o)), "median_abs_pct_err": np.nan,
                "nse": np.nan, "bias_pct": np.nan}
    return {
        "n": int(len(o)),
        "median_abs_pct_err": float(np.median(np.abs(p - o) / o) * 100),
        "nse": float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)),
        "bias_pct": float((p.sum() - o.sum()) / o.sum() * 100),
    }


def leave_one_out(x, q, fit, predict):
    """Leave-one-out cross-validation: predict each point from a fit to all
    other points, so that the error is not underestimated by in-sample
    fitting. Points for which the fit fails are returned as NaN."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        p = fit(x[m], q[m])
        if p is not None:
            out[i] = predict(p, x[i:i + 1])[0]
    return out


# ---- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--reach", default=None,
                    help="reach ID to validate (default: gauge_reach in the config)")
    ap.add_argument("--version", choices=["C", "D"], default=None,
                    help="use observations from this SWOT product version only")
    ap.add_argument("--utc-offset-h", type=float, default=-7.0,
                    help="UTC offset of the gauge day in hours (Alberta: -7)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    reach = args.reach or cfg.get("gauge_reach")
    if not reach:
        sys.exit("gauge_reach is not set in swot_config.yaml.")
    reach = str(reach)
    stn = cfg["gauges"]["main"]

    swot = pd.read_csv("data/swot_clean.csv", dtype={"reach_id": str})
    swot = swot[swot["reach_id"] == reach].copy()
    tag = reach
    if args.version:
        swot = swot[swot["version"] == args.version]
        tag = f"{reach}_v{args.version}"
        print(f"Using Version {args.version} readings only.")
    elif "version" in swot:
        print("Observations by product version: " +
              ", ".join(f"{v}: {n}" for v, n in swot["version"].value_counts().items()))
    if swot.empty:
        sys.exit(f"No clean SWOT observations for reach {reach}.")
    swot["time"] = pd.to_datetime(swot["time"], utc=True)
    # SWOT times are in UTC while the gauge reports daily values in local
    # time; shift before extracting the date so that each overpass is
    # matched to the correct gauge day
    swot["date"] = (swot["time"] + pd.Timedelta(hours=args.utc_offset_h)).dt.date

    gauge = pd.read_csv(f"data/gauge_{stn}.csv")
    gauge["date"] = pd.to_datetime(gauge["date"]).dt.date

    m = swot.merge(gauge[["date", "discharge_m3s", "level_m", "source"]],
                   on="date", how="inner")
    m = m.dropna(subset=["wse", "discharge_m3s"])
    m = m[m["discharge_m3s"] > 0].sort_values("time").reset_index(drop=True)
    print(f"Reach {reach}: {len(swot)} clean SWOT observations, "
          f"{len(m)} on dates with gauged discharge.")
    if len(m) < 8:
        sys.exit("Fewer than 8 matched dates, too few for validation. Check that "
                 "the gauge record covers the SWOT overpass dates (10_fetch_gauge.py "
                 "--extra adds recent provisional data).")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    rows = []

    # ---- Result 1: SWOT WSE anomaly vs gauge stage anomaly
    lv = m.dropna(subset=["level_m"])
    if len(lv) >= 5:
        # Gauge stage is referenced to a local datum, so compare anomalies
        # rather than absolute values
        sa = lv["wse"] - lv["wse"].mean()
        ga = lv["level_m"] - lv["level_m"].mean()
        diff = sa - ga
        r = float(np.corrcoef(sa, ga)[0, 1])
        slope, icpt = np.polyfit(ga, sa, 1)
        slope = float(slope)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        # The RMSE of the anomaly difference is meaningful only at the gauge
        # reach. Elsewhere the stage variation can be larger or smaller than
        # at the gauge (regression slope not equal to 1), so the standard
        # deviation of the residuals about the regression line is used as
        # the measure of SWOT noise (ddof = 2 for the two fitted parameters)
        scatter = float(np.std(sa - (slope * ga + icpt), ddof=2))
        rows.append({"result": "1 height vs gauge level", "n": len(lv),
                     "rmse_m": rmse, "scatter_m": scatter,
                     "correlation": r, "slope": slope})
        print(f"\nResult 1: SWOT WSE anomaly vs gauge stage anomaly (n={len(lv)})")
        print(f"  correlation {r:.3f}, residual scatter about regression line {100*scatter:.0f} cm, "
              f"slope {slope:.2f}")
        print(f"  RMSE of anomaly difference {100*rmse:.0f} cm "
              "(meaningful only at the gauge reach)")

        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(ga, sa, s=22, color="#1f6fb4")
        lim = [min(ga.min(), sa.min()) - 0.2, max(ga.max(), sa.max()) + 0.2]
        ax.plot(lim, lim, color="k", lw=1, ls="--", label="1:1 (equal anomalies)")
        xx = np.array(lim)
        ax.plot(xx, slope * xx + icpt, color="#d95f02", lw=1.2,
                label=f"linear regression (slope {slope:.2f})")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("gauge stage anomaly (m)")
        ax.set_ylabel("SWOT WSE anomaly (m)")
        ax.set_title(f"Reach {reach}: r = {r:.2f}, residual scatter {100*scatter:.0f} cm "
                     f"({len(lv)} dates)",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"figures/result1_height_vs_level_{tag}.png", dpi=150)
    else:
        print("\nResult 1 skipped: no gauge stage data on the matched dates.")

    # ---- Result 2: discharge from SWOT WSE (rating curve)
    h = m["wse"].values
    q = m["discharge_m3s"].values

    full = fit_rating(h, q)
    if full is None:
        sys.exit(f"No rating curve with b in {B_MIN}-{B_MAX} fits these observations; "
                 "SWOT WSE does not track discharge at this reach.")
    loo = leave_one_out(h, q, fit_rating, predict_rating)
    # m is sorted by time, so this fits on the first 70% of dates and tests
    # on the last 30%. It is a stricter test than leave-one-out because the
    # model must predict forward in time
    k = int(round(0.7 * len(m)))
    split_fit = fit_rating(h[:k], q[:k])
    split_pred = predict_rating(split_fit, h[k:]) if split_fit else np.full(len(m) - k, np.nan)

    mh_loo = metrics(q, loo)
    mh_split = metrics(q[k:], split_pred)
    rows.append({"result": "2 flow from height, leave-one-out", **mh_loo,
                 "a": full["a"], "b": full["b"], "h0": full["h0"]})
    rows.append({"result": "2 flow from height, time split (last 30%)", **mh_split})

    print(f"\nResult 2: discharge predicted from SWOT WSE")
    print(f"  rating curve: Q = {full['a']:.3g} * (h - {full['h0']:.2f})^{full['b']:.2f}")
    if min(abs(full["b"] - B_MIN), abs(full["b"] - B_MAX)) < 0.05:
        print(f"  Note: exponent b is at a bound of the allowed range ({B_MIN}-{B_MAX}), "
              "so the rating curve is poorly constrained at this reach.")
    print(f"  leave-one-out:       median abs. error {mh_loo['median_abs_pct_err']:.1f}%, "
          f"NSE {mh_loo['nse']:.2f}, bias {mh_loo['bias_pct']:+.1f}%  (n={mh_loo['n']})")
    print(f"  chronological split: median abs. error {mh_split['median_abs_pct_err']:.1f}%, "
          f"NSE {mh_split['nse']:.2f}, bias {mh_split['bias_pct']:+.1f}%  (n={mh_split['n']})")
    print("  (NSE: Nash-Sutcliffe efficiency; 1 = perfect, 0 = no better than the observed mean)")

    # Width as an alternative predictor. Fitting discharge from width with the
    # same method as for WSE failed (NSE of about -7e8) because width varies
    # little with discharge on this river, and inverting a power law with a
    # small exponent amplifies noise. Width is therefore assessed by
    # correlation first, and a fit is attempted only if a usable
    # relationship exists (r >= 0.3 and b >= 0.02)
    wm = m.dropna(subset=["width"])
    wm = wm[wm["width"] > 0]
    if len(wm) >= 8:
        w = wm["width"].values
        qw = wm["discharge_m3s"].values
        r_w = float(np.corrcoef(np.log(w), np.log(qw))[0, 1])
        r_h = float(np.corrcoef(h, q)[0, 1])
        bw = fit_width(w, qw)["b"]
        if r_w < 0.3 or bw < 0.02:
            rows.append({"result": "2b flow from width", "n": len(wm),
                         "correlation": r_w, "note": "no usable relationship"})
            print(f"  width vs discharge: correlation {r_w:.2f} "
                  f"(WSE: {r_h:.2f}); width is not a usable predictor of discharge here")
        else:
            loo_w = leave_one_out(w, qw, fit_width, predict_width)
            # 1/b is large when b is small, so cap predictions at 10 times the
            # maximum observed discharge to stop a single extreme value
            # dominating the metrics
            loo_w = np.clip(loo_w, 0, 10 * qw.max())
            mw = metrics(qw, loo_w)
            rows.append({"result": "2b flow from width, leave-one-out", **mw,
                         "correlation": r_w})
            print(f"  discharge from width (leave-one-out): median abs. error "
                  f"{mw['median_abs_pct_err']:.1f}%, NSE {mw['nse']:.2f} "
                  f"(correlation {r_w:.2f}; WSE {r_h:.2f})")
            m.loc[wm.index, "q_pred_width_loo"] = loo_w

    m["q_pred_height_loo"] = loo
    m.to_csv(f"outputs/validation_matched_{tag}.csv", index=False)
    pd.DataFrame(rows).assign(reach_id=reach).to_csv(f"outputs/validation_summary_{tag}.csv", index=False)

    # Figures: rating curve, then predicted vs gauged discharge
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(h[:k], q[:k], s=22, color="#1f6fb4", label="calibration dates (first 70%)")
    ax.scatter(h[k:], q[k:], s=26, facecolors="none", edgecolors="#d95f02",
               label="test dates (last 30%)")
    hh = np.linspace(h.min() - 0.3, h.max() + 0.3, 200)
    ax.plot(hh, predict_rating(full, hh), color="k", lw=1.2, label="rating curve (all dates)")
    ax.set_xlabel("SWOT water-surface elevation (m a.s.l.)")
    ax.set_ylabel("gauged discharge (m³/s)")
    ax.set_title(f"Reach {reach}: SWOT WSE vs gauged discharge", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/result2_rating_curve_{tag}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ok = np.isfinite(loo)
    ax.scatter(q[ok], loo[ok], s=22, color="#1f6fb4", label="from WSE")
    if "q_pred_width_loo" in m:
        ok_w = m["q_pred_width_loo"].notna()
        ax.scatter(m.loc[ok_w, "discharge_m3s"], m.loc[ok_w, "q_pred_width_loo"],
                   s=18, marker="x", color="#999999", label="from width")
    lim = [0, max(q.max(), np.nanmax(loo)) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("gauged discharge (m³/s)")
    ax.set_ylabel("predicted discharge, leave-one-out (m³/s)")
    ax.set_title(f"Reach {reach}: median abs. error {mh_loo['median_abs_pct_err']:.0f}% "
                 f"({mh_loo['n']} dates)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/result2_predicted_vs_gauged_{tag}.png", dpi=150)

    print(f"\nWrote outputs/validation_summary_{tag}.csv, "
          f"outputs/validation_matched_{tag}.csv, figures/result*_{tag}.png")
    print("Next: python3 scripts/14_power.py")


if __name__ == "__main__":
    main()
