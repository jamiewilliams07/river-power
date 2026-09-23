"""
12_clean_swot.py - Apply quality filters to the raw SWOT reach observations.

Usage:
    python3 scripts/12_clean_swot.py [--config swot_config.yaml] [--raw data/swot_raw.csv]

Filters are applied in sequence and each removed observation is attributed to
the first filter that rejects it, so the report shows the effect of each
filter. The sequence covers missing time or WSE, Version D IDs that refer to a
different location, duplicate passes in Versions C and D (one copy kept),
non-river reaches, ice-affected months, quality flags (reach_q, partial_f,
dark_frac, xovr_cal_q, ice_clim_f, ice_dyn_f) and WSE uncertainty, followed by
a per-reach median absolute deviation (MAD) outlier test on WSE. Thresholds
are set under `quality:` in swot_config.yaml.

Inputs:  swot_config.yaml, data/swot_raw.csv (from 11_fetch_swot.py)
Outputs: data/swot_clean.csv               observations that pass every filter
         outputs/swot_cleaning_report.csv  observations removed by each filter, per reach
         figures/swot_cleaning.png         timeline of retained and removed observations
"""

import argparse
import importlib.util
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
    sys.exit("Requires pyyaml:  pip install pyyaml")

FILL_BELOW = -1e10   # SWOT fill value is -999999999999

# Load version_check() from 13b_version_check.py; the file name begins with a
# digit, so a regular import is not possible
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "version_check_mod", os.path.join(_here, "13b_version_check.py"))
_vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vc)


def robust_z(x):
    """Modified z-score based on the median and the median absolute deviation (MAD).

    The median and MAD are far less sensitive to the outliers under test than
    the mean and standard deviation. The factor 0.6745 makes the score
    comparable to a standard z-score for normally distributed data. Returns
    zeros if the MAD is zero or undefined.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def clean(raw, cfg):
    q = cfg["quality"]
    months = set(cfg["open_water_months"])
    df = raw.copy()
    df["reach_id"] = df["reach_id"].astype(str)

    # Convert every non-text column to numeric and replace fill values with NaN
    num_cols = [c for c in df.columns
                if c not in ("reach_id", "time_str", "river_name", "version", "queried_id")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < FILL_BELOW, c] = np.nan
    df["time"] = pd.to_datetime(df["time_str"], utc=True, errors="coerce")

    # Version D includes the reprocessed earlier passes, so the Version C copy
    # of a pass present in both versions is dropped to avoid double counting.
    # The exception is a reach that 13b flags as DISAGREE: its Version D ID
    # refers to a different location, so Version D is dropped and C is kept.
    no = pd.Series(False, index=df.index)
    wrong_place, dup = no, no
    if "version" in df.columns:
        vc = _vc.version_check(raw)
        mismatch = set(vc.loc[vc["verdict"] == "DISAGREE", "reach_id"])
        if mismatch:
            print(f"Version D ID refers to a different location for "
                  f"{sorted(mismatch)}; their Version D observations are excluded.")
        wrong_place = (df["version"] == "D") & df["reach_id"].isin(mismatch)
        # A pass is identified by reach ID and overpass time rounded to 15 min
        key = df["reach_id"] + "|" + df["time"].dt.round("15min").astype(str)
        d_keys = set(key[(df["version"] == "D") & ~wrong_place & df["time"].notna()])
        dup = (df["version"] == "C") & key.isin(d_keys)

    # A missing flag (NaN) is treated as passing
    def flag_ok(col, maxval):
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        return df[col].isna() | (df[col] <= maxval)

    # Each filter is (name, function returning True for observations to keep).
    # The names are the column names in the cleaning report; do not rename them.
    filters = [
        ("no time or height", lambda: df["time"].notna() & df["wse"].notna()),
        ("v.D ID is a different place", lambda: ~wrong_place),
        ("same pass in C and D", lambda: ~dup),
        # The last digit of a SWORD reach ID is the type (1 = river; others are lakes, dams, etc.)
        ("river reaches only", lambda: (df["reach_id"].str[-1] == "1")
            if q.get("river_reaches_only", True) else pd.Series(True, index=df.index)),
        ("frozen months", lambda: df["time"].dt.month.isin(months)),
        ("reach_q (NASA quality)", lambda: flag_ok("reach_q", q["max_reach_q"])),
        ("partly observed", lambda: flag_ok("partial_f", 0)
            if q.get("require_partial_f_zero", True) else pd.Series(True, index=df.index)),
        ("too dark", lambda: flag_ok("dark_frac", q["max_dark_frac"])),
        ("crossover calibration", lambda: flag_ok("xovr_cal_q", q["max_xovr_cal_q"])),
        ("ice (climatology)", lambda: flag_ok("ice_clim_f", q["max_ice_clim_f"])),
        ("ice (observed)", lambda: flag_ok("ice_dyn_f", q["max_ice_dyn_f"])),
        ("height uncertainty", lambda: flag_ok("wse_u", q["max_wse_u_m"])),
    ]

    # Each removed observation is attributed only to the first filter that rejects it
    df["removed_by"] = ""
    alive = pd.Series(True, index=df.index)
    for name, fn in filters:
        keep = fn().fillna(False).astype(bool)
        newly = alive & ~keep
        df.loc[newly, "removed_by"] = name
        alive &= keep

    # The MAD outlier test runs last, per reach, on the observations that remain
    z_lim = q.get("outlier_mad_z", 4.0)
    for rid, idx in df[alive].groupby("reach_id").groups.items():
        z = robust_z(df.loc[idx, "wse"].values)
        out = pd.Index(idx)[np.abs(z) > z_lim]
        df.loc[out, "removed_by"] = "height outlier"
        alive.loc[out] = False

    df["kept"] = alive
    # Report columns follow the order in which the filters were applied
    order = ["kept"] + [n for n, _ in filters] + ["height outlier"]
    report = (df.assign(outcome=np.where(df["kept"], "kept", df["removed_by"]))
                .groupby(["reach_id", "outcome"]).size()
                .unstack(fill_value=0))
    report = report.reindex(columns=[c for c in order if c in report.columns],
                            fill_value=0)
    report.insert(0, "total", report.sum(axis=1))
    return df, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--raw", default="data/swot_raw.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if not os.path.exists(args.raw):
        sys.exit(f"{args.raw} not found. Run 11_fetch_swot.py first.")
    raw = pd.read_csv(args.raw, dtype={"reach_id": str})

    df, report = clean(raw, cfg)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    kept = df[df["kept"]].drop(columns=["kept", "removed_by"])
    kept = kept.sort_values(["reach_id", "time"])
    kept.to_csv("data/swot_clean.csv", index=False)
    report.to_csv("outputs/swot_cleaning_report.csv")

    pd.set_option("display.width", 200)
    print("Cleaning report (observations per reach):")
    print(report.to_string())
    tot = report.sum()
    print(f"\nTotal: {int(tot['total'])} observations -> {int(tot.get('kept', 0))} kept")

    gauge_reach = cfg.get("gauge_reach")
    if gauge_reach:
        n = int(report.loc[str(gauge_reach), "kept"]) if str(gauge_reach) in report.index else 0
        # Approximate minimum sample sizes for the comparison against the gauge
        verdict = ("good" if n >= 25 else "usable, but limited" if n >= 15
                   else "TOO FEW")
        print(f"\nGauge reach {gauge_reach}: {n} clean observations -> {verdict}")
    else:
        print("\nSet gauge_reach in swot_config.yaml before running 13_validate.py.")

    # Timeline plot: one row per reach, grey = removed, blue = kept. Reaches are
    # sorted by descending distance from outlet (p_dist_out) when available,
    # which places the most upstream reach at the bottom.
    reaches = sorted(df["reach_id"].unique(),
                     key=lambda r: -np.nanmedian(df.loc[df.reach_id == r, "p_dist_out"])
                     if "p_dist_out" in df else r)
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(reaches) + 1.5))
    for i, r in enumerate(reaches):
        sub = df[(df.reach_id == r) & df["time"].notna()]
        ax.scatter(sub.loc[~sub.kept, "time"], [i] * (~sub.kept).sum(),
                   s=10, color="#bbbbbb", label="removed" if i == 0 else None)
        ax.scatter(sub.loc[sub.kept, "time"], [i] * sub.kept.sum(),
                   s=14, color="#1f6fb4", label="kept" if i == 0 else None)
    ax.set_yticks(range(len(reaches)))
    ax.set_yticklabels(reaches, fontsize=7)
    ax.set_xlabel("date")
    ax.set_title("SWOT observations per reach (upstream at bottom)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/swot_cleaning.png", dpi=150)

    print("\nWrote data/swot_clean.csv, outputs/swot_cleaning_report.csv, "
          "figures/swot_cleaning.png")
    print("Next step: python3 scripts/13_validate.py")


if __name__ == "__main__":
    main()
