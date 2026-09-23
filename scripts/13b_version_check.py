"""
13b_version_check.py - Compare SWOT Version C and Version D WSE on shared passes.

Usage:
    python3 scripts/13b_version_check.py

Version D reprocessed the earlier passes, so many passes exist in both
versions. For each reach, WSE from C and D is compared on the shared passes
(May to October, reach_q <= 1); differences of a few centimetres are expected.
A median difference above 0.25 m or an interquartile range above 0.5 m means
the ID most likely refers to a different location in Version D. This check is
how I found the SWORD renumbering: some reaches differed by 13-20 m.

version_check() is also used by 11b and by 12_clean_swot.py, which drops
Version D observations for every reach that fails the check.

Inputs:  data/swot_raw.csv (from 11_fetch_swot.py)
Outputs: outputs/version_check.csv
"""

import os
import sys

import numpy as np
import pandas as pd

FILL_BELOW = -1e10
MAX_MEDIAN_DIFF_M = 0.25   # a different location gives differences of metres, not centimetres
MAX_SPREAD_M = 0.50        # interquartile range of the differences


def version_check(raw):
    """Compare Version C and Version D WSE on passes present in both.

    Returns one row per reach. verdict is 'agree', 'DISAGREE', or
    "can't compare" if there are fewer than 3 shared passes.
    """
    r = raw.copy()
    r["reach_id"] = r["reach_id"].astype(str)
    for c in ("wse", "width", "reach_q"):
        if c in r:
            r[c] = pd.to_numeric(r[c], errors="coerce")
            r.loc[r[c] < FILL_BELOW, c] = np.nan
    if "width" not in r:
        r["width"] = np.nan
    r["time"] = pd.to_datetime(r["time_str"], utc=True, errors="coerce")
    # Restrict to May-October so that river ice does not bias WSE
    ok = r["wse"].notna() & r["time"].notna() & r["time"].dt.month.between(5, 10)
    if "reach_q" in r:
        ok &= r["reach_q"].fillna(0) <= 1   # exclude degraded or bad observations (reach_q 2 or 3)
    r = r[ok].copy()
    # Round to 15 min so the same pass matches between C and D despite small time differences
    r["t"] = r["time"].dt.round("15min")

    c = r[r.version == "C"][["reach_id", "t", "wse", "width"]]
    d = r[r.version == "D"][["reach_id", "t", "wse", "width"]]
    pairs = c.merge(d, on=["reach_id", "t"], suffixes=("_C", "_D"))
    pairs["dh"] = pairs["wse_D"] - pairs["wse_C"]
    pairs["dw"] = pairs["width_D"] - pairs["width_C"]

    rows = []
    for rid in sorted(raw["reach_id"].astype(str).unique()):
        p = pairs[pairs.reach_id == rid]
        base = {"reach_id": rid, "C_readings": int((c.reach_id == rid).sum()),
                "D_readings": int((d.reach_id == rid).sum()),
                "same_pass_pairs": int(len(p))}
        if len(p) < 3:
            rows.append({**base, "verdict": "can't compare"})
            continue
        med = float(p["dh"].median())
        spread = float(np.percentile(p["dh"], 75) - np.percentile(p["dh"], 25))
        agree = abs(med) <= MAX_MEDIAN_DIFF_M and spread <= MAX_SPREAD_M
        rows.append({**base,
                     "height_diff_D_minus_C_cm": round(100 * med, 1),
                     "height_diff_spread_cm": round(100 * spread, 1),
                     "width_diff_D_minus_C_m": round(float(p["dw"].median()), 1),
                     "verdict": "agree" if agree else "DISAGREE"})
    return pd.DataFrame(rows)


def main():
    if not os.path.exists("data/swot_raw.csv"):
        sys.exit("data/swot_raw.csv not found. Run 11_fetch_swot.py first.")
    raw = pd.read_csv("data/swot_raw.csv", dtype={"reach_id": str})
    if "version" not in raw:
        sys.exit("data/swot_raw.csv has no version column. Rerun 11_fetch_swot.py.")
    out = version_check(raw)
    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/version_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("Version D minus Version C, matched by pass and reach ID (May-October)")
    print(out.to_string(index=False))
    print("\n'spread' is the interquartile range (25th to 75th percentile) of the differences.")
    if (out["verdict"] == "DISAGREE").any():
        print("\nDISAGREE: the reach ID refers to a different location in Version D.")
        print("12_clean_swot.py excludes Version D observations for these reaches.")
        print("To recover their Version D data, run 11b_find_version_d_ids.py.")
    print("Wrote outputs/version_check.csv")


if __name__ == "__main__":
    main()
