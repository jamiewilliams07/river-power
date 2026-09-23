"""
11b_find_version_d_ids.py - Match renumbered reaches to their SWOT Version D IDs by location.

Usage:
    python3 scripts/11b_find_version_d_ids.py [--max-num N] [--basins-either-side N]
                                              [--max-km KM]

SWOT Version D uses SWORD v17b, whereas the reach IDs in the config come from
SWORD v16 in Earth Engine. Some reaches were renumbered in v17b and some IDs
were reused for different locations, which I detected when the same pass gave
WSE values 13-20 m apart in Versions C and D. Distance from outlet differs by
about 560 km between the two SWORD versions, so reaches are matched by
coordinates rather than by distance.

Procedure:
  1. Select reaches with no Version D data or a DISAGREE verdict from 13b.
  2. Locate each reach from its Version C observations (median p_lat, p_lon).
  3. Query Hydrocron for every candidate reach ID in the surrounding basins
     (about 500 IDs with the defaults, roughly 5 minutes).
  4. Match each reach to the nearest Version D reach within --max-km.

Inputs:  swot_config.yaml, data/swot_raw.csv (from 11_fetch_swot.py)
Outputs: none written. Prints a version_d_ids block to append to
         swot_config.yaml; then rerun 11_fetch_swot.py and 12_clean_swot.py.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    sys.exit("Requires pyyaml:  pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))


# Script file names begin with a digit, so they cannot be loaded with a regular import
def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch", "11_fetch_swot.py")
vcheck = _load("vcheck", "13b_version_check.py")

PROBE_FIELDS = ["reach_id", "time_str", "p_lat", "p_lon"]   # location fields only
FILL_BELOW = -1e10   # SWOT fill value is -999999999999


def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (haversine formula)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def centre(df):
    """Median latitude and longitude of the observations, excluding fill values."""
    lat =pd.to_numeric(df["p_lat"], errors="coerce")
    lon = pd.to_numeric(df["p_lon"], errors="coerce")
    ok = (lat > FILL_BELOW) & (lon > FILL_BELOW)
    if not ok.any():
        return np.nan, np.nan
    return float(lat[ok].median()), float(lon[ok].median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--max-num", type=int, default=99,
                    help="highest 4-digit reach number to query in each basin")
    ap.add_argument("--basins-either-side", type=int, default=2)
    ap.add_argument("--max-km", type=float, default=4.0,
                    help="maximum centre-to-centre distance (km) accepted as a match")
    ap.add_argument("--probe-start", default="2025-06-01")
    ap.add_argument("--probe-end", default="2025-09-30")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    raw = pd.read_csv("data/swot_raw.csv", dtype={"reach_id": str})
    if not {"p_lat", "p_lon"} <= set(raw.columns):
        sys.exit("data/swot_raw.csv has no p_lat/p_lon columns, so reaches cannot be "
                 "located.")

    vc = vcheck.version_check(raw)
    no_d = set(vc.loc[vc["D_readings"] == 0, "reach_id"])
    wrong = set(vc.loc[vc["verdict"] == "DISAGREE", "reach_id"])
    targets = sorted(no_d | wrong)
    if not targets:
        print("Every reach has Version D data that agree with Version C. No action required.")
        return
    print(f"Reaches with no Version D data:             {sorted(no_d)}")
    print(f"Reaches with Version D at another location: {sorted(wrong)}")

    where = {}
    for r in targets:
        lat, lon = centre(raw[(raw.reach_id == r) & (raw.version == "C")])
        if np.isfinite(lat):
            where[r] = (lat, lon)
        else:
            print(f"  {r}: no Version C location; skipped")

    # The first 6 digits of a SWORD reach ID are the basin code. Neighbouring
    # basins are also searched in case v17b assigned the reach to another basin.
    basins = set()
    for r in where:
        b = int(r[:6])
        for k in range(-args.basins_either_side, args.basins_either_side + 1):
            basins.add(f"{b + k:06d}")
    # Reach ID = 6-digit basin code + 4-digit reach number + type digit (1 = river)
    candidates = [f"{b}{n:04d}1" for b in sorted(basins) for n in range(args.max_num + 1)]
    print(f"\nQuerying {len(candidates)} candidate reach IDs in basins "
          f"{', '.join(sorted(basins))} (several minutes)...")

    found = []
    for i, cid in enumerate(candidates, 1):
        df, _ = fetch.query(cid, fetch.COLLECTIONS["D"], args.probe_start,
                            args.probe_end, PROBE_FIELDS)
        if df is not None and len(df):
            lat, lon = centre(df)
            if np.isfinite(lat):
                found.append({"new_id": cid, "lat": lat, "lon": lon})
        if i % 100 == 0:
            print(f"  ...{i}/{len(candidates)} queried, {len(found)} reaches found")
        time.sleep(0.15)   # limit the request rate over several hundred queries
    if not found:
        print("\nNo Version D reaches found in these basins. The affected reaches "
              "will use Version C only.")
        return
    found = pd.DataFrame(found)
    print(f"  {len(found)} Version D reaches found.")

    print("\nMatches by location (original reach -> nearest Version D reach):")
    mapping = {}
    for r, (lat, lon) in where.items():
        d = km_between(lat, lon, found["lat"].values, found["lon"].values)
        i = int(np.argmin(d))
        new_id, dist = found.loc[i, "new_id"], float(d[i])
        # The nearest Version D reach has the same ID, so renumbering does not
        # explain the discrepancy
        if new_id == r:
            print(f"  {r} -> itself ({dist:.1f} km). Same location, so the disagreement "
                  "has another cause; Version D remains excluded.")
            continue
        ok = dist <= args.max_km
        flag = "" if ok else f"   <- NOT used (centres more than {args.max_km:g} km apart)"
        print(f"  {r} -> {new_id}   centres {dist:.1f} km apart{flag}")
        if ok:
            mapping[r] = new_id

    # Version D occasionally merged two v16 reaches into one
    dupes = pd.Series(mapping, dtype=str).value_counts()
    for new_id, n in dupes[dupes > 1].items():
        print(f"  Note: {n} original reaches map to Version D reach {new_id} (merged in "
              "Version D). Count that river length only once in the power table.")

    # The test suite extracts the output between "version_d_ids:" and
    # "Then rerun", so both of those lines must remain unchanged
    if mapping:
        print("\nAppend the following block to the end of swot_config.yaml:\n")
        print("version_d_ids:")
        for k, v in mapping.items():
            print(f'  "{k}": "{v}"')
        print("\nThen rerun:  python3 scripts/11_fetch_swot.py  and  "
              "python3 scripts/12_clean_swot.py")
    else:
        print("\nNo usable matches. These reaches will use Version C only.")


if __name__ == "__main__":
    main()
