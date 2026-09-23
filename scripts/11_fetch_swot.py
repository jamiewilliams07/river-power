"""
11_fetch_swot.py - Download SWOT reach time series from the Hydrocron API.

Usage:
    python3 scripts/11_fetch_swot.py [--config swot_config.yaml]

Requests the RiverSP reach product (WSE, width, slope, their uncertainties
and quality flags) for every reach in swot_config.yaml, one reach per request
(no Earthdata login required). Both product versions are downloaded:
  C: July 2023 to May 2025
  D: from May 2025, plus the earlier passes reprocessed
Version D uses SWORD v17b and renumbered some reaches, so version_d_ids in the
config maps an original (v16) reach ID to its Version D ID (found by 11b).
Records from both versions are stored under the original ID, unfiltered.

Inputs:  swot_config.yaml (reaches, start, end, version_d_ids)
Outputs: data/swot_raw.csv (all records), data/swot_fetch_log.csv (records
         returned per reach and version)
"""

import argparse
import io
import os
import sys
import time

import pandas as pd

try:
    import requests
    import yaml
except ImportError:
    sys.exit("Requires requests and pyyaml:  pip install requests pyyaml")

URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"

COLLECTIONS = {
    "C": "SWOT_L2_HR_RiverSP_2.0",
    "D": "SWOT_L2_HR_RiverSP_D",
}

# p_* fields are prior (static) reach attributes from SWORD, not SWOT measurements
FIELDS_FULL = [
    "reach_id", "time_str", "cycle_id", "pass_id", "river_name",
    "wse", "wse_u", "width", "width_u", "slope", "slope_u",
    "reach_q", "dark_frac", "ice_clim_f", "ice_dyn_f", "partial_f",
    "n_good_nod", "obs_frac_n", "xovr_cal_q",
    "p_width", "p_length", "p_dist_out", "p_lat", "p_lon",
]
# Reduced field list, used if a collection rejects one of the optional fields
FIELDS_CORE = ["reach_id", "time_str", "wse", "wse_u", "width",
               "slope", "reach_q", "p_dist_out", "p_length"]


def query(reach, collection, start, end, fields):
    """Send one Hydrocron request. Return (DataFrame or None, status message)."""
    params = {
        "feature": "Reach",
        "feature_id": reach,
        "start_time": f"{start}T00:00:00Z",
        "end_time": f"{end}T23:59:59Z",
        "output": "csv",
        "fields": ",".join(fields),
        "collection_name": collection,
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as e:
            msg = f"network error: {e}"
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            csv = r.json().get("results", {}).get("csv", "")
            if not csv.strip():
                return None, "empty"
            return pd.read_csv(io.StringIO(csv), dtype={"reach_id": str}), "ok"
        if r.status_code == 400:
            # Hydrocron also returns 400 when a reach has no data, not only for malformed requests
            return None, f"400: {r.text[:160]}"
        if r.status_code == 413:
            # Response too large; fetch_reach splits the request by year
            return None, "413 too large"
        msg = f"HTTP {r.status_code}: {r.text[:160]}"
        time.sleep(2 ** attempt)
    return None, msg


def fetch_reach(reach, collection, start, end):
    """Retrieve all records for one reach from one collection.

    Hydrocron returns HTTP 413 for long date ranges, so after a 413 the
    request is split into one request per calendar year. If Hydrocron rejects
    one of the optional fields, the request is repeated with FIELDS_CORE.
    """
    for fields in (FIELDS_FULL, FIELDS_CORE):
        df, msg = query(reach, collection, start, end, fields)
        if df is not None:
            return df, msg
        if msg.startswith("413"):
            parts = []
            for y in range(int(start[:4]), int(end[:4]) + 1):
                s = max(start, f"{y}-01-01")
                e = min(end, f"{y}-12-31")
                d, m = query(reach, collection, s, e, fields)
                if d is not None:
                    parts.append(d)
            if parts:
                return pd.concat(parts, ignore_index=True), "ok (split by year)"
            return None, msg
        field_problem = "field" in msg.lower() and msg.startswith("400")
        if not field_problem:
            return None, msg
    return None, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    reaches = [str(r) for r in (cfg.get("reaches") or [])]
    if not reaches:
        sys.exit("No reaches listed in swot_config.yaml. Identify them first with "
                 "scripts/gee_find_reaches.js (Google Earth Engine).")
    bad = [r for r in reaches if not (r.isdigit() and len(r) == 11)]
    if bad:
        sys.exit(f"These are not valid 11-digit reach IDs: {bad}. "
                 "Quote them in the YAML so they are parsed as strings.")

    start, end = str(cfg["start"]), str(cfg["end"])
    os.makedirs("data", exist_ok=True)

    # Original (v16) ID -> Version D ID, produced by 11b. Records from both
    # versions are stored under the original ID, so the downstream scripts are
    # unaffected by the renumbering.
    d_ids = {str(k): str(v) for k, v in (cfg.get("version_d_ids") or {}).items()}

    frames, log = [], []
    for reach in reaches:
        for ver, coll in COLLECTIONS.items():
            query_id = d_ids.get(reach, reach) if ver == "D" else reach
            df, msg = fetch_reach(query_id, coll, start, end)
            n = 0 if df is None else len(df)
            log.append({"reach_id": reach, "version": ver, "queried_id": query_id,
                        "rows": n, "message": msg})
            note = f"(as {query_id})" if query_id != reach else ""
            print(f"  {reach}  v{ver}: {n:4d} observations  {note} {'' if n else msg}")
            if df is not None:
                df["reach_id"] = reach          # always the original ID
                df["queried_id"] = query_id
                df["version"] = ver
                frames.append(df)
            time.sleep(0.5)   # brief pause between requests to limit server load

    pd.DataFrame(log).to_csv("data/swot_fetch_log.csv", index=False)
    if not frames:
        sys.exit("\nNo SWOT data returned for any reach. Check the reach IDs "
                 "and data/swot_fetch_log.csv.")

    raw = pd.concat(frames, ignore_index=True)
    # Drop the *_units columns that Hydrocron appends to the output
    raw = raw[[c for c in raw.columns if not c.endswith("_units")]]
    raw.to_csv("data/swot_raw.csv", index=False)

    summary = raw.groupby(["reach_id", "version"]).size().unstack(fill_value=0)
    print("\nObservations per reach and version (before cleaning):")
    print(summary.to_string())
    # No Version D records usually indicates that the reach was renumbered in SWORD v17b
    missing_d = [r for r in reaches
                 if "D" not in summary.columns or r not in summary.index
                 or summary.loc[r, "D"] == 0]
    if missing_d:
        print(f"\n{len(missing_d)} reach(es) have no Version D data: {missing_d}")
        print("These reaches were probably renumbered in Version D. Run:")
        print("    python3 scripts/11b_find_version_d_ids.py")
    print(f"\nWrote data/swot_raw.csv ({len(raw)} rows) and data/swot_fetch_log.csv")
    print("Next step: python3 scripts/12_clean_swot.py")


if __name__ == "__main__":
    main()
