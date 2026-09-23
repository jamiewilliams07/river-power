"""
10_fetch_gauge.py - Download daily discharge and water level for the WSC gauges.

Usage:
    python3 scripts/10_fetch_gauge.py [--config swot_config.yaml]
                                      [--extra 07DA001=data/07DA001_realtime.csv]

Retrieves the approved daily-mean record (HYDAT) for each Water Survey of
Canada gauge listed in swot_config.yaml from the ECCC OGC API
(api.weather.gc.ca; no account required). The approved record currently ends
on 2025-12-31, so 2026 values come from the provisional real-time CSV, which
is downloaded manually from the Water Office and merged with --extra. The
script prints download instructions when the record ends more than 60 days
before the current date.

Inputs:  swot_config.yaml (gauges, start); optional Water Office real-time CSVs
Outputs: data/gauge_<STATION>.csv (date, discharge_m3s, level_m,
         discharge_flag, level_flag, source)
"""

import argparse
import os
import sys
import time

import pandas as pd

try:
    import requests
    import yaml
except ImportError:
    sys.exit("Requires requests and pyyaml:  pip install requests pyyaml")

API = "https://api.weather.gc.ca/collections"
PAGE = 10000   # maximum number of features returned per API request


# Retry with exponential backoff, then exit if every attempt fails
def get_json(url, params, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"    Network error: {e}")
        time.sleep(2 ** attempt)
    sys.exit(f"Request to {url} failed after repeated attempts. Check the network connection.")


def station_info(stn):
    js = get_json(f"{API}/hydrometric-stations/items",
                  {"STATION_NUMBER": stn, "f": "json"})
    feats = js.get("features", [])
    if not feats:
        sys.exit(f"Station {stn} not found. Verify the station number on wateroffice.ec.gc.ca.")
    p = feats[0]["properties"]
    lon, lat = feats[0]["geometry"]["coordinates"][:2]
    return {
        "name": p.get("STATION_NAME"),
        "lat": lat, "lon": lon,
        "drainage_km2": p.get("DRAINAGE_AREA_GROSS"),
        "status": p.get("STATUS_EN"),
    }


def daily_means(stn):
    """Download the full approved daily-mean record in 20-year windows.

    A 20-year window contains at most about 7300 days, which is below the
    10,000-feature limit per request, so no pagination is required.
    """
    rows = []
    this_year = pd.Timestamp.today().year
    for y0 in range(1900, this_year + 1, 20):
        window = f"{y0}-01-01/{min(y0 + 19, this_year)}-12-31"
        js = get_json(f"{API}/hydrometric-daily-mean/items",
                      {"STATION_NUMBER": stn, "f": "json",
                       "limit": PAGE, "datetime": window})
        feats = js.get("features", [])
        if len(feats) >= PAGE:
            print(f"    Warning: window {window} reached the row limit; some days may be missing.")
        for f in feats:
            p = f["properties"]
            rows.append({
                "date": p.get("DATE"),
                "discharge_m3s": p.get("DISCHARGE"),
                "level_m": p.get("LEVEL"),
                "discharge_flag": p.get("DISCHARGE_SYMBOL_EN"),
                "level_flag": p.get("LEVEL_SYMBOL_EN"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df["source"] = "HYDAT (approved)"
    return df


def read_wateroffice_realtime(path):
    """Read a Water Office real-time CSV and aggregate it to daily means.

    Parameter 46 is water level (m) and parameter 47 is discharge (m^3/s).
    Column names are bilingual (English/French) and vary between downloads,
    so each column is located by a case-insensitive substring match.
    """
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.readlines()
    # The data table may be preceded by metadata lines, so locate the header row
    start = next(i for i, ln in enumerate(lines)
                 if "Date" in ln and "Param" in ln)
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines[start:])))
    col = lambda key: next(c for c in df.columns if key.lower() in c.lower())
    dcol, pcol, vcol = col("Date"), col("Param"), col("Value")
    # The timestamps carry a UTC offset. Parsing them as timezone-aware values
    # shifted evening readings into the next (UTC) day. The first 10 characters
    # of the timestamp string are already the local date, so only those are used.
    df["date"] = pd.to_datetime(df[dcol].astype(str).str[:10], errors="coerce").dt.date
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    q = df[df[pcol] == 47].groupby("date")[vcol].mean().rename("discharge_m3s")
    h = df[df[pcol] == 46].groupby("date")[vcol].mean().rename("level_m")
    out = pd.concat([q, h], axis=1).reset_index()
    out["discharge_flag"] = None
    out["level_flag"] = None
    out["source"] = "Water Office real-time (provisional)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="swot_config.yaml")
    ap.add_argument("--extra", action="append", default=[],
                    help="provisional real-time CSV for one station, as STATION=path (repeatable)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs("data", exist_ok=True)

    # Parse "07DA001=file.csv" into {"07DA001": "file.csv"}
    extras = dict(e.split("=", 1) for e in args.extra)
    swot_start = pd.to_datetime(cfg["start"]).date()

    for role, stn in cfg["gauges"].items():
        info = station_info(stn)
        print(f"\n{role}: {stn}  {info['name']}")
        print(f"  Location: lat {info['lat']:.4f}, lon {info['lon']:.4f}   "
              f"<- set GAUGE_LAT and GAUGE_LON in gee_find_reaches.js")
        print(f"  Drainage area: {info['drainage_km2']} km2   Status: {info['status']}")

        df = daily_means(stn)
        if df.empty:
            print("  No approved daily data returned.")
        else:
            first, last = df["date"].min(), df["date"].max()
            print(f"  Approved record: {first} to {last}  ({len(df)} days)")
            print(f"  Days with discharge: {df['discharge_m3s'].notna().sum()}, "
                  f"with water level: {df['level_m'].notna().sum()}")

        if stn in extras:
            rt = read_wateroffice_realtime(extras[stn])
            # Append only days after the approved record ends, so the approved
            # value takes precedence on any date present in both sources
            if not df.empty:
                rt = rt[rt["date"] > df["date"].max()]
            print(f"  Merged {len(rt)} provisional days from {extras[stn]}")
            df = pd.concat([df, rt], ignore_index=True)

        out = f"data/gauge_{stn}.csv"
        df.sort_values("date").to_csv(out, index=False)
        print(f"  Wrote {out}")

        in_swot = df[pd.to_datetime(df["date"]).dt.date >= swot_start] if len(df) else df
        print(f"  Days within the SWOT period (from {swot_start}): {len(in_swot)}")
        # If the record ends more than 60 days ago, explain how to add provisional data
        if len(df) and df["date"].max() < pd.Timestamp.today().date() - pd.Timedelta(days=60):
            print("  Note: the record ends well before the current date. To add recent days:")
            print(f"    wateroffice.ec.gc.ca -> search {stn} -> Real-Time Data ->")
            print("    longest period -> Download CSV, then rerun with")
            print(f"    --extra {stn}=data/{stn}_realtime.csv")


if __name__ == "__main__":
    main()
