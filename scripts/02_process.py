"""
Derive channel width and slope along the centreline from the exported imagery.

    python scripts/02_process.py --config config.yaml

Reads the GeoTIFFs in data/ and writes
  outputs/widths.csv            width along the centreline for each date and sensor
  outputs/sensor_agreement.csv  optical/SAR water mask agreement (IoU)
  outputs/slope.csv             DEM elevation and slope along the centreline
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riverlib import masks as M
from riverlib import width as W
from riverlib import slope as S

try:
    import rasterio
    import yaml
except ImportError:
    sys.exit("rasterio and pyyaml are required:  pip install rasterio pyyaml")


def read_bands(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def date_from_path(path):
    # data/S2_20240715.tif -> "20240715"
    stem = os.path.basename(path).split(".")[0]
    return stem.split("_")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pixel = float(cfg.get("scale_m", 10))
    min_size = int(cfg.get("min_blob_px", 20))
    min_branch = int(cfg.get("min_branch_px", 10))
    reach_spacing = float(cfg.get("reach_spacing_m", 200))

    rows = []
    mask_store = {}

    # Optical (Sentinel-2)
    for path in sorted(glob.glob("data/S2_*.tif")):
        date = date_from_path(path)
        arr, _, _ = read_bands(path)
        green, nir, swir = arr[0], arr[1], arr[2]

        idx = M.mndwi(green, swir)
        mask = M.threshold_index(idx, value=cfg.get("mndwi_threshold", 0.0))
        res = W.widths_from_mask(mask, pixel_size=pixel,
                                 min_size=min_size, min_branch=min_branch)
        if len(res["width"]) == 0:
            print(f"  {date} optical: no channel detected, skipping")
            continue
        ch, wd = W.resample_along_chainage(res["chainage"], res["width"],
                                           spacing=reach_spacing)
        for c, w in zip(ch, wd):
            rows.append({"date": date, "sensor": "S2",
                         "chainage_m": c, "width_m": w})
        mask_store.setdefault(date, {})["S2"] = mask
        print(f"  {date} optical: {len(ch)} reaches, "
              f"median width {np.median(wd):.0f} m")

    # SAR (Sentinel-1)
    for path in sorted(glob.glob("data/S1_*.tif")):
        date = date_from_path(path)
        arr, _, _ = read_bands(path)
        vv = arr[0]
        # The Earth Engine S1_GRD collection is already in dB. A non-negative
        # median indicates linear power values, which are converted
        vv_db = vv if np.nanmedian(vv) < 0 else M.to_db(vv)

        mask, thr = M.sar_water_mask(
            vv_db, speckle_size=int(cfg.get("speckle_window", 5)),
            manual_threshold=cfg.get("sar_threshold_db"),
        )
        res = W.widths_from_mask(mask, pixel_size=pixel,
                                 min_size=min_size, min_branch=min_branch)
        if len(res["width"]) == 0:
            print(f"  {date} SAR: no channel detected, skipping")
            continue
        ch, wd = W.resample_along_chainage(res["chainage"], res["width"],
                                           spacing=reach_spacing)
        for c, w in zip(ch, wd):
            rows.append({"date": date, "sensor": "S1",
                         "chainage_m": c, "width_m": w})
        mask_store.setdefault(date, {})["S1"] = mask
        print(f"  {date} SAR: {len(ch)} reaches, "
              f"median width {np.median(wd):.0f} m, Otsu threshold {thr:.1f} dB")

    if not rows:
        sys.exit("No usable imagery in data/. Run 01_fetch_gee.py first.")

    widths = pd.DataFrame(rows)
    widths.to_csv("outputs/widths.csv", index=False)
    print(f"\nwrote outputs/widths.csv ({len(widths)} rows)")

    # Compare optical and SAR masks on dates with both acquisitions
    agree = []
    for date, d in mask_store.items():
        if "S2" in d and "S1" in d and d["S2"].shape == d["S1"].shape:
            a = M.agreement(d["S2"], d["S1"])
            a["date"] = date
            agree.append(a)
    if agree:
        ag = pd.DataFrame(agree)
        ag.to_csv("outputs/sensor_agreement.csv", index=False)
        print(f"wrote outputs/sensor_agreement.csv "
              f"(median IoU {ag['iou'].median():.3f})")
        if ag["iou"].median() < 0.7:
            print("  Median IoU below 0.7: the sensors disagree. "
                  "Treat the difference as the width uncertainty.")
    else:
        print("No same-day optical/SAR pairs; skipping sensor agreement.")

    # Slope from the DEM
    dem_paths = glob.glob("data/DEM*.tif")
    if not dem_paths:
        print("\nNo DEM found in data/; skipping the slope step.")
        return

    dem, _, _ = read_bands(dem_paths[0])
    dem = dem[0].astype(float)

    # Use the centreline from the date with the largest median width, as that
    # mask is the least likely to be fragmented
    ref_date = widths.groupby("date")["width_m"].median().idxmax()
    ref = widths[widths.date == ref_date]
    print(f"\nUsing the {ref_date} centreline for the slope profile.")

    ref_path = (sorted(glob.glob(f"data/S2_{ref_date}*.tif"))
                or sorted(glob.glob(f"data/S1_{ref_date}*.tif")))
    arr, _, _ = read_bands(ref_path[0])
    if ref_path[0].split("/")[-1].startswith("S2"):
        mask = M.threshold_index(M.mndwi(arr[0], arr[2]))
    else:
        mask, _ = M.sar_water_mask(arr[0] if np.nanmedian(arr[0]) < 0
                                   else M.to_db(arr[0]))
    res = W.widths_from_mask(mask, pixel_size=pixel,
                             min_size=min_size, min_branch=min_branch)

    # DEM pixels are 30 m against 10 m image pixels (by default), so the node
    # indices are rescaled.
    # TODO: this assumes the DEM and image exports share a CRS and grid
    # origin. Not verified, because the script was never run on real imagery
    dem_scale = float(cfg.get("dem_scale_m", 30)) / pixel
    nodes_dem = np.clip(
        np.rint(res["nodes"] / dem_scale).astype(int),
        0, np.array(dem.shape) - 1,
    )
    z = S.sample_profile(dem, nodes_dem, nodata=cfg.get("dem_nodata"))
    z = S.monotonic_fill(z)
    window = float(cfg.get("slope_window_m", 1000))
    sl = S.windowed_slope(res["chainage"], z, window=window)

    pd.DataFrame({
        "chainage_m": res["chainage"],
        "elevation_m": z,
        "slope": sl,
    }).to_csv("outputs/slope.csv", index=False)

    print(f"wrote outputs/slope.csv "
          f"(median slope {100*np.nanmedian(sl):.3f}%, "
          f"window {window:.0f} m)")
    print(f"  reach drop over full profile: "
          f"{S.reach_drop(res['chainage'], z, res['chainage'].min(), res['chainage'].max()):.1f} m")


if __name__ == "__main__":
    main()
