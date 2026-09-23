"""
Retrieve Sentinel-2, Sentinel-1 and DEM imagery from Google Earth Engine.

    earthengine authenticate
    python scripts/01_fetch_gee.py --config config.yaml

The scenes are too large for direct download, so the script queues Earth
Engine export tasks to Google Drive; the exported GeoTIFFs are then
copied into data/ manually.

Sentinel-1 is restricted to a single relative orbit so that backscatter
is comparable between dates. All acquisition dates are printed at the
end, because gauged discharge is required for the same days.
"""

import argparse
import json
import sys

try:
    import ee
except ImportError:
    sys.exit("earthengine-api not installed:  pip install earthengine-api")

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not installed:  pip install pyyaml")


def mask_s2_clouds(img):
    """Mask cloud, cirrus and cloud shadow using the Scene Classification Layer."""
    scl = img.select("SCL")
    # SCL classes: 3 cloud shadow, 8-9 cloud, 10 cirrus
    bad = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)))
    return img.updateMask(bad.Not()).divide(10000).copyProperties(
        img, ["system:time_start"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--project", default=None,
                    help="Google Cloud project ID for Earth Engine (overrides config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    project = args.project or cfg.get("ee_project")
    ee.Initialize(project=project)

    aoi = ee.Geometry.Polygon(cfg["aoi"]["coordinates"])
    start, end = cfg["date_range"]["start"], cfg["date_range"]["end"]
    drive_folder = cfg.get("drive_folder", "river_power")
    scale = cfg.get("scale_m", 10)

    # ---- Sentinel-2 (optical) ----
    s2 =(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",
                               cfg.get("max_cloud_pct", 20)))
          .map(mask_s2_clouds))

    n_s2 = s2.size().getInfo()
    print(f"Sentinel-2 scenes matching: {n_s2}")
    if n_s2 == 0:
        print("  No optical scenes passed the cloud filter. Raise max_cloud_pct, "
              "widen the date range, or rely on Sentinel-1 alone.")

    s2_dates = s2.aggregate_array("system:time_start").getInfo()

    # B3 and B11 for MNDWI, B8 for NDWI. B11 has a native resolution of
    # 20 m, so exporting it at 10 m adds no spatial detail
    for i, t in enumerate(s2_dates):
        img = ee.Image(s2.toList(n_s2).get(i)).select(["B3", "B8", "B11"])
        date = ee.Date(t).format("YYYYMMdd").getInfo()
        task = ee.batch.Export.image.toDrive(
            image=img.clip(aoi),
            description=f"S2_{date}",
            folder=drive_folder,
            fileNamePrefix=f"S2_{date}",
            region=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        task.start()
        print(f"  queued S2_{date}")

    # ---- Sentinel-1 (SAR) ----
    s1 =(ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))

    orbit_pass = cfg.get("s1_orbit_pass")
    rel_orbit = cfg.get("s1_relative_orbit")

    if orbit_pass:
        s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    if rel_orbit is not None:
        s1 = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", int(rel_orbit)))
    else:
        # No relative orbit selected: list the available orbits and stop.
        # The Sentinel-2 exports above have already been queued.
        orbits = s1.aggregate_array("relativeOrbitNumber_start").getInfo()
        if orbits:
            counts = {o: orbits.count(o) for o in set(orbits)}
            print(f"\nAvailable relative orbits (count): {counts}")
            print("Set s1_relative_orbit in config.yaml to the most frequent "
                  "orbit and re-run.")
            return

    n_s1 = s1.size().getInfo()
    print(f"\nSentinel-1 scenes matching (single orbit): {n_s1}")
    s1_dates = s1.aggregate_array("system:time_start").getInfo()

    for i, t in enumerate(s1_dates):
        img = ee.Image(s1.toList(n_s1).get(i)).select(["VV"])
        date = ee.Date(t).format("YYYYMMdd").getInfo()
        task = ee.batch.Export.image.toDrive(
            image=img.clip(aoi),
            description=f"S1_{date}",
            folder=drive_folder,
            fileNamePrefix=f"S1_{date}",
            region=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        task.start()
        print(f"  queued S1_{date}")

    # ---- DEM ----
    dem_id = cfg.get("dem", "JAXA/ALOS/AW3D30/V3_2")
    dem = ee.Image(dem_id).select(cfg.get("dem_band", "DSM"))
    task = ee.batch.Export.image.toDrive(
        image=dem.clip(aoi),
        description="DEM",
        folder=drive_folder,
        fileNamePrefix="DEM",
        region=aoi,
        scale=cfg.get("dem_scale_m", 30),
        maxPixels=1e9,
    )
    task.start()
    print(f"\n  queued DEM ({dem_id})")

    manifest = {
        "s2_dates": [ee.Date(t).format("YYYY-MM-dd").getInfo() for t in s2_dates],
        "s1_dates": [ee.Date(t).format("YYYY-MM-dd").getInfo() for t in s1_dates],
        "dem": dem_id,
        "scale_m": scale,
    }
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nExports queued. Monitor at https://code.earthengine.google.com/tasks")
    print("Download into data/ when complete.")
    print("\nNext: obtain gauged discharge for these dates:")
    for d in manifest["s2_dates"] + manifest["s1_dates"]:
        print(f"  {d}")


if __name__ == "__main__":
    main()
