# Discontinued method: channel width from Sentinel-2 and Sentinel-1

## Summary

This was the project's original approach, applied to the Yubetsu River in
Hokkaido (near the farm where I worked over the summer). It estimated
discharge from satellite-derived channel width and combined it with head
from a DEM to estimate hydropower potential. Synthetic tests showed that at
10 m pixel resolution the method requires a channel wider than the Yubetsu,
so the project moved to SWOT observations of the Athabasca River (see the
[README](../README.md)). The code is retained because the synthetic tests
and the minimum-width result remain useful.

Files belonging to this method:

- `riverlib/`
- `scripts/01_fetch_gee.py`, `02_process.py`, `03_analyse.py`
- `scripts/gee_separability.js`
- `config.yaml`
- `tests/test_synthetic.py`

The Python scripts (01 to 03) were never run on real imagery. The Earth
Engine screening script `gee_separability.js` was.

## Method

1. Retrieve Sentinel-2 (optical), Sentinel-1 (SAR) and a DEM (AW3D30) for
   the reach around a discharge gauge.
2. Derive water masks: MNDWI with a fixed threshold for Sentinel-2, and Otsu
   thresholding of speckle-filtered VV backscatter for Sentinel-1.
3. Extract the centreline by skeletonization; width = 2 × the Euclidean
   distance transform at each centreline pixel.
4. Estimate slope by least-squares fitting of DEM elevation against chainage
   over a moving window of 1 km or more.
5. Fit the at-a-station hydraulic geometry relation `w = aQ^b` (Leopold and
   Maddock 1953) to gauged discharge, invert it to obtain Q from width, and
   compute power as `P = ρgQHη`.
6. Propagate input uncertainties by Monte Carlo simulation, and attribute
   the variance of log(power) to each input by variance decomposition.

## Findings

All results are from `tests/test_synthetic.py` (23/23 checks pass), which
applies the pipeline to synthetic channels with known geometry.

- **Width extraction is accurate on clean masks.** Bias was 0.00 px on
  straight channels 40-160 m wide, and 82.5 m was recovered for a sinuous
  80 m channel.
- **Point-to-point DEM slope is dominated by noise.** With 5 m vertical
  error on 30 m pixels, the standard deviation of the point-to-point slope
  was 23.6% for a true slope of 0.2%. A least-squares fit over a moving
  window reduces the slope error to 17% with a 1 km window and 1% with a
  3 km window.
- **Inverting the width-discharge relation amplifies width error.** A
  relative width error e produces a discharge error of roughly e/b. With
  `b = 0.15`, `1/b` = 6.7, so a 10% width error gives an 89% discharge
  error.
- **A minimum channel width is required.** For the fit to recover b, the
  width change over the seasonal flow range must exceed 3× the width noise
  (about 10 m at 10 m pixels, i.e. one pixel per bank). The corresponding
  minimum low-flow widths are:

| seasonal flow range | minimum low-flow width |
|---|---|
| 5× | 110 m |
| 10× | 73 m |
| 30× | 45 m |

## Reason for discontinuing

Even for a 30× seasonal flow range, the channel must be at least about
45 m wide at low flow, and the Yubetsu is narrower than this.
`gee_separability.js` gave a consistent result: at the Yubetsu points,
neither MNDWI nor VV backscatter separated water from land reliably.

## Limitations

These would apply even to a sufficiently wide river:

- The hydraulic geometry fit is valid only for reaches with geometry similar
  to the section at which it was fitted.
- Head is the gross DEM drop; no head losses are subtracted.
- The effective optical width resolution is 20 m, not 10 m, because MNDWI
  uses band B11.
- `02_process.py` assumes the DEM and image grids are aligned, which was not
  verified.

## Running the code

```bash
pip install -r requirements.txt
earthengine authenticate

python tests/test_synthetic.py           # synthetic tests (no network needed)
python scripts/01_fetch_gee.py           # queues exports to Google Drive
# download the exports into data/ and add data/gauge_daily.csv
python scripts/02_process.py
python scripts/03_analyse.py
```

`data/gauge_daily.csv` requires a `date` column (YYYY-MM-DD) and a
`discharge_m3s` column.
