"""
Longitudinal slope along the centreline from a DEM.

The AW3D30 DEM has a vertical error of about 5 m on 30 m pixels, so a
slope computed between adjacent pixels is dominated by noise. The
point-to-point result in tests/test_synthetic.py motivated a
least-squares fit of elevation against chainage (distance along the
centreline) over a moving window of 1 km or more.
"""

import numpy as np


def sample_profile(dem, nodes, nodata=None):
    """DEM elevation (m) at each centreline node; nodata values become NaN."""
    dem = np.asarray(dem, dtype=float)
    nodes = np.asarray(nodes)
    if len(nodes) == 0:
        return np.array([])
    z = dem[nodes[:, 0], nodes[:, 1]].astype(float)
    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)
    return z


def monotonic_fill(z):
    """Running minimum, so that elevation never increases downstream.

    Removes spurious rises caused by DEM noise, at the cost of a slight
    low bias in the resulting slope.
    """
    z = np.asarray(z, dtype=float)
    out = np.copy(z)
    running = np.inf
    for i, val in enumerate(out):
        if np.isnan(val):
            continue
        running = min(running, val)
        out[i] = running
    return out


def windowed_slope(chainage, elevation, window=1000.0, min_points=5):
    """Least-squares slope of elevation against chainage over a moving window.

    chainage and elevation are in m; window is the full window length, not
    the half-width. Windows with fewer than min_points valid samples return
    NaN. Returns S = -dz/dx, positive for a channel that descends
    downstream.
    """
    chainage = np.asarray(chainage, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    n = len(chainage)
    slope = np.full(n, np.nan)
    half = window / 2.0

    for i in range(n):
        sel = np.abs(chainage - chainage[i]) <= half
        sel &= np.isfinite(elevation)
        if sel.sum() < min_points:
            continue
        x = chainage[sel]
        y = elevation[sel]
        if np.ptp(x) <= 0:
            continue
        grad = np.polyfit(x, y, 1)[0]
        slope[i] = -grad
    return slope


def reach_drop(chainage, elevation, start, end):
    """Elevation drop (m) between two chainages, used as the gross head.

    The drop is taken from a least-squares line over the whole reach, so a
    single noisy pixel at either end cannot determine the result.
    """
    chainage = np.asarray(chainage, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    sel = (chainage >= start) & (chainage <= end) & np.isfinite(elevation)
    if sel.sum() < 2:
        return np.nan
    x, y = chainage[sel], elevation[sel]
    slope, intercept = np.polyfit(x, y, 1)
    z_start = slope * x.min() + intercept
    z_end = slope * x.max() + intercept
    return float(z_start - z_end)
