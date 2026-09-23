"""
Water masks from Sentinel-2 (optical) and Sentinel-1 (SAR).

Optical imagery is unavailable under cloud. C-band SAR penetrates cloud,
and calm open water appears dark because of specular reflection, but
wind roughening or vegetation on the water surface can raise its
backscatter to levels similar to land. The intention was to derive both
masks and use their disagreement as an estimate of width uncertainty.
"""

import numpy as np
from skimage.filters import threshold_otsu


def mndwi(green, swir1):
    """Modified Normalized Difference Water Index (Xu 2006).

    MNDWI = (Green - SWIR1) / (Green + SWIR1). On Sentinel-2 these are B3
    (10 m) and B11 (20 m), so B11 must be resampled to 10 m or the whole
    calculation carried out at 20 m. MNDWI was chosen over NDWI because it
    suppresses built-up surfaces more effectively.
    """
    green = np.asarray(green, dtype=float)
    swir1 = np.asarray(swir1, dtype=float)
    denom = green + swir1
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom != 0, (green - swir1) / denom, np.nan)
    return out


def ndwi(green, nir):
    """Normalized Difference Water Index (McFeeters 1996), from B3 and B8.

    Not used by 02_process.py; retained for comparison with MNDWI.
    """
    green = np.asarray(green, dtype=float)
    nir = np.asarray(nir, dtype=float)
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom != 0, (green - nir) / denom, np.nan)
    return out


def threshold_index(index, value=0.0, adaptive=False):
    """Classify pixels with index > value as water (0.0 is standard for MNDWI).

    With adaptive=True the threshold is selected by Otsu thresholding
    instead. Otsu's method fails when water occupies only a small fraction
    of the scene, so the area of interest must be kept tight around the
    channel.
    """
    index = np.asarray(index, dtype=float)
    valid = np.isfinite(index)
    if not valid.any():
        return np.zeros(index.shape, dtype=bool)
    if adaptive:
        value = threshold_otsu(index[valid])
    return valid & (index > value)


def to_db(linear, floor=1e-6):
    """Convert Sentinel-1 backscatter from linear power to dB."""
    linear = np.asarray(linear, dtype=float)
    return 10.0 * np.log10(np.maximum(linear, floor))


def lee_filter(img, size=5):
    """Simplified Lee speckle filter.

    Smooths homogeneous areas such as open water while largely preserving
    edges (the banks), which a plain mean filter would blur. The global
    image variance stands in for the speckle noise variance.
    """
    from scipy.ndimage import uniform_filter

    img = np.asarray(img, dtype=float)
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img ** 2, size)
    var = np.maximum(sq_mean - mean ** 2, 0.0)
    overall_var = np.var(img)
    weights = var / (var + overall_var + 1e-12)
    return mean + weights * (img - mean)


def sar_water_mask(vv_db, speckle_size=5, manual_threshold=None):
    """Water mask from Sentinel-1 VV backscatter in dB. Returns (mask, threshold).

    Pixels below the threshold are classified as water. By default the
    threshold is found by Otsu thresholding on each scene, because the
    backscatter distribution shifts between acquisitions;
    manual_threshold overrides it. Scenes should come from a single
    relative orbit so that the viewing geometry is constant.
    """
    vv_db = np.asarray(vv_db, dtype=float)
    if speckle_size and speckle_size > 1:
        vv_db = lee_filter(vv_db, size=speckle_size)
    valid = np.isfinite(vv_db)
    if not valid.any():
        return np.zeros(vv_db.shape, dtype=bool), np.nan
    thr = (float(manual_threshold) if manual_threshold is not None
           else float(threshold_otsu(vv_db[valid])))
    return valid & (vv_db < thr), thr


def agreement(mask_a, mask_b):
    """Agreement between two water masks of the same scene.

    Returns the IoU (intersection over union) and the pixel counts. An IoU
    below about 0.7 was taken to indicate that the two sensors disagree.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return {
        "iou": float(inter / union) if union else float("nan"),
        "only_a": int(np.logical_and(a, ~b).sum()),
        "only_b": int(np.logical_and(b, ~a).sum()),
        "intersection": int(inter),
        "union": int(union),
    }
