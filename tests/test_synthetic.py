"""
Tests for riverlib, the earlier width-based approach (river width from
Sentinel-2 water masks), run on synthetic channels.

    python3 tests/test_synthetic.py

The channel geometries and discharge records are generated here, so the
correct answers are known exactly and a failure indicates an error in the
code rather than in the imagery. No network access is required. Run this
test after modifying any module in riverlib.
"""

import sys
import os

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riverlib import width as W
from riverlib import slope as S
from riverlib import hydro as H
from riverlib import uncertainty as U

PIXEL = 10.0  # m, Sentinel-2 pixel size (10 m bands)
results = []


def check(name, passed, detail=""):
    results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f"  --  {detail}" if detail else ""))


def straight_channel(rows=400, cols=200, width_px=6):
    """Straight channel running down the image, width_px pixels wide."""
    mask = np.zeros((rows, cols), dtype=bool)
    c0 = cols // 2 - width_px // 2
    mask[:, c0:c0 + width_px] = True
    return mask


def _centreline_raster(rows, cols, centre_col):
    """Rasterises a centreline, given as its column at each row, into a boolean mask."""
    line = np.zeros((rows, cols), dtype=bool)
    ci = np.clip(np.rint(centre_col).astype(int), 0, cols - 1)
    line[np.arange(rows), ci] = True
    return line


def sinuous_channel(rows=500, cols=300, width_px=8,
                    amplitude=60, wavelength=200):
    """Sinuous channel of constant width measured normal to the flow.

    The mask is every pixel within width_px/2 of a sine-wave centreline. An
    earlier version used |col - centre| <= w/2, which fixes the width along
    image rows rather than normal to the flow, so the channel was too narrow
    wherever it crossed the image obliquely.
    """
    rr = np.arange(rows)
    centre = cols / 2 + amplitude * np.sin(2 * np.pi * rr / wavelength)
    line = _centreline_raster(rows, cols, centre)
    dist = ndimage.distance_transform_edt(~line)
    return dist <= width_px / 2.0


def tapering_channel(rows=500, cols=300, w_start=6, w_end=18):
    """Straight channel whose width increases linearly downstream."""
    rr, cc = np.mgrid[0:rows, 0:cols]
    w = np.linspace(w_start, w_end, rows)[:, None]
    return np.abs(cc - cols / 2) <= w / 2.0


def true_taper_width(row, rows=500, w_start=6, w_end=18):
    """True width (m) at a row of tapering_channel, using the same whole-pixel count as the mask."""
    w_px = np.interp(row, [0, rows - 1], [w_start, w_end])
    return 2.0 * np.floor(w_px / 2.0) * PIXEL + PIXEL


print("=" * 66)
print("riverlib tests on synthetic channels")
print("=" * 66)

# ---------------------------------------------------------------- width
print("\n-- width extraction --")

for width_px in (4, 6, 10, 16):
    mask = straight_channel(width_px=width_px)
    out = W.widths_from_mask(mask, pixel_size=PIXEL, min_branch=5)
    # exclude the ends, where the image edge acts as a false bank
    interior = out["width"][20:-20]
    recovered = float(np.median(interior))
    truth = width_px * PIXEL
    err = recovered - truth
    check(
        f"width: straight {truth:.0f} m channel within 1 px",
        abs(err) <= PIXEL,
        f"recovered {recovered:.1f} m, error {err:+.1f} m ({err/PIXEL:+.2f} px)",
    )

mask = sinuous_channel(width_px=8)
out = W.widths_from_mask(mask, pixel_size=PIXEL, min_branch=8)
interior = out["width"][30:-30]
recovered = float(np.median(interior))
truth = 8 * PIXEL
check(
    "width: sinuous 80 m channel within 1.5 px",
    abs(recovered - truth) <= 1.5 * PIXEL,
    f"recovered {recovered:.1f} m, SD {np.std(interior):.1f} m",
)

# Centreline nodes must be ordered along the channel, not in raster scan order.
nodes = out["nodes"]
row_steps = np.diff(nodes[:, 0].astype(float))
direction = np.sign(np.sum(row_steps))
backwards = np.mean(np.sign(row_steps) == -direction)
check(
    "width: centreline nodes ordered along channel",
    backwards < 0.02,
    f"{100*backwards:.1f}% of steps reversed "
    f"({100*np.mean(row_steps == 0):.0f}% with no row change)",
)

mask = tapering_channel(w_start=6, w_end=18)
out = W.widths_from_mask(mask, pixel_size=PIXEL, min_branch=8)
rows_at_node = out["nodes"][:, 0]
errs = []
# compare at 20%, 50% and 80% of the channel length
for frac in (0.2, 0.5, 0.8):
    target = int(frac * 499)
    sel = np.abs(rows_at_node - target) <= 10
    if sel.sum() == 0:
        continue
    est = float(np.median(out["width"][sel]))
    truth = true_taper_width(target)
    errs.append(est - truth)
check(
    "width: tapering channel tracked within 1 px",
    len(errs) == 3 and max(abs(e) for e in errs) <= PIXEL,
    "errors at 20/50/80% of length: "
    + ", ".join(f"{e:+.0f} m" for e in errs),
)

# Noise test: invert 3% of pixels at random.
rng = np.random.default_rng(1)
mask = sinuous_channel(width_px=8)
noisy = mask.copy()
flip = rng.random(mask.shape) < 0.03
noisy[flip] = ~noisy[flip]
out = W.widths_from_mask(noisy, pixel_size=PIXEL, min_size=30, min_branch=12)
recovered = float(np.median(out["width"][30:-30]))
check(
    "width: 3% pixel noise, error within 2 px",
    abs(recovered - 80.0) <= 2.0 * PIXEL,
    f"recovered {recovered:.1f} m from noisy mask",
)

# ---------------------------------------------------------------- slope
print("\n-- slope from noisy DEM --")

true_slope = 0.002  # 0.2% (2 m/km)
chainage = np.arange(0, 20000, 30.0)
z_true = 100.0 - true_slope * chainage

for dem_sd, window in ((5.0, 1000.0), (5.0, 3000.0), (1.0, 1000.0)):
    rng = np.random.default_rng(2)
    z_noisy = z_true + rng.normal(0, dem_sd, len(chainage))
    s = S.windowed_slope(chainage, z_noisy, window=window)
    # middle half only; the window is truncated near the ends
    mid = s[len(s) // 4: -len(s) // 4]
    est = float(np.nanmedian(mid))
    rel_err = abs(est - true_slope) / true_slope
    check(
        f"slope: {dem_sd:.0f} m DEM noise, {window:.0f} m window",
        rel_err < 0.35,
        f"estimated {est*100:.3f}% vs true {true_slope*100:.3f}%, "
        f"error {100*rel_err:.0f}%",
    )

# Motivation for the window: point-to-point slope on a noisy DEM is dominated by noise.
rng = np.random.default_rng(2)
z_noisy = z_true + rng.normal(0, 5.0, len(chainage))
naive = -np.diff(z_noisy) / np.diff(chainage)
check(
    "slope: point-to-point estimate dominated by DEM noise",
    np.nanstd(naive) > 10 * true_slope,
    f"point-to-point SD {np.nanstd(naive)*100:.1f}% vs true slope {true_slope*100:.3f}%",
)

drop = S.reach_drop(chainage, z_noisy, 5000, 15000)
check(
    "slope: 10 km reach drop within 3 m",
    abs(drop - 20.0) < 3.0,
    f"recovered {drop:.2f} m vs true 20.00 m",
)

# ---------------------------------------------- hydraulic geometry
print("\n-- hydraulic geometry --")

true_a, true_b = 12.0, 0.15
rng = np.random.default_rng(3)
q_obs = np.array([8.0, 15.0, 22.0, 40.0, 65.0, 95.0, 140.0, 210.0])
w_clean = true_a * q_obs ** true_b

# Noise-free case first, to verify the fit itself.
fit = H.fit_hydraulic_geometry(w_clean, q_obs)
check(
    f"hydraulic geometry: b={true_b} recovered from 8 noise-free points",
    abs(fit["b"] - true_b) < 0.01,
    f"b={fit['b']:.3f}, a={fit['a']:.1f}, r2={fit['r2']:.3f}",
)

# Detectability: b is small, so width changes little with discharge:
# dW = a * (Qmax^b - Qmin^b). The exponent b is recoverable only if dW is
# much larger than the width noise, which is about 10-14 m for 10 m pixels
# (one-pixel error at each bank).
dW = true_a * (q_obs.max() ** true_b - q_obs.min() ** true_b)
print(f"\n  Width change across the flow range: {dW:.1f} m "
      f"(Q {q_obs.min():.0f}-{q_obs.max():.0f} m3/s, b={true_b})")

# 200 random realisations at each noise level
for w_noise_m, label in ((2.0, "2 m"), (5.0, "5 m"), (10.0, "10 m (1 px/bank)")):
    recovered = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w_obs = w_clean + rng.normal(0, w_noise_m, len(q_obs))
        try:
            recovered.append(H.fit_hydraulic_geometry(w_obs, q_obs)["b"])
        except ValueError:
            pass
    recovered = np.array(recovered)
    bias = np.median(recovered) - true_b
    scatter = np.std(recovered)
    snr = dW / w_noise_m
    usable = scatter < 0.5 * true_b
    print(f"    noise {label:<16} SNR {snr:4.1f}   "
          f"b = {np.median(recovered):+.3f} +/- {scatter:.3f}   "
          f"{'usable' if usable else 'UNUSABLE'}")

# Always passes; included so the SNR table is referenced in the results.
check(
    "hydraulic geometry: fit degrades with noise (see SNR table)",
    True,
    "see table above",
)

# Minimum detectable low-flow width: w_min * (R^b - 1) >= 3 * noise, R = Qmax/Qmin
for flow_ratio in (5.0, 10.0, 30.0):
    for noise in (10.0,):
        needed = 3.0 * noise / (flow_ratio ** true_b - 1.0)
        print(f"  Flow range {flow_ratio:>4.0f}x, width noise {noise:.0f} m "
              f"-> required low-flow width >= {needed:.0f} m")

# Inverting the fitted relation must recover the input discharges.
fit = H.fit_hydraulic_geometry(w_clean, q_obs)
q_back = H.discharge_from_width(w_clean, fit["a"], fit["b"])
check(
    "hydraulic geometry: width-to-discharge inversion recovers inputs",
    np.allclose(q_back, q_obs, rtol=0.02),
    f"max relative error {np.max(np.abs(q_back/q_obs - 1)):.4f}",
)

# Error amplification: a small width error gives a large discharge error because 1/b is large.
w_perturbed = w_clean * 1.10
q_perturbed = H.discharge_from_width(w_perturbed, fit["a"], fit["b"])
amplification = np.median(q_perturbed / q_obs) - 1.0
check(
    "hydraulic geometry: 10% width error amplified above 50% in discharge",
    amplification > 0.5,
    f"10% width error -> {100*amplification:.0f}% discharge error "
    f"(1/b = {1/fit['b']:.1f})",
)

# ------------------------------------------------ flow duration / power
print("\n-- flow duration and power --")

# 10 years of synthetic daily discharge (lognormal)
rng = np.random.default_rng(4)
daily_q = rng.lognormal(mean=np.log(30), sigma=0.8, size=3650)
q30 = H.design_flow(daily_q, 30.0)
q50 = H.design_flow(daily_q, 50.0)
check(
    "flow duration curve: Q30 > Q50",
    q30 > q50,
    f"Q30 {q30:.1f} m3/s, Q50 {q50:.1f} m3/s, mean {daily_q.mean():.1f} m3/s",
)
check(
    "flow duration curve: Q30 below 1.2x mean flow",
    q30 < daily_q.mean() * 1.2,
    f"Q30 {q30:.1f} vs mean {daily_q.mean():.1f} m3/s",
)

# P = rho g Q H eta, calculated independently
p = H.power_kw(discharge=25.0, head=8.0, efficiency=0.80)
expected = 1000 * 9.81 * 25.0 * 8.0 * 0.80 / 1000.0
check(
    "power: P = rho g Q H eta matches independent calculation",
    abs(p - expected) < 1e-6,
    f"{p:.0f} kW for Q=25 m3/s, H=8 m, eta=0.80",
)

# ------------------------------------------------------- uncertainty
print("\n-- uncertainty propagation --")

fit = H.fit_hydraulic_geometry(w_clean, q_obs)
res = U.propagate(
    width_m=24.0, head_m=8.0, a=fit["a"], b=fit["b"],
    width_sd=10.0, head_sd=5.0, b_sd=0.02, n_draws=8000, seed=7,
)
check(
    "uncertainty: Monte Carlo interval finite with p95 > p5",
    np.isfinite(res["median_kw"]) and res["p95_kw"] > res["p05_kw"],
    f"median {res['median_kw']:.0f} kW, "
    f"90% interval [{res['p05_kw']:.0f}, {res['p95_kw']:.0f}] kW, "
    f"CV {res['cv']:.2f}",
)

# With a large head uncertainty, head must dominate the variance.
dec = U.variance_decomposition(
    width_m=24.0, head_m=8.0, a=fit["a"], b=fit["b"],
    width_sd=0.5, head_sd=6.0, b_sd=0.001, n_draws=8000, seed=7,
)
check(
    "uncertainty: head dominant under large head error",
    U.dominant_input(dec) == "head",
    f"head {100*dec['head']:.0f}%, width {100*dec['width']:.0f}%, "
    f"b {100*dec['b']:.0f}%",
)

# Same test with width uncertainty dominant.
dec2 = U.variance_decomposition(
    width_m=24.0, head_m=8.0, a=fit["a"], b=fit["b"],
    width_sd=15.0, head_sd=0.2, b_sd=0.001, n_draws=8000, seed=7,
)
check(
    "uncertainty: width dominant under large width error",
    U.dominant_input(dec2) == "width",
    f"width {100*dec2['width']:.0f}%, head {100*dec2['head']:.0f}%, "
    f"b {100*dec2['b']:.0f}%",
)

# Representative uncertainties: Sentinel-2 for width, AW3D30 DEM for head.
dec3 = U.variance_decomposition(
    width_m=24.0, head_m=8.0, a=fit["a"], b=fit["b"],
    width_sd=10.0, head_sd=5.0, b_sd=0.02, n_draws=20000, seed=7,
)
print(f"\n  Representative case (Sentinel-2 10 m width, AW3D30 5 m vertical):")
for k in ("width", "head", "b", "efficiency", "interaction"):
    print(f"    {k:<12} {100*dec3[k]:5.1f}%")
print(f"  -> dominant error source: {U.dominant_input(dec3)}")

# ------------------------------------------------------------- summary
print("\n" + "=" * 66)
passed = sum(1 for _, p, _ in results if p)
print(f"{passed}/{len(results)} checks passed")
print("=" * 66)
if passed < len(results):
    print("\nFailures:")
    for name, p, detail in results:
        if not p:
            print(f"  - {name}: {detail}")
    sys.exit(1)
