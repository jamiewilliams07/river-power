"""
riverlib: channel width and slope from satellite imagery, and the
resulting hydropower estimate.

This package implements the project's original, now discontinued,
approach (see docs/sentinel-width-method.md). It is retained because the
synthetic tests in tests/test_synthetic.py still exercise it.

Modules:
    masks        water masks from Sentinel-2 (optical) and Sentinel-1 (SAR)
    width        centreline extraction and channel width from a water mask
    slope        longitudinal slope and reach drop from a DEM
    hydro        at-a-station hydraulic geometry, flow duration curve, power
    uncertainty  Monte Carlo error propagation and variance decomposition
"""

from . import masks, width, slope, hydro, uncertainty

__all__ = ["masks", "width", "slope", "hydro", "uncertainty"]
__version__ = "0.1.0"
