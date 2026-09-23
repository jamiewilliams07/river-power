"""
Channel width from a binary water mask (True = water).

The mask is reduced to a one-pixel-wide centreline by skeletonization.
The Euclidean distance transform of the mask gives, at each centreline
pixel, the distance to the nearest non-water pixel, and the local width
is twice that distance. No cross-sections are constructed, so bends
require no estimate of local channel orientation. The method is similar
in principle to RivWidth (Pavelsky and Smith 2008). Accuracy is assessed
in tests/test_synthetic.py.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def clean_mask(mask, min_size=20, fill_holes=True):
    """Remove water objects smaller than min_size pixels and fill holes.

    Holes are enclosed non-water regions within the channel, such as
    exposed bars.
    """
    mask = np.asarray(mask, dtype=bool)
    if min_size > 0:
        # skimage's remove_small_objects is avoided because its arguments
        # changed in scikit-image 0.26
        labels, n = ndimage.label(mask)
        if n > 0:
            sizes = np.bincount(labels.ravel())
            too_small = sizes < min_size
            too_small[0] = False  # background
            mask = mask & ~too_small[labels]
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return mask


def largest_component(mask):
    """Keep only the largest connected water body (drops ponds and paddy fields)."""
    mask = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels == keep


def prune_skeleton(skel, min_branch=10):
    """Remove short side branches (spurs) from the skeleton.

    Endpoint pixels are stripped min_branch times, which removes any branch
    shorter than min_branch pixels. Unpruned spurs terminate near the bank
    and produce spuriously small widths. The main centreline is also
    shortened by up to min_branch pixels at each end.
    """
    skel = np.asarray(skel, dtype=bool).copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(int(min_branch)):
        neighbours = ndimage.convolve(
            skel.astype(np.uint8), kernel, mode="constant", cval=0
        )
        endpoints = skel & (neighbours <= 1)
        if not endpoints.any():
            break
        skel &= ~endpoints
    return skel


def order_centreline(skel):
    """Order the skeleton pixels along the channel.

    A breadth-first search (BFS) from an arbitrary pixel finds one end of
    the centreline, and a second BFS from that end finds the other. Returns
    the path between the two ends as an (N, 2) int array of (row, col).
    """
    skel = np.asarray(skel, dtype=bool)
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return pts
    index = {tuple(p): i for i, p in enumerate(pts)}

    # Adjacency list for each pixel (8-connectivity)
    neigh = [[] for _ in pts]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]
    for i, (r, c) in enumerate(pts):
        for dr, dc in offsets:
            j = index.get((r + dr, c + dc))
            if j is not None:
                neigh[i].append(j)

    def bfs(start):
        dist = {start: 0}
        order = [start]
        head = 0
        while head < len(order):
            u = order[head]
            head += 1
            for v in neigh[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    order.append(v)
        return dist, order

    dist_a, _ = bfs(0)
    far_a = max(dist_a, key=dist_a.get)
    dist_b, _ = bfs(far_a)
    far_b = max(dist_b, key=dist_b.get)

    # Trace back from far_b to far_a along decreasing BFS distance
    path = [far_b]
    current = far_b
    while current != far_a:
        nxt = min(
            (v for v in neigh[current] if v in dist_b),
            key=lambda v: dist_b[v],
            default=None,
        )
        if nxt is None or dist_b[nxt] >= dist_b[current]:
            break
        path.append(nxt)
        current = nxt
    return pts[path]


def widths_from_mask(mask, pixel_size=10.0, min_size=20,
                     min_branch=10, isolate=True):
    """Channel width along the centreline from a water mask.

    pixel_size is in m (10 for Sentinel-2 and Sentinel-1 exports).
    min_size and min_branch are passed to clean_mask and prune_skeleton;
    isolate=True keeps only the largest water body.
    Returns a dict with nodes (N, 2), chainage (m), width (m) and skeleton.
    """
    mask = clean_mask(mask, min_size=min_size)
    if isolate:
        mask = largest_component(mask)

    dt = ndimage.distance_transform_edt(mask)
    skel = skeletonize(mask)
    skel = prune_skeleton(skel, min_branch=min_branch)
    nodes = order_centreline(skel)

    if len(nodes) == 0:
        return {"nodes": nodes, "chainage": np.array([]),
                "width": np.array([]), "skeleton": skel}

    width = 2.0 * dt[nodes[:, 0], nodes[:, 1]] * pixel_size

    steps = np.sqrt(np.sum(np.diff(nodes.astype(float), axis=0) ** 2, axis=1))
    chainage = np.concatenate([[0.0], np.cumsum(steps)]) * pixel_size

    return {"nodes": nodes, "chainage": chainage,
            "width": width, "skeleton": skel}


def resample_along_chainage(chainage, values, spacing=100.0, reducer=np.median):
    """Aggregate values into reaches of length `spacing` (m).

    Each reach is summarised with reducer (the median by default).
    Returns (bin_centres, reduced_values).
    """
    chainage = np.asarray(chainage, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(chainage) == 0:
        return np.array([]), np.array([])
    edges = np.arange(chainage.min(), chainage.max() + spacing, spacing)
    if len(edges) < 2:
        return np.array([chainage.mean()]), np.array([reducer(values)])
    idx = np.digitize(chainage, edges) - 1
    centres, out = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        out.append(reducer(values[sel]))
    return np.array(centres), np.array(out)
