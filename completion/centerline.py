"""Probability-guided geodesic centreline forest (no binary thinning).

The optional binary ``domain`` only bounds where paths may travel. Within each
8-connected domain component, paths follow a continuous probability and radius
cost and join a common root. This is a TEASAR-like prototype, not an exact
topology-preserving medial axis: loops can be represented by a spanning tree.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric


_CONNECTIVITY = np.ones((3, 3), dtype=bool)
_DEFAULT_DOMAIN_THRESHOLD = 0.15


def _cover_path(
    available: np.ndarray,
    path: np.ndarray,
    radius: np.ndarray,
) -> None:
    """Suppress local vessel cross-sections already represented by a path."""
    height, width = available.shape
    for y, x in path:
        # The support radius, rather than A/V confidence, must cover the vessel
        # width even where its class probability momentarily falls to zero.
        distance = 1.25 * float(radius[y, x]) + 0.5
        extent = int(math.ceil(distance))
        y0, y1 = max(0, y - extent), min(height, y + extent + 1)
        x0, x1 = max(0, x - extent), min(width, x + extent + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        covered = (yy - y) ** 2 + (xx - x) ** 2 <= distance**2
        available[y0:y1, x0:x1][covered] = False


def _one_component(
    probability: np.ndarray,
    domain: np.ndarray,
    min_branch_length: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Padding makes the distance radius well defined even for an all-true crop.
    radius = ndi.distance_transform_edt(np.pad(domain, 1))[1:-1, 1:-1]
    smooth_probability = ndi.gaussian_filter(probability * domain, 0.8)
    smooth_mass = ndi.gaussian_filter(domain.astype(np.float64), 0.8)
    smooth_probability /= np.maximum(smooth_mass, 1e-12)
    soft_radius = radius * np.sqrt(0.15 + 0.85 * smooth_probability)

    # Probability is a soft preference, never a forbidden gap. The radius term
    # penalizes edge shortcuts and remains useful across a low-confidence gap.
    path_cost = (0.15 + (1.0 - smooth_probability) ** 2) / (
        0.5 + soft_radius
    ) ** 2
    path_cost[~domain] = np.inf
    root = tuple(int(i) for i in np.unravel_index(np.argmax(soft_radius), domain.shape))
    solver = MCP_Geometric(path_cost, fully_connected=True)
    solver.find_costs([root])

    # Geometric distance ranks extent, independently of the cost scale. Ranking
    # by probability cost itself would preferentially grow into uncertain edges.
    geometric_cost = np.where(domain, 1.0, np.inf)
    geometric_distance, _ = MCP_Geometric(
        geometric_cost, fully_connected=True
    ).find_costs([root])
    priority = np.where(domain, geometric_distance + 0.5 * soft_radius, -np.inf)
    available = domain.copy()
    result = np.zeros_like(domain)
    result[root] = True
    _cover_path(available, np.asarray([root]), radius)
    accepted_lengths: list[float] = []
    rejected = 0
    while np.any(available):
        target = tuple(
            int(i) for i in np.unravel_index(
                np.argmax(np.where(available, priority, -np.inf)), domain.shape
            )
        )
        path = np.asarray(solver.traceback(target), dtype=np.intp)
        # The existing path is connected to the common root. Keep only the new
        # suffix, including its connection pixel, to measure branch length.
        intersections = np.flatnonzero(result[path[:, 0], path[:, 1]])
        branch = path[int(intersections[-1]) :]
        length = float(np.linalg.norm(np.diff(branch, axis=0), axis=1).sum())
        if not accepted_lengths or length >= min_branch_length:
            result[branch[:, 0], branch[:, 1]] = True
            accepted_lengths.append(length)
        else:
            rejected += 1
        # Cover rejected short twigs too, so every iteration removes at least
        # the selected target and termination is independent of the threshold.
        _cover_path(available, branch, radius)
    component_debug = {
        "domain_pixels": int(domain.sum()),
        "centerline_pixels": int(result.sum()),
        "root_yx": list(root),
        "paths_accepted": len(accepted_lengths),
        "short_paths_rejected": rejected,
        "accepted_path_lengths": accepted_lengths,
        "radius_max": float(radius.max()),
        "soft_radius_max": float(soft_radius.max()),
    }
    return result, component_debug


def extract_centerline(
    probability: np.ndarray,
    domain: np.ndarray | None = None,
    min_branch_length: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic, connected path forest inside a vessel domain.

    Parameters
    ----------
    probability:
        Finite 2-D probability array in [0, 1]. Empty arrays return empty output.
    domain:
        Permitted vessel support, normally derived from the common BV channel.
        If omitted, ``probability >= 0.15`` removes background. This fallback
        explicitly uses a domain threshold; it does not threshold and thin a
        class mask. Supplying BV support permits paths through zero A/V values.
    min_branch_length:
        Minimum additional geometric path length in pixels. A component's first
        path is retained regardless, so tiny components never silently vanish.

    Notes
    -----
    One connected result is produced per 8-connected input component. There are
    no paths outside ``domain``. Close branches can touch in the pixel output,
    and cycles are not guaranteed to survive this spanning-tree construction.
    """
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probability must be a 2-D array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probability must contain finite values in [0, 1]")
    if not np.isfinite(min_branch_length) or min_branch_length < 0:
        raise ValueError("min_branch_length must be finite and nonnegative")
    if domain is None:
        support = values >= _DEFAULT_DOMAIN_THRESHOLD
        domain_source = "probability_background_threshold"
    else:
        support_array = np.asarray(domain)
        if support_array.shape != values.shape:
            raise ValueError("domain and probability must have the same shape")
        if not np.isfinite(support_array).all():
            raise ValueError("domain must contain finite values")
        support = support_array.astype(bool)
        domain_source = "provided"
    result = np.zeros(values.shape, dtype=bool)
    labels, count = ndi.label(support, structure=_CONNECTIVITY)
    components = []
    regions = ndi.find_objects(labels, max_label=int(count)) if count else []
    for component_id, region in enumerate(regions, start=1):
        if region is None:
            continue
        local_domain = labels[region] == component_id
        local_result, detail = _one_component(
            values[region], local_domain, float(min_branch_length)
        )
        result[region] |= local_result
        detail["component_id"] = component_id
        detail["root_yx"] = [
            detail["root_yx"][axis] + region[axis].start for axis in range(2)
        ]
        components.append(detail)
    _, centerline_components = ndi.label(result, structure=_CONNECTIVITY)
    debug = {
        "algorithm": "probability_radius_geodesic_forest_v1",
        "domain_source": domain_source,
        "domain_threshold": _DEFAULT_DOMAIN_THRESHOLD if domain is None else None,
        "min_branch_length": float(min_branch_length),
        "domain_pixels": int(support.sum()),
        "domain_components": int(count),
        "centerline_pixels": int(result.sum()),
        "centerline_components": int(centerline_components),
        "outside_domain_pixels": int(np.count_nonzero(result & ~support)),
        "paths_accepted": sum(item["paths_accepted"] for item in components),
        "short_paths_rejected": sum(item["short_paths_rejected"] for item in components),
        "components": components,
    }
    return result, debug
