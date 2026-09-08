"""Complete complementary A/V masks with paths inside the BV probability field.

Class confidence creates semantic anchors, never an independent vessel mask.
Ambiguous BV pixels receive a class by a within-vessel geodesic competition.
Short gaps between compatible, facing class components are joined along BV
probability paths. All accepted routes are retained and reported, including
crossing overlap. No binary skeleton is used to find or join these gaps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.graph import MCP_Geometric
from skimage.morphology import remove_small_objects


S8 = np.ones((3, 3), dtype=bool)
STEPS = [(dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1)
         for dx in (-1, 0, 1) if dy or dx]


@dataclass(frozen=True)
class ProbabilityConfig:
    low_bv: float = 0.25
    high_bv: float = 0.50
    min_bv_area: int = 20
    semantic_margin: float = 0.12
    min_anchor_area: int = 4
    max_bridge_length: float = 40.0
    max_opposite_run: int = 14
    min_facing_cosine: float = 0.10
    direction_radius: float = 10.0
    bridge_passes: int = 2
    strict_direction_check: bool = False

    def __post_init__(self):
        if not math.isfinite(self.max_bridge_length) or self.max_bridge_length < 0:
            raise ValueError("max_bridge_length must be finite and nonnegative")
        for name in ("max_opposite_run", "bridge_passes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def _validate(*arrays):
    values = [np.asarray(array, dtype=np.float32) for array in arrays]
    if any(a.ndim != 2 or not a.size for a in values):
        raise ValueError("probabilities must be nonempty 2D arrays")
    if any(a.shape != values[0].shape for a in values):
        raise ValueError("probabilities must have the same shape")
    if any(not np.isfinite(a).all() or a.min() < 0 or a.max() > 1 for a in values):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return values


def _geodesic_distance(cost, seeds):
    if not np.any(seeds):
        return np.full(cost.shape, np.inf)
    solver = MCP_Geometric(cost, fully_connected=True)
    distance, _ = solver.find_costs(np.argwhere(seeds))
    return distance


def _outward_direction(labels, component, point, radius, strict=False):
    y, x = point
    r = int(math.ceil(radius))
    y0, y1 = max(0, y-r), min(labels.shape[0], y+r+1)
    x0, x1 = max(0, x-r), min(labels.shape[1], x+r+1)
    coords = np.argwhere(labels[y0:y1, x0:x1] == component) + [y0, x0]
    offsets = coords - point
    lengths = np.linalg.norm(offsets, axis=1)
    useful = (lengths >= max(2, radius * 0.3)) & (lengths <= radius)
    if useful.sum() < 3:
        return None
    vector = -np.mean(offsets[useful], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm < 0.5:
        return None
    if not strict:
        return vector / norm
    centered = offsets[useful] - np.mean(offsets[useful], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    if eigenvalues[-1] < 1.5 * max(eigenvalues[0], 1e-6):
        return None
    tangent = eigenvectors[:, -1]
    projection = float(np.dot(tangent, vector / norm))
    # A point on a long vessel's side has an outward normal but is not a
    # terminal arm. Such a normal must never join two parallel vessels.
    if abs(projection) < 0.45:
        return None
    return tangent * (1 if projection > 0 else -1)


def _longest_run(values):
    longest = run = 0
    for value in values:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return longest


def _front_meetings(mask, domain, cost, max_distance):
    """Multi-source Dijkstra with component provenance and explicit traceback."""
    labels, count = ndi.label(mask, structure=S8)
    shape = mask.shape
    width = shape[1]
    owner = labels.ravel().copy()
    distance = np.where(mask, 0.0, np.inf).ravel()
    parent = np.full(mask.size, -1, dtype=np.int64)
    boundary = mask & ndi.binary_dilation(domain & ~mask, structure=S8)
    queue = [(0.0, int(i)) for i in np.flatnonzero(boundary)]
    heapq.heapify(queue)
    flat_domain, flat_cost = domain.ravel(), cost.ravel()
    meetings = {}
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance[current] or current_distance > max_distance:
            continue
        y, x = divmod(current, width)
        for dy, dx, step in STEPS:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < shape[0] and 0 <= xx < width):
                continue
            neighbour = yy * width + xx
            if not flat_domain[neighbour]:
                continue
            edge = step * (flat_cost[current] + flat_cost[neighbour]) * 0.5
            candidate = current_distance + edge
            if owner[neighbour] and owner[neighbour] != owner[current]:
                pair = tuple(sorted((int(owner[current]), int(owner[neighbour]))))
                total = candidate + distance[neighbour]
                if total < meetings.get(pair, (np.inf, 0, 0))[0]:
                    meetings[pair] = (total, current, neighbour)
            if candidate < distance[neighbour] and candidate <= max_distance:
                distance[neighbour] = candidate
                owner[neighbour] = owner[current]
                parent[neighbour] = current
                heapq.heappush(queue, (candidate, neighbour))

    def trace(index):
        path = [index]
        while parent[index] >= 0:
            index = int(parent[index])
            path.append(index)
        return path

    paths = []
    for _, (score, left, right) in sorted(meetings.items(), key=lambda item: item[1]):
        path = trace(left)[::-1] + trace(right)
        coords = np.asarray([divmod(index, width) for index in path], dtype=int)
        paths.append((score, coords))
    return labels, int(count), paths


def _bridge(mask, domain, bv, class_probability, cfg):
    # Finite throughout the BV domain: a low class probability raises cost,
    # but cannot by itself delete a real corridor.
    cost = 1.0 + 1.5 * (1.0-bv) + 1.5 * np.maximum(0, 0.5-class_probability)
    labels, count, candidates = _front_meetings(
        mask, domain, cost, cfg.max_bridge_length * 3.0,
    )
    result = mask.copy()
    current_labels = labels
    bridges = []
    rejected = {}
    roots = list(range(count+1))

    def root(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    for score, route in candidates:
        first, last = tuple(route[0]), tuple(route[-1])
        left, right = int(labels[first]), int(labels[last])
        if not left or not right or root(left) == root(right):
            continue
        length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
        gap = ~result[route[:, 0], route[:, 1]]
        reason = None
        span = route[-1] - route[0]
        separation = float(np.linalg.norm(span))
        opposite_run = _longest_run(class_probability[route[:, 0], route[:, 1]] < 0.30)
        if length > cfg.max_bridge_length:
            reason = "too_long"
        elif separation == 0 or length > 2.5 * separation + 2:
            reason = "detour"
        elif opposite_run > cfg.max_opposite_run:
            reason = "long_opposite_vessel"
        else:
            # Earlier seam repairs can turn an apparent endpoint into a side
            # of a complete vessel. Re-evaluate against the updated geometry.
            direction_left = _outward_direction(
                current_labels, int(current_labels[first]), route[0], cfg.direction_radius,
                cfg.strict_direction_check,
            )
            direction_right = _outward_direction(
                current_labels, int(current_labels[last]), route[-1], cfg.direction_radius,
                cfg.strict_direction_check,
            )
            unit = span / max(separation, 1e-6)
            alignment_left = None if direction_left is None else float(np.dot(direction_left, unit))
            alignment_right = None if direction_right is None else float(np.dot(direction_right, -unit))
            # One- or two-pixel classification seams need no unstable tangent.
            if gap.sum() > 2:
                if alignment_left is None or alignment_right is None:
                    reason = "insufficient_direction"
                elif min(alignment_left, alignment_right) < cfg.min_facing_cosine:
                    reason = "not_facing"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        result[route[:, 0], route[:, 1]] = True
        current_labels, _ = ndi.label(result, structure=S8)
        roots[root(right)] = root(left)
        bridges.append({
            "route_yx": route.tolist(), "length": length,
            "added_pixels": int(gap.sum()), "cost": float(score),
            "min_bv": float(bv[route[:, 0], route[:, 1]].min()),
            "max_opposite_run": opposite_run,
        })
    return result, bridges, rejected


def complete_probability_masks(a, v, bv, *, config=None):
    """Return A mask, V mask, and diagnostic arrays/explicit accepted paths."""
    a, v, bv = _validate(a, v, bv)
    cfg = config or ProbabilityConfig()
    if not 0 <= cfg.low_bv <= cfg.high_bv <= 1:
        raise ValueError("invalid BV thresholds")
    domain = remove_small_objects(
        apply_hysteresis_threshold(bv, cfg.low_bv, cfg.high_bv),
        min_size=cfg.min_bv_area,
    )
    # The two independent sigmoid outputs are evidence, not complementary
    # probabilities. Normalize only the semantic decision inside BV.
    conditional_a = (a + 1e-5) / (a+v+2e-5)
    semantic_evidence = a+v >= 0.25
    a_seeds = remove_small_objects(
        domain & semantic_evidence & (conditional_a >= 0.5+cfg.semantic_margin),
        min_size=cfg.min_anchor_area,
    )
    v_seeds = remove_small_objects(
        domain & semantic_evidence & (conditional_a <= 0.5-cfg.semantic_margin),
        min_size=cfg.min_anchor_area,
    )
    radius = ndi.distance_transform_edt(domain)
    spatial_cost = (1.0 + 2.0*(1.0-bv)) / np.sqrt(0.5+radius)
    spatial_cost[~domain] = np.inf
    distance_a = _geodesic_distance(spatial_cost, a_seeds)
    distance_v = _geodesic_distance(spatial_cost, v_seeds)
    finite_a, finite_v = np.isfinite(distance_a), np.isfinite(distance_v)
    reachable = finite_a | finite_v
    a_mask = domain & reachable & (distance_a <= distance_v)
    v_mask = domain & reachable & (distance_v <= distance_a)
    unknown = domain & ~reachable
    a_mask[unknown] = True
    v_mask[unknown] = True
    initial_a, initial_v = a_mask.copy(), v_mask.copy()
    paths = {"A": [], "V": []}
    rejected = {"A": {}, "V": {}}
    for vessel_type, conditional in (("A", conditional_a), ("V", 1-conditional_a)):
        target = a_mask if vessel_type == "A" else v_mask
        for iteration in range(cfg.bridge_passes):
            target, accepted, refused = _bridge(target, domain, bv, conditional, cfg)
            for item in accepted:
                item["pass"] = iteration
            paths[vessel_type].extend(accepted)
            for key, value in refused.items():
                rejected[vessel_type][key] = rejected[vessel_type].get(key, 0) + value
            if not accepted:
                break
        if vessel_type == "A":
            a_mask = target
        else:
            v_mask = target
    return a_mask, v_mask, {
        "config": asdict(cfg), "bv_mask": domain,
        "conditional_a": conditional_a, "a_seeds": a_seeds, "v_seeds": v_seeds,
        "initial_a_mask": initial_a, "initial_v_mask": initial_v,
        "bridges": paths, "rejected": rejected,
        "unseeded_bv_pixels": int(unknown.sum()),
        "union_equals_bv": bool(np.array_equal(a_mask | v_mask, domain)),
        "retained_a_anchors": bool(np.all(a_mask[a_seeds])),
        "retained_v_anchors": bool(np.all(v_mask[v_seeds])),
    }
