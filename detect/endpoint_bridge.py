"""Direction-guided repairs of a single-class RBAD skeleton.

Trace each endpoint backwards along its parent arm, extrapolate its tangent,
and join another component by endpoint pairing or an endpoint-to-body landing.
Original pixels are never removed. A and V must be supplied in separate calls.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

S8 = np.ones((3, 3), dtype=bool)
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


@dataclass(frozen=True)
class EndpointBridgeConfig:
    max_distance: float = 40.0
    max_angle_deg: float = 35.0
    lookback: float = 12.0
    min_parent_length: float = 4.0
    min_bv_mean: float = 0.25
    low_bv: float = 0.10
    max_unsupported_run: int = 8
    passes: int = 2
    allow_body_landing: bool = True
    min_body_angle_deg: float = 25.0

    def __post_init__(self):
        for name in ('max_distance', 'lookback', 'min_parent_length'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.min_parent_length <= 0 or self.lookback < self.min_parent_length:
            raise ValueError('lookback must be >= min_parent_length > 0')
        for name in ('max_angle_deg', 'min_body_angle_deg'):
            if not 0 <= getattr(self, name) < 90:
                raise ValueError(f'{name} must be in [0, 90)')
        for name in ('min_bv_mean', 'low_bv'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        for name in ('passes', 'max_unsupported_run'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')


def neighbours(skel, point):
    y, x = point
    result = []
    for dy, dx in OFFSETS:
        yy, xx = y+dy, x+dx
        if not (0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1] and skel[yy, xx]):
            continue
        # Ignore redundant diagonals around a raster corner, preserving a
        # single topological continuation instead of a three-pixel triangle.
        if dy and dx and (skel[y, xx] or skel[yy, x]):
            continue
        result.append((yy, xx))
    return result


def _walk(skel, point, start, distance):
    path = [point, start]
    length = float(np.linalg.norm(np.subtract(start, point)))
    while length < distance:
        choices = neighbours(skel, path[-1])
        if len(choices) != 2:
            break
        choices = [p for p in choices if p != path[-2] and p not in path]
        if not choices:
            break
        nxt = choices[0]
        length += float(np.linalg.norm(np.subtract(nxt, path[-1])))
        path.append(nxt)
    return np.asarray(path), length


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-8 else np.asarray(vector)/norm


def _angle(first, second):
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1, 1))))


def endpoints(skel, cfg):
    result = []
    for point in zip(*np.nonzero(skel)):
        adjacent = neighbours(skel, point)
        if len(adjacent) != 1:
            continue
        path, length = _walk(skel, point, adjacent[0], cfg.lookback)
        if length < cfg.min_parent_length:
            continue
        # Several parent-arm pixels stabilize the tangent of jagged diagonals.
        direction = _unit(np.asarray(point)-np.mean(path[max(1,len(path)//2):], axis=0))
        if direction is not None:
            result.append({'point': point, 'direction': direction, 'parent': tuple(path[-1]),
                           'parent_length': length})
    return result


def _path(first, last, first_tangent, last_tangent=None):
    first, last = np.asarray(first, float), np.asarray(last, float)
    distance = float(np.linalg.norm(last-first))
    if last_tangent is None:
        last_tangent = _unit(last-first)
    control1 = first+first_tangent*distance/3
    control2 = last-last_tangent*distance/3
    ts = np.linspace(0, 1, max(3, int(math.ceil(distance*4))+1))[:,None]
    samples = ((1-ts)**3*first + 3*(1-ts)**2*ts*control1
               + 3*(1-ts)*ts**2*control2 + ts**3*last)
    rounded = np.rint(samples).astype(int)
    keep = np.r_[True, np.any(np.diff(rounded, axis=0), axis=1)]
    return rounded[keep]


def _longest_run(values):
    best = run = 0
    for value in values:
        run = run+1 if value else 0
        best = max(best, run)
    return best


def bridge_endpoints(skeleton, *, bv_probability=None, config=None):
    cfg = config or EndpointBridgeConfig()
    values = np.asarray(skeleton)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('skeleton must be a finite nonempty 2D array')
    original = values > 0
    result = original.copy()
    bv = None
    if bv_probability is not None:
        bv = np.asarray(bv_probability, float)
        if bv.shape != result.shape or not np.isfinite(bv).all() or np.any((bv<0)|(bv>1)):
            raise ValueError('BV probability must match skeleton and lie in [0, 1]')
    before_count = int(ndi.label(result, S8)[1])
    accepted, rejected = [], Counter()
    for pass_index in range(cfg.passes if cfg.max_distance else 0):
        labels, count = ndi.label(result, S8)
        ends = endpoints(result, cfg)
        if count < 2 or not ends:
            break
        coords = np.asarray([e['point'] for e in ends])
        tree = cKDTree(coords)
        candidates = []
        for i, j in sorted(tree.query_pairs(cfg.max_distance)):
            first, last = ends[i], ends[j]
            p, q = first['point'], last['point']
            if labels[p] == labels[q]:
                continue
            span = np.asarray(q)-p
            unit = _unit(span)
            alpha, beta = _angle(first['direction'], unit), _angle(last['direction'], -unit)
            if max(alpha, beta) > cfg.max_angle_deg:
                rejected['endpoint_direction'] += 1
                continue
            distance = float(np.linalg.norm(span))
            route = _path(p, q, first['direction'], -last['direction'])
            candidates.append((distance*(1+(alpha+beta)/90), i, j, route, 'endpoint_endpoint', alpha, beta))

        if cfg.allow_body_landing:
            body = np.asarray([p for p in zip(*np.nonzero(result)) if len(neighbours(result,p)) == 2])
            if len(body):
                body_tree = cKDTree(body)
                for i, end in enumerate(ends):
                    p = end['point']
                    # Keep only the best eligible landing for each foreign component.
                    best = {}
                    for body_index in body_tree.query_ball_point(p, cfg.max_distance):
                        q = tuple(body[body_index])
                        component = int(labels[q])
                        if component == labels[p]:
                            continue
                        span = np.asarray(q)-p
                        direction = _unit(span)
                        alpha = _angle(end['direction'], direction)
                        if alpha > cfg.max_angle_deg:
                            continue
                        adjacent = neighbours(result,q)
                        arms = [_walk(result,q,start,cfg.lookback)[0][-1] for start in adjacent]
                        tangent = _unit(arms[1]-arms[0])
                        if tangent is None:
                            continue
                        body_angle = min(_angle(direction,tangent),_angle(direction,-tangent))
                        if body_angle < cfg.min_body_angle_deg:
                            continue
                        distance = float(np.linalg.norm(span))
                        score = distance*(1+alpha/45)+5
                        if component not in best or score < best[component][0]:
                            best[component] = (score,i,None,_path(p,q,end['direction']),
                                               'endpoint_body',alpha,body_angle)
                    candidates.extend(best.values())
        candidates.sort(key=lambda item: (item[0],item[1],tuple(item[3][-1])))
        used = set()
        joined = 0
        for score, i, j, route, kind, alpha, beta in candidates:
            if i in used or (j is not None and j in used):
                continue
            p, q = tuple(route[0]), tuple(route[-1])
            current_labels, current_count = ndi.label(result, S8)
            if current_labels[p] == current_labels[q]:
                continue
            if len(neighbours(result,p)) != 1 or (j is not None and len(neighbours(result,q)) != 1):
                continue
            if np.any(route < 0) or np.any(route[:,0] >= result.shape[0]) or np.any(route[:,1] >= result.shape[1]):
                rejected['outside_image'] += 1
                continue
            length = float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())
            if length > cfg.max_distance:
                rejected['path_too_long'] += 1
                continue
            addition = np.zeros_like(result)
            addition[route[:,0],route[:,1]] = True
            added = addition & ~result
            if not added.any():
                continue
            contacts = ndi.binary_dilation(added,S8) & result
            yy,xx = np.nonzero(contacts)
            far = (np.hypot(yy-p[0],xx-p[1]) > 3) & (np.hypot(yy-q[0],xx-q[1]) > 3)
            if far.any():
                rejected['intermediate_branch_contact'] += 1
                continue
            mean_support = None
            longest_low = None
            if bv is not None:
                support = bv[route[:,0],route[:,1]]
                mean_support = float(np.mean(support))
                longest_low = _longest_run(support < cfg.low_bv)
                if mean_support < cfg.min_bv_mean or longest_low > cfg.max_unsupported_run:
                    rejected['insufficient_bv_support'] += 1
                    continue
            trial = result | addition
            if int(ndi.label(trial,S8)[1]) != current_count-1:
                rejected['unexpected_topology_change'] += 1
                continue
            result = trial
            used.add(i)
            if j is not None:
                used.add(j)
            joined += 1
            accepted.append({
                'kind':kind,'pass':pass_index,'route_yx':route.tolist(),
                'source_xy':[int(x) for x in p[::-1]],'target_xy':[int(x) for x in q[::-1]],
                'source_parent_xy':[int(x) for x in ends[i]['parent'][::-1]],
                'target_parent_xy':None if j is None else [int(x) for x in ends[j]['parent'][::-1]],
                'source_direction_xy':ends[i]['direction'][::-1].tolist(),
                'source_angle_deg':alpha,'target_angle_deg':beta,
                'length':length,'added_pixels':int(added.sum()),
                'mean_bv_support':mean_support,'longest_low_bv_run':longest_low,
                'components_before':int(current_count),'components_after':int(current_count-1),
            })
        if not joined:
            break
    return result, {
        'config':asdict(cfg),'support_mode':'BV_probability' if bv is not None else 'geometry_only',
        'components_before':before_count,'components_after':int(ndi.label(result,S8)[1]),
        'original_pixels_retained':bool(np.all(result[original])),
        'added_pixels':int(np.sum(result&~original)),
        'bridges':accepted,'rejected':dict(rejected),
    }
