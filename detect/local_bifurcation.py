# -*- coding: utf-8 -*-
"""
局部分叉检测 —— 不依赖全局连通路径。

原理：
  分叉点检测本质是"局部"的：一个骨架像素是不是分叉点，只看它 3x3 邻域里
  有几个骨架邻居（>=3 即分叉）。因此即使血管骨架断成很多段，每一段内部的
  分叉点照样能找出来。

  本实现：
    1. 骨架化
    2. 逐像素数 3x3 邻居，>=3 的标记为分叉像素
    3. 相邻分叉像素聚类（同一物理分叉的多个像素合并为一个）
    4. 对每个分叉，沿各分支固定走 tail 像素（遇分叉走最直续接），得分支方向
    5. 合并方向相近的分支，计算最张开两分支的夹角，过滤后返回

  与 RBAD 原版 fast_keypoints 的区别：
    原版从单个 root 做全局 BFS 遍历，任何不连到 root 的分量都会被漏掉；
    本实现逐分量处理，对断点免疫。
"""
import math
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label

# 8 邻域偏移
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]


def _walk_fixed(skel, junction, start, tail):
    """从 start 沿骨架走，避开 junction，固定走 tail 步。
    遇分叉时选最直续接方向。返回走过的 (y,x) 位置列表。"""
    path = [start]
    cur = start
    prev_p = junction
    for _ in range(tail - 1):
        nexts = []
        for dy, dx in _OFFSETS:
            ny, nx = cur[0] + dy, cur[1] + dx
            if (ny, nx) == prev_p:
                continue
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                nexts.append((ny, nx))
        if not nexts:
            break
        # 选最直续接方向（与来向点积最大）
        d_prev = (cur[0] - prev_p[0], cur[1] - prev_p[1])
        best, best_score = None, -float("inf")
        for nxt in nexts:
            d_nxt = (nxt[0] - cur[0], nxt[1] - cur[1])
            score = d_prev[0] * d_nxt[0] + d_prev[1] * d_nxt[1]
            if score > best_score:
                best_score, best = score, nxt
        path.append(best)
        prev_p, cur = cur, best
    return path


def _angle_between(v1, v2):
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _dedup_branches(branches, min_angle=30.0):
    """合并方向相近的分支（厚分叉处多个邻居可能属于同一分支）。"""
    kept = []
    for v in branches:
        if any((a := _angle_between(v, k)) is not None and a < min_angle
               for k in kept):
            continue
        kept.append(v)
    return kept


def _unit(v):
    norm = math.hypot(v[0], v[1])
    if norm < 1e-8:
        return None
    return (v[0] / norm, v[1] / norm)


def _identify_parent(branches):
    """识别主干（parent）分支。

    分叉有方向：从主干到两条子支。主干 = 与两条子支角平分线最相反的那条
    （即"来向"，类似树根）。返回 (parent_idx, [daughter_idx, daughter_idx])。
    """
    n = len(branches)
    if n < 3:
        return None, None
    best_p, best_score = None, -float("inf")
    for k in range(n):
        others = [i for i in range(n) if i != k]
        d1, d2 = branches[others[0]], branches[others[1]]
        bisector = (d1[0] + d2[0], d1[1] + d2[1])
        bnorm = math.hypot(bisector[0], bisector[1])
        if bnorm < 1e-8:
            continue
        bisector = (bisector[0] / bnorm, bisector[1] / bnorm)
        # 主干应指向子支角平分线的反方向（dot 接近 -1）
        score = -(branches[k][0] * bisector[0] + branches[k][1] * bisector[1])
        if score > best_score:
            best_score, best_p = score, k
    if best_p is None:
        return None, None
    return best_p, [i for i in range(n) if i != best_p]


def detect_bifurcations(bin_mask, tail=15, angle_min=20.0, angle_max=120.0,
                        dedup_angle=30.0, already_skeletonized=False):
    """局部分叉检测。

    Parameters
    ----------
    bin_mask : HxW 二值血管掩码（uint8/bool，0/255 均可）。
    tail : 沿分支固定走的像素数（角度估计的稳定度）。
    angle_min/angle_max : 保留的角度范围（度）。
    dedup_angle : 合并方向相近分支的角度阈值。

    Returns
    -------
    angle_list : [{'points':[x,y], 'angle':deg, 'parent':[dx,dy],
                   'daughters':[[dx,dy],[dx,dy]], 'branches':[[dx,dy],...]}, ...]
        parent 是主干单位方向向量（指向分叉点，即来向）；
        daughters 是两条子支单位方向向量（从分叉点指出）；
        angle 是两条子支的夹角（分叉角）。
    """
    if already_skeletonized:
        skel = (bin_mask > 0).astype(np.uint8)
    else:
        skel = skeletonize((bin_mask > 0).astype(np.uint8)).astype(np.uint8)

    # 分叉像素：3x3 邻域骨架邻居 >=3
    junc_mask = np.zeros_like(skel, dtype=bool)
    for y, x in zip(*np.nonzero(skel)):
        n = sum(1 for dy, dx in _OFFSETS
                if 0 <= y + dy < skel.shape[0] and 0 <= x + dx < skel.shape[1]
                and skel[y + dy, x + dx])
        if n >= 3:
            junc_mask[y, x] = True

    # 相邻分叉像素聚类（同一物理分叉）
    clusters = label(junc_mask, connectivity=2)

    angle_list = []
    for comp_id in range(1, clusters.max() + 1):
        ys, xs = np.nonzero(clusters == comp_id)
        # 收集该分叉所有分支方向
        branches = []
        for y, x in zip(ys, xs):
            for dy, dx in _OFFSETS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]):
                    continue
                if not skel[ny, nx]:
                    continue
                path = _walk_fixed(skel, (y, x), (ny, nx), tail)
                if len(path) < 2:
                    continue
                end = path[-1]
                branches.append((end[1] - x, end[0] - y))

        branches = _dedup_branches(branches, dedup_angle)
        if len(branches) < 3:
            continue  # 需要主干 + 两条子支

        # 单位方向向量
        units = [_unit(b) for b in branches]
        if any(u is None for u in units):
            continue
        units = [u for u in units if u is not None]

        # 识别主干（parent）和两条子支（daughters）
        parent_idx, daughter_idx = _identify_parent(units)
        if parent_idx is None or len(daughter_idx) != 2:
            continue

        # 分叉角 = 两条子支的夹角
        angle = _angle_between(units[daughter_idx[0]], units[daughter_idx[1]])
        if angle is None:
            continue
        if angle_min <= angle <= angle_max:
            cx = int(round(np.mean(xs)))
            cy = int(round(np.mean(ys)))
            angle_list.append({
                "points": [cx, cy],
                "angle": float(angle),
                "parent": [round(units[parent_idx][0], 4),
                           round(units[parent_idx][1], 4)],
                "daughters": [[round(units[i][0], 4), round(units[i][1], 4)]
                              for i in daughter_idx],
                "branches": [[round(u[0], 4), round(u[1], 4)] for u in units],
            })

    return angle_list
