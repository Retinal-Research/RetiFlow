# -*- coding: utf-8 -*-
"""
质心计算与共识投票 —— 独立可复用模块。

提供：
  binary_centroid : 二值掩码质心（可选最大连通分量，抗噪声）
  gaussian_center : 骨架高斯密度质心
  vote_centroid   : 多候选质心共识投票（防止单个离群点带偏）

不依赖任何分割网络或特定工作流，任何上游输出都能用。
"""
from __future__ import annotations

import math

import numpy as np
import cv2


def binary_centroid(mask, robust=True, min_area_frac=0.005):
    """二值掩码质心。

    Parameters
    ----------
    mask : HxW 二值掩码（bool/uint8，或 3 通道取通道 0）。
    robust : 若 True，用最大连通分量的质心（排除噪声碎片）。
    min_area_frac : robust 模式下，最大分量面积低于图像面积的该比例则视为
                    不可靠，返回 None。

    Returns
    -------
    (y, x) 质心，或 None（掩码为空/不可靠）。
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    if not np.any(mask):
        return None
    if not robust:
        ys, xs = np.nonzero(mask)
        return (float(np.mean(ys)), float(np.mean(xs)))
    # 最大连通分量
    from scipy import ndimage as ndi
    labels, count = ndi.label(mask, structure=np.ones((3, 3), bool))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    largest = int(np.argmax(sizes[1:])) + 1
    if sizes[largest] < min_area_frac * mask.size:
        return None
    ys, xs = np.nonzero(labels == largest)
    return (float(np.mean(ys)), float(np.mean(xs)))


def gaussian_center(skeleton, sigma=21.0):
    """骨架高斯密度质心。

    对骨架做高斯模糊，取密度峰值位置。作为 disc 质心的 fallback/交叉对比。

    Returns
    -------
    (y, x) 质心。
    """
    from skimage.morphology import skeletonize
    skel = skeletonize((skeleton > 0).astype(np.uint8)).astype(bool)
    density = cv2.GaussianBlur(skel.astype(np.float32), (0, 0), sigma)
    return np.unravel_index(int(np.argmax(density)), density.shape)


def vote_centroid(candidates, threshold=50.0):
    """共识投票：从多个质心候选中选一个可靠的。

    Parameters
    ----------
    candidates : [(y,x), ...] 或含 None。
    threshold : 两候选视为"共识"的最大距离（像素）。

    Returns
    -------
    (y, x) 或 None。

    逻辑：
      1. 找两两距离最近的一对。
      2. 若距离 <= threshold，用这对的均值（共识）。
      3. 否则用所有有效候选的中位数（对离群点鲁棒）。
    """
    valid = [c for c in candidates if c is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    best_pair, best_dist = None, float("inf")
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            d = math.hypot(valid[i][0] - valid[j][0],
                           valid[i][1] - valid[j][1])
            if d < best_dist:
                best_dist, best_pair = d, (i, j)
    if best_dist <= threshold:
        i, j = best_pair
        return ((valid[i][0] + valid[j][0]) / 2,
                (valid[i][1] + valid[j][1]) / 2)
    # 无共识 → 中位数
    ys = sorted(c[0] for c in valid)
    xs = sorted(c[1] for c in valid)
    return (ys[len(ys) // 2], xs[len(xs) // 2])
