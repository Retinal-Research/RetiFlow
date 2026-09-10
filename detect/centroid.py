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
    """交叉骨（CrossBone）—— 交叉点高斯密度质心，RetiFlow 核心优势。

    原理：血管在视盘（root）处最密集，交叉点（分叉）密度最高的位置即视盘。

    方法：
      1. 骨架化输入 mask。
      2. 找交叉点：3x3 邻域骨架邻居数 >=3 的像素（纯局部计数，不跑 RBAD）。
      3. 把交叉点做成稀疏图，高斯模糊 —— 等价于每个交叉点生成一个高斯再求和。
      4. 密度峰值位置即 root。

    相比 RBAD 论文方法（用 fast_keypoints 找所有关键点再生成高斯）：
      - 不依赖 root：局部计数不需要先知道 root（避免鸡生蛋问题）。
      - 更快：一次卷积就找到所有交叉点，不用跑完整 RBAD 遍历。
      - 对断点免疫：逐像素局部判断，断点不影响其他交叉点。

    多数情况下非常准，是 RetiFlow 的核心优势。

    Returns
    -------
    (y, x) 质心。
    """
    from skimage.morphology import skeletonize
    skel = skeletonize((skeleton > 0).astype(np.uint8)).astype(bool)
    # 交叉点：3x3 邻域骨架邻居 >=3
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    degree = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    junctions = skel & (degree >= 3)
    if not np.any(junctions):
        # 无交叉点 → fallback 到整个骨架密度
        density = cv2.GaussianBlur(skel.astype(np.float32), (0, 0), sigma)
        return np.unravel_index(int(np.argmax(density)), density.shape)
    # 交叉点稀疏图 + 高斯模糊 = 每个交叉点的高斯求和
    sparse = junctions.astype(np.float32)
    density = cv2.GaussianBlur(sparse, (0, 0), sigma)
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
