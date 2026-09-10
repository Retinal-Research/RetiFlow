# -*- coding: utf-8 -*-
"""
RetiFlow — 视网膜分叉角度检测（独立于分割网络）。

RetiFlow 只消费"血管图"，不做分割。输入可以是概率图、二值掩码或骨架，
给什么都能跑，缺什么就降级。

输入模式（单张）：
  --prob <3通道图>       A/V/BV 概率图（R=A, G=V, B=BV，参考 rrwnet AV3 约定）
  --mask <二值图>        单张二值掩码（无 A/V 之分）
  --skeleton <二值图>    单张骨架（无 A/V 之分）

批量模式：
  --input-dir <目录> --input-type <prob|mask|skeleton>
  处理目录下所有文件，每个文件是一张图的输入。

可选：
  --root-x <x> --root-y <y>   指定视盘/杯质心作为 root（比高斯密度启发式更准）

输出（越全面越好）：
  prob: 补全掩码 + 中心线 + A/V overlay + 组合图 + summary.json
  mask/skeleton: 中心线 + overlay + summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from RetiFlow.config import Config
from RetiFlow.completion.completion import complete_probability_masks, ProbabilityConfig
from RetiFlow.completion.centerline import extract_centerline
from RetiFlow.detect.two_pass import detect_two_pass
from RetiFlow.detect.endpoint_bridge import EndpointBridgeConfig as EBConfig


# --------------------------------------------------------------------------- #
# 输入读取
# --------------------------------------------------------------------------- #
def read_prob_image(path):
    """读 3 通道概率图（R=A, G=V, B=BV），返回 [0,1] 的 (a, v, bv)。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    a = img[..., 2].astype(np.float32) / 255.0   # R
    v = img[..., 1].astype(np.float32) / 255.0   # G
    bv = img[..., 0].astype(np.float32) / 255.0  # B
    return a, v, bv


def read_binary(path):
    """读二值图（掩码/骨架），返回 bool 数组。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    return (img > 0).astype(bool)


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def draw_overlay(background, skeleton, detections, color=(0, 0, 255), alpha=0.6):
    """背景 + 骨架 + 分叉点 + 两条子支箭头 + 角度。返回 BGR。"""
    bgr = (background * 255).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[skeleton] = color
    out = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    L = 18
    for d in detections:
        x, y = d["points"]
        cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(out, f"{d['angle']:.0f}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        for dx, dy in d.get("daughters", []):
            ex, ey = int(round(x + dx * L)), int(round(y + dy * L))
            cv2.arrowedLine(out, (x, y), (ex, ey), (0, 255, 0), 1,
                            cv2.LINE_AA, tipLength=0.3)
    return out


def draw_av_bv_rgb(bv, a, v):
    """原版 rrwnet 映射：R=A, G=V, B=BV。A=粉, V=青, BV=蓝, 交叉=白。返回 RGB。"""
    rgb = np.zeros((*bv.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (a > 0).astype(np.uint8) * 255
    rgb[..., 1] = (v > 0).astype(np.uint8) * 255
    rgb[..., 2] = (bv > 0).astype(np.uint8) * 255
    return rgb


def plot_centroid(bg, points, out_path, labels=None, colors=None):
    """把质心点画到背景图上，保存。

    Parameters
    ----------
    bg : HxWx3 背景图（BGR 或 RGB，uint8 或 [0,1]）。
    points : [(y, x), ...] 质心坐标。
    out_path : 保存路径。
    labels : [str, ...] 每个点的标签（可选）。
    colors : [BGR, ...] 每个点的颜色（可选）。
    """
    bgr = (bg * 255).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    canvas = bgr.copy()
    default_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255),
                      (255, 0, 255), (255, 255, 0)]
    for i, point in enumerate(points):
        if point is None:
            continue
        y, x = point
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, color, -1)
        cv2.drawMarker(canvas, (int(round(x)), int(round(y))), color,
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        if labels and i < len(labels):
            cv2.putText(canvas, str(labels[i]), (int(round(x)) + 8, int(round(y)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def _to_draw(angles):
    """把两遍法 angles 转成 draw_overlay 需要的格式（含 daughters 方向）。"""
    out = []
    for a in angles:
        x, y = a["points"]
        daughters = []
        for c in a["children"]:
            dx, dy = c[0] - x, c[1] - y
            norm = math.hypot(dx, dy)
            if norm > 1e-8:
                daughters.append([dx / norm, dy / norm])
        out.append({"points": a["points"], "angle": a["angle"],
                    "daughters": daughters})
    return out


def _angle_stats(angles):
    if not angles:
        return {"count": 0, "mean_angle": None}
    return {"count": len(angles),
            "mean_angle": round(float(np.mean([d["angle"] for d in angles])), 2)}


# --------------------------------------------------------------------------- #
# 核心处理
# --------------------------------------------------------------------------- #
def _root_from_bv(bv_prob):
    from skimage.morphology import skeletonize
    bv_skel = skeletonize((bv_prob > 0.5).astype(np.uint8)).astype(bool)
    density = cv2.GaussianBlur(bv_skel.astype(np.float32), (0, 0), 21)
    return np.unravel_index(int(np.argmax(density)), density.shape)


def _bridge_config(cfg):
    from dataclasses import asdict
    return EBConfig(**{k: v for k, v in asdict(cfg.endpoint_bridge).items()
                       if k in EBConfig.__dataclass_fields__})


def _rbad_kwargs(cfg):
    return dict(tail=cfg.rbad.tail, child_min_dist=cfg.rbad.child_min_dist,
                angle_min=cfg.rbad.angle_min, angle_max=cfg.rbad.angle_max)


def process_prob(a, v, bv, cfg, out, root_yx=None):
    """概率图输入：补全 + 中心线 + 两遍法 RBAD（带 A/V 归属）。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    ccfg = ProbabilityConfig(**{k: v for k, v in asdict(cfg.completion).items()
                                if k in ProbabilityConfig.__dataclass_fields__})
    am, vm, debug = complete_probability_masks(a, v, bv, config=ccfg)
    sa, _ = extract_centerline(bv * (0.65 + 0.35 * debug["conditional_a"]), am)
    sv, _ = extract_centerline(bv * (0.65 + 0.35 * (1 - debug["conditional_a"])), vm)
    for name, arr in [("A_mask", am), ("V_mask", vm),
                      ("A_centerline", sa), ("V_centerline", sv)]:
        cv2.imwrite(str(out / f"{name}.png"), arr.astype(np.uint8) * 255)

    if root_yx is None:
        root_yx = _root_from_bv(bv)
    eb = _bridge_config(cfg)
    kw = _rbad_kwargs(cfg)
    t0 = time.perf_counter()
    _, res_a = detect_two_pass(sa, bv_probability=bv, bridge_config=eb,
                               root_yx=root_yx, **kw)
    _, res_v = detect_two_pass(sv, bv_probability=bv, bridge_config=eb,
                               root_yx=root_yx, **kw)
    t_rbad = time.perf_counter() - t0
    det_a, det_v = res_a["after"]["angles"], res_v["after"]["angles"]

    bg = cv2.cvtColor((bv * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) / 255.0
    plot_centroid(bg, [root_yx], out / "root.png", labels=["root"])
    cv2.imwrite(str(out / "overlay_A_aligned.png"),
                draw_overlay(bg, sa, _to_draw(det_a), color=(0, 0, 255)))
    cv2.imwrite(str(out / "overlay_V_aligned.png"),
                draw_overlay(bg, sv, _to_draw(det_v), color=(255, 0, 0)))
    av_bv = draw_av_bv_rgb(am | vm, am, vm)
    cv2.imwrite(str(out / "AV_BV_combined.png"), cv2.cvtColor(av_bv, cv2.COLOR_RGB2BGR))

    return {
        "mode": "prob",
        "root_yx": [int(root_yx[0]), int(root_yx[1])],
        "RBAD": {
            "A": {**_angle_stats(det_a), "angles": [round(d["angle"], 3) for d in det_a]},
            "V": {**_angle_stats(det_v), "angles": [round(d["angle"], 3) for d in det_v]},
        },
        "rbad_seconds": round(t_rbad, 3),
    }


def process_single(skel, cfg, out, root_yx=None, is_mask=False):
    """单张掩码/骨架输入：中心线（若掩码）+ 两遍法 RBAD（无 A/V 归属）。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if is_mask:
        from skimage.morphology import skeletonize
        skel = skeletonize(skel.astype(np.uint8)).astype(bool)
        cv2.imwrite(str(out / "centerline.png"), skel.astype(np.uint8) * 255)
    eb = _bridge_config(cfg)
    kw = _rbad_kwargs(cfg)
    t0 = time.perf_counter()
    _, res = detect_two_pass(skel, bv_probability=None, bridge_config=eb,
                             root_yx=root_yx, **kw)
    t_rbad = time.perf_counter() - t0
    det = res["after"]["angles"]

    bg = np.zeros((*skel.shape, 3), dtype=np.uint8)
    if root_yx is not None:
        plot_centroid(bg, [root_yx], out / "root.png", labels=["root"])
    cv2.imwrite(str(out / "overlay.png"),
                draw_overlay(bg, skel, _to_draw(det), color=(0, 0, 255)))

    return {
        "mode": "mask" if is_mask else "skeleton",
        "root_yx": [int(res["after"]["root_yx"][0]), int(res["after"]["root_yx"][1])]
                   if res["after"]["root_yx"] else None,
        "RBAD": {**_angle_stats(det), "angles": [round(d["angle"], 3) for d in det]},
        "rbad_seconds": round(t_rbad, 3),
    }


def run_single(input_path, input_type, cfg, out, root_yx=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if input_type == "prob":
        a, v, bv = read_prob_image(input_path)
        summary = process_prob(a, v, bv, cfg, out, root_yx)
    elif input_type == "mask":
        skel = read_binary(input_path)
        summary = process_single(skel, cfg, out, root_yx, is_mask=True)
    elif input_type == "skeleton":
        skel = read_binary(input_path)
        summary = process_single(skel, cfg, out, root_yx, is_mask=False)
    else:
        raise ValueError(f"unknown input_type: {input_type}")
    summary["input"] = str(input_path)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_batch(input_dir, input_type, cfg, out, root_yx=None):
    input_dir = Path(input_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".npz")
    files = sorted(f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no input files in {input_dir}")
    results = []
    for f in files:
        print(f"[{files.index(f)+1}/{len(files)}] {f.name}")
        sub = out / f.stem
        try:
            summary = run_single(f, input_type, cfg, sub, root_yx)
            results.append(summary)
        except Exception as e:
            results.append({"input": str(f), "error": str(e)})
            print(f"  ERROR: {e}")
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input_dir": str(input_dir), "input_type": input_type,
                   "n": len(results), "results": results},
                  f, indent=2, ensure_ascii=False)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prob", type=Path, help="单张 3 通道 A/V/BV 概率图")
    g.add_argument("--mask", type=Path, help="单张二值掩码")
    g.add_argument("--skeleton", type=Path, help="单张骨架")
    g.add_argument("--input-dir", type=Path, help="批量输入目录")
    ap.add_argument("--input-type", choices=("prob", "mask", "skeleton"),
                    help="批量输入类型（--input-dir 时必填）")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root-x", type=float)
    ap.add_argument("--root-y", type=float)
    # 参数覆盖
    ap.add_argument("--max-bridge-length", type=float)
    ap.add_argument("--max-opposite-run", type=int)
    ap.add_argument("--bridge-passes", type=int)
    ap.add_argument("--max-distance", type=float)
    ap.add_argument("--max-angle", type=float)
    ap.add_argument("--passes", type=int)
    ap.add_argument("--tail", type=int)
    args = ap.parse_args()

    if args.input_dir is not None and args.input_type is None:
        ap.error("--input-dir requires --input-type")
    if (args.root_x is None) != (args.root_y is None):
        ap.error("--root-x and --root-y must be provided together")

    cfg = Config()
    if args.max_bridge_length is not None:
        cfg.completion.max_bridge_length = args.max_bridge_length
    if args.max_opposite_run is not None:
        cfg.completion.max_opposite_run = args.max_opposite_run
    if args.bridge_passes is not None:
        cfg.completion.bridge_passes = args.bridge_passes
    if args.max_distance is not None:
        cfg.endpoint_bridge.max_distance = args.max_distance
    if args.max_angle is not None:
        cfg.endpoint_bridge.max_angle_deg = args.max_angle
    if args.passes is not None:
        cfg.endpoint_bridge.passes = args.passes
    if args.tail is not None:
        cfg.rbad.tail = args.tail

    root_yx = (args.root_y, args.root_x) if args.root_x is not None else None

    if args.input_dir is not None:
        run_batch(args.input_dir, args.input_type, cfg, args.out, root_yx)
    elif args.prob is not None:
        run_single(args.prob, "prob", cfg, args.out, root_yx)
    elif args.mask is not None:
        run_single(args.mask, "mask", cfg, args.out, root_yx)
    elif args.skeleton is not None:
        run_single(args.skeleton, "skeleton", cfg, args.out, root_yx)


if __name__ == "__main__":
    main()
