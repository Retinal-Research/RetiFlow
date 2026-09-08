# -*- coding: utf-8 -*-
"""
RBAD_v2 端到端推理：眼底图 -> RRWNet 分割 -> v3 概率补全 -> 两遍法 RBAD 分叉检测。

用法（从图像）：
  python -m RBAD_v2.infer --image <眼底图> --weights <rrwnet权重> --out <输出目录>

用法（从已存概率图，跳过 RRWNet）：
  python -m RBAD_v2.infer --prob-dir <含 seg_probabilities.npz 或 seg_A/V/BV.png 的目录> --out <输出目录>

参数控制见 config.py，命令行可覆盖关键参数。
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

from RBAD_v2.config import Config
from RBAD_v2.completion.completion import complete_probability_masks
from RBAD_v2.completion.centerline import extract_centerline
from RBAD_v2.detect.two_pass import detect_two_pass
from RBAD_v2.detect.endpoint_bridge import EndpointBridgeConfig as EBConfig


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #
def segment(image, weights, iterations=5, thred=25):
    """RRWNet 分割，返回 (pred, enhanced_img)。pred 通道 0=A 1=V 2=BV。"""
    from rrwnet.model import RRWNet
    from rrwnet.preprocessing import enhance_image_api
    from rrwnet.utils import pad_images_unet, to_torch_tensors
    import torch

    model = RRWNet(iterations=iterations)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    img, mask = enhance_image_api(image, thred=thred)
    imgs, paddings = pad_images_unet([img, mask])
    img_pad, padding = imgs[0], paddings[0]
    roi = np.stack([imgs[1]] * 3, axis=2)
    tensors = to_torch_tensors([img_pad, roi])
    image_tensor = tensors[0].unsqueeze(0)
    mask_tensor = tensors[1].unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        mask_tensor = mask_tensor.cuda()
    with torch.no_grad():
        predictions = model(image_tensor)
        last_pred = torch.sigmoid(predictions[-1])
        last_pred[mask_tensor < 0.5] = 0
        last_pred = last_pred[:, :, padding[0][0]:-padding[0][1],
                             padding[1][0]:-padding[1][1]]
    return last_pred[0].cpu().numpy(), img


def load_probabilities(prob_dir):
    """从目录读 A/V/BV 概率图（优先 npz，回退 PNG）。"""
    prob_dir = Path(prob_dir)
    npz = prob_dir / "seg_probabilities.npz"
    if npz.is_file():
        with np.load(npz, allow_pickle=False) as a:
            return {k: np.asarray(a[k], np.float32) for k in ("A", "V", "BV")}
    pngs = {k: prob_dir / f"seg_{k}.png" for k in ("A", "V", "BV")}
    if all(p.is_file() for p in pngs.values()):
        return {k: cv2.imread(str(p), 0).astype(np.float32) / 255.0
                for k, p in pngs.items()}
    raise FileNotFoundError("no seg_probabilities.npz or seg_A/V/BV.png in " + str(prob_dir))


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


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(image=None, prob_dir=None, weights=None, out=None, config=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or Config()

    # 1. 分割
    if prob_dir is not None:
        probs = load_probabilities(prob_dir)
        enhanced = None
        print("[1/4] 从已存概率图读取")
    else:
        print("[1/4] RRWNet 分割")
        pred, enhanced = segment(image, weights, cfg.segmentation.iterations,
                                 cfg.segmentation.thred)
        probs = {"A": pred[0], "V": pred[1], "BV": pred[2]}
        if enhanced is not None:
            cv2.imwrite(str(out / "enhanced_background.png"),
                        (enhanced * 255).astype(np.uint8))
    a_prob, v_prob, bv_prob = probs["A"], probs["V"], probs["BV"]

    # 2. v3 补全
    print("[2/4] v3 概率路径补全")
    from RBAD_v2.completion.completion import ProbabilityConfig
    from dataclasses import asdict
    ccfg = ProbabilityConfig(**{k: v for k, v in asdict(cfg.completion).items()
                                if k in ProbabilityConfig.__dataclass_fields__})
    am, vm, debug = complete_probability_masks(a_prob, v_prob, bv_prob, config=ccfg)
    sa, _ = extract_centerline(bv_prob * (0.65 + 0.35 * debug["conditional_a"]), am)
    sv, _ = extract_centerline(bv_prob * (0.65 + 0.35 * (1 - debug["conditional_a"])), vm)
    for name, arr in [("A_mask", am), ("V_mask", vm),
                      ("A_centerline", sa), ("V_centerline", sv)]:
        cv2.imwrite(str(out / f"{name}.png"), arr.astype(np.uint8) * 255)

    # 3. 两遍法 RBAD（用 BV root）
    print("[3/4] 两遍法 RBAD 分叉检测（BV root）")
    from skimage.morphology import skeletonize
    bv_skel = skeletonize((bv_prob > 0.5).astype(np.uint8)).astype(bool)
    density = cv2.GaussianBlur(bv_skel.astype(np.float32), (0, 0), 21)
    bv_root = np.unravel_index(int(np.argmax(density)), density.shape)
    eb = EBConfig(**{k: v for k, v in asdict(cfg.endpoint_bridge).items()
                     if k in EBConfig.__dataclass_fields__})
    t0 = time.perf_counter()
    _, res_a = detect_two_pass(sa, bv_probability=bv_prob, bridge_config=eb,
                               root_yx=bv_root, tail=cfg.rbad.tail,
                               child_min_dist=cfg.rbad.child_min_dist,
                               angle_min=cfg.rbad.angle_min,
                               angle_max=cfg.rbad.angle_max)
    _, res_v = detect_two_pass(sv, bv_probability=bv_prob, bridge_config=eb,
                               root_yx=bv_root, tail=cfg.rbad.tail,
                               child_min_dist=cfg.rbad.child_min_dist,
                               angle_min=cfg.rbad.angle_min,
                               angle_max=cfg.rbad.angle_max)
    t_rbad = time.perf_counter() - t0
    det_a, det_v = res_a["after"]["angles"], res_v["after"]["angles"]
    print(f"  A: {len(det_a)} 分叉, V: {len(det_v)} 分叉")

    # 4. 可视化 + 摘要
    print("[4/4] 可视化 + 摘要")
    if enhanced is not None:
        bg = enhanced
    else:
        bg = cv2.cvtColor((bv_prob * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) / 255.0
    cv2.imwrite(str(out / "overlay_A_aligned.png"),
                draw_overlay(bg, sa, _to_draw(det_a), color=(0, 0, 255)))
    cv2.imwrite(str(out / "overlay_V_aligned.png"),
                draw_overlay(bg, sv, _to_draw(det_v), color=(255, 0, 0)))
    av_bv = draw_av_bv_rgb(am | vm, am, vm)
    cv2.imwrite(str(out / "AV_BV_combined.png"), cv2.cvtColor(av_bv, cv2.COLOR_RGB2BGR))

    summary = {
        "image": image, "prob_dir": str(prob_dir) if prob_dir else None,
        "config": cfg.to_dict(),
        "BV_root_yx": [int(bv_root[0]), int(bv_root[1])],
        "RBAD": {
            "A": {"count": len(det_a),
                  "mean_angle": round(float(np.mean([d["angle"] for d in det_a])), 2) if det_a else None},
            "V": {"count": len(det_v),
                  "mean_angle": round(float(np.mean([d["angle"] for d in det_v])), 2) if det_v else None},
        },
        "rbad_seconds": round(t_rbad, 3),
        "accepted_bridges": {k: len(v) for k, v in debug["bridges"].items()},
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n输出: {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--prob-dir", type=Path)
    ap.add_argument("--weights", default=str(HERE / "rrwnet_HRF_0.pth"))
    ap.add_argument("--out", required=True, type=Path)
    # 关键参数覆盖
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--max-bridge-length", type=float)
    ap.add_argument("--max-opposite-run", type=int)
    ap.add_argument("--bridge-passes", type=int)
    ap.add_argument("--max-distance", type=float)
    ap.add_argument("--max-angle", type=float)
    ap.add_argument("--passes", type=int)
    ap.add_argument("--tail", type=int)
    args = ap.parse_args()

    if args.image is None and args.prob_dir is None:
        ap.error("must provide --image or --prob-dir")

    cfg = Config()
    if args.iterations is not None:
        cfg.segmentation.iterations = args.iterations
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

    run(image=str(args.image) if args.image else None,
        prob_dir=str(args.prob_dir) if args.prob_dir else None,
        weights=args.weights, out=str(args.out), config=cfg)


if __name__ == "__main__":
    main()
