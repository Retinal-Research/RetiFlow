# -*- coding: utf-8 -*-
"""
AutoMorph 输出适配插件。

把 AutoMorph M2 的输出格式适配到 RetiFlow 的输入。

AutoMorph M2 目录结构（Results/M2/）：
  artery_vein/
    artery_binary_process/   动脉二值掩码 (912x912, 0/255)
    artery_binary_skeleton/ 动脉骨架
    vein_binary_process/    静脉二值掩码
    vein_binary_skeleton/   静脉骨架
    ... (disc/macular centred, Zone_B/C 变体)
  binary_vessel/
    binary_process/          血管二值掩码
    binary_skeleton/         血管骨架
  optic_disc_cup/
    raw/                     视盘/杯 (1890x1890, R=disc, B=cup)

用法：
  from RetiFlow.plugins.automorph import AutomorphAdapter
  adapter = AutomorphAdapter(".../Results/M2")
  adapter.run(out_dir=".../retiflow_out", mode="skeleton", use_disc_root=True)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from RetiFlow.infer import process_single, process_prob, _angle_stats, plot_centroid
from RetiFlow.config import Config
from RetiFlow.detect.centroid import binary_centroid, gaussian_center, vote_centroid


def _resize_av3(a, v, bv, target):
    """把 A/V/BV 缩放到 target 边长（最长边）。"""
    h, w = a.shape[:2]
    scale = target / float(max(h, w))
    if scale >= 1.0:
        return a, v, bv
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    a = cv2.resize(a.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    if v is not None:
        v = cv2.resize(v.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    if bv is not None:
        bv = cv2.resize(bv.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    return a, v, bv


class AutomorphAdapter:
    """把 AutoMorph M2 输出适配到 RetiFlow。"""

    def __init__(self, m2_dir):
        self.m2_dir = Path(m2_dir)
        self.av_dir = self.m2_dir / "artery_vein"
        self.bv_dir = self.m2_dir / "binary_vessel"
        self.dc_dir = self.m2_dir / "optic_disc_cup"
        if not self.av_dir.is_dir():
            raise FileNotFoundError(f"no artery_vein dir in {m2_dir}")

    # -- 发现 ------------------------------------------------------------- #
    def list_images(self):
        """返回所有图像名（基于动脉骨架目录）。"""
        skel_dir = self.av_dir / "artery_binary_skeleton"
        return sorted(p.stem for p in skel_dir.glob("*.png"))

    # -- 读取 ------------------------------------------------------------- #
    def _read(self, subdir, name, sub="artery_vein"):
        base = self.av_dir if sub == "artery_vein" else self.bv_dir
        p = base / subdir / f"{name}.png"
        if not p.is_file():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return None if img is None else (img > 0).astype(bool)

    def get_skeletons(self, name):
        """返回 (a_skel, v_skel) 动脉/静脉骨架。"""
        a = self._read("artery_binary_skeleton", name)
        v = self._read("vein_binary_skeleton", name)
        return a, v

    def get_masks(self, name):
        """返回 (a_mask, v_mask) 动脉/静脉二值掩码。"""
        a = self._read("artery_binary_process", name)
        v = self._read("vein_binary_process", name)
        return a, v

    def get_av3(self, name):
        """返回 AV3 概率图 (a, v, bv)，由分离的 A/V/BV 文件合并（R=A, G=V, B=BV）。

        注意：artery_vein/raw 的通道不相交，不是标准 AV3；用分离文件更可靠。
        """
        a = self._read("artery_binary_process", name)
        v = self._read("vein_binary_process", name)
        bv = self._read("binary_process", name, sub="binary_vessel")
        if a is None or v is None or bv is None:
            return None
        # 合并成 AV3（R=A, G=V, B=BV），0/1 概率
        av3 = np.zeros((*a.shape, 3), dtype=np.float32)
        av3[..., 2] = a.astype(np.float32)   # R = A
        av3[..., 1] = v.astype(np.float32)   # G = V
        av3[..., 0] = bv.astype(np.float32)  # B = BV
        return av3[..., 2], av3[..., 1], av3[..., 0]

    def get_vessel_skeleton(self, name):
        """返回血管骨架（binary_vessel）。"""
        return self._read("binary_skeleton", name, sub="binary_vessel")

    def get_disc_center(self, name, robust=True):
        """返回视盘质心归一化比例 (dy, dx)，来自 optic_disc_cup/raw 的 R 通道。

        返回 [0,1] 的比例（相对 disc mask 分辨率），分辨率无关，任何分辨率都能对齐。
        """
        p = self.dc_dir / "raw" / f"{name}.png"
        if not p.is_file():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        c = binary_centroid(img[..., 2] > 0, robust=robust)  # R 通道 = disc
        if c is None:
            return None
        return (c[0] / img.shape[0], c[1] / img.shape[1])  # (dy, dx) 比例

    def get_cup_center(self, name, robust=True):
        """返回视杯质心归一化比例 (dy, dx)，来自 optic_disc_cup/raw 的 B 通道。

        cup 比 disc 小，用更小的 min_area_frac 避免被当噪声过滤。
        返回 [0,1] 的比例（相对 disc mask 分辨率）。
        """
        p = self.dc_dir / "raw" / f"{name}.png"
        if not p.is_file():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        c = binary_centroid(img[..., 0] > 0, robust=robust,
                            min_area_frac=0.001)  # B 通道 = cup
        if c is None:
            return None
        return (c[0] / img.shape[0], c[1] / img.shape[1])  # (dy, dx) 比例

    def get_gaussian_center(self, skeleton):
        """返回高斯密度质心 (y, x)。"""
        return gaussian_center(skeleton)

    # -- 运行 ------------------------------------------------------------- #
    def run(self, out_dir, mode="skeleton", use_disc_root=True, config=None,
            root_yx=None, resize=None, gradable_csv=None, limit=None):
        """对每张图跑 RetiFlow 分叉检测。

        mode: 'prob'（喂 AV3 raw 给完整流水线，用补全）
              'skeleton'（用 AutoMorph 骨架）或 'mask'（用掩码提取中心线）。
        use_disc_root: 用视盘质心作为 root。
        root_yx: 手动指定 root (y, x)，覆盖视盘质心。
        resize: 若指定，把输入缩放到该边长（如 512），加速处理。
        gradable_csv: M1_quality_final.csv 路径，只处理 Gradable=True 的图像。
        limit: 若指定，只处理前 N 张（调试用）。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = config or Config()
        names = self.list_images()
        if gradable_csv:
            import csv
            with open(gradable_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            gradable = {Path(r["Name"]).stem for r in rows
                        if r["Gradable"].strip().lower() == "true"}
            names = [n for n in names if n in gradable]
            print(f"gradable 图像: {len(names)} / {len(self.list_images())}")
        if limit:
            names = names[:limit]
            print(f"limit: 只处理前 {len(names)} 张")
        results = []
        for i, name in enumerate(tqdm(names, desc="RetiFlow-Angle", unit="img")):
            sub = out_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            try:
                # 先拿 A/V（得到分辨率），再算 disc/cup（需缩放到该分辨率）
                gauss_center = None
                if mode == "prob":
                    av3 = self.get_av3(name)
                    if av3 is None:
                        raise FileNotFoundError(f"missing AV3 raw for {name}")
                    a, v, bv = av3
                    if resize:
                        a, v, bv = _resize_av3(a, v, bv, resize)
                    gauss_center = self.get_gaussian_center((a > 0.5) | (v > 0.5))
                else:
                    if mode == "skeleton":
                        a, v = self.get_skeletons(name)
                    else:
                        a, v = self.get_masks(name)
                    if a is None or v is None:
                        raise FileNotFoundError(f"missing A/V for {name}")
                    if resize:
                        a, v = _resize_av3(a, v, None, resize)[:2]
                    # 高斯质心用整个血管（A|V），不是只用 A
                    gauss_center = self.get_gaussian_center((a > 0) | (v > 0))
                # disc/cup 质心是归一化比例 (dy, dx)，gauss 是像素 → 转比例后投票
                disc_ratio = self.get_disc_center(name, robust=True) if use_disc_root else None
                cup_ratio = self.get_cup_center(name, robust=True) if use_disc_root else None
                gauss_ratio = None
                if gauss_center is not None:
                    gauss_ratio = (gauss_center[0] / a.shape[0], gauss_center[1] / a.shape[1])
                # 只有 mask（无视盘）时，只算高斯
                voted_ratio = vote_centroid([disc_ratio, cup_ratio, gauss_ratio])
                if voted_ratio is not None:
                    r = (voted_ratio[0] * a.shape[0], voted_ratio[1] * a.shape[1])
                else:
                    r = None
                r = root_yx or r

                # 处理
                if mode == "prob":
                    res = process_prob(a, v, bv, cfg, sub, r)
                    res_a, res_v = res["RBAD"]["A"], res["RBAD"]["V"]
                    bg = cv2.cvtColor((bv * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) / 255.0
                else:
                    res_a = process_single(a, cfg, sub / "A", r, is_mask=(mode == "mask"))
                    res_v = process_single(v, cfg, sub / "V", r, is_mask=(mode == "mask"))
                    res_a, res_v = res_a["RBAD"], res_v["RBAD"]
                    bg = cv2.cvtColor((a.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB) / 255.0

                # 质心候选投票图（disc / cup / gauss / voted），比例转像素
                disc_px = (disc_ratio[0] * a.shape[0], disc_ratio[1] * a.shape[1]) if disc_ratio else None
                cup_px = (cup_ratio[0] * a.shape[0], cup_ratio[1] * a.shape[1]) if cup_ratio else None
                plot_centroid(
                    bg,
                    [disc_px, cup_px, gauss_center, r],
                    sub / "centroid.png",
                    labels=["disc", "cup", "gauss", "voted"],
                )

                entry = {
                    "image": name,
                    "mode": mode,
                    "root_yx": [int(r[0]), int(r[1])] if r else None,
                    "disc_ratio": [round(disc_ratio[0], 4), round(disc_ratio[1], 4)]
                                   if disc_ratio else None,
                    "cup_ratio": [round(cup_ratio[0], 4), round(cup_ratio[1], 4)]
                                  if cup_ratio else None,
                    "gauss_center": [int(gauss_center[0]), int(gauss_center[1])]
                                     if gauss_center else None,
                    "voted_center": [int(r[0]), int(r[1])] if r else None,
                    "root_source": "manual" if root_yx else "vote",
                    "A": res_a,
                    "V": res_v,
                }
                results.append(entry)
                with (sub / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({"image": name, "error": str(e)})

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"m2_dir": str(self.m2_dir), "mode": mode,
                       "n": len(results), "results": results},
                      f, indent=2, ensure_ascii=False)
        return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("prob", "skeleton", "mask"), default="skeleton")
    ap.add_argument("--resize", type=int, default=None, help="缩放到该边长（如 512）加速")
    ap.add_argument("--gradable-csv", type=Path, default=None,
                    help="M1_quality_final.csv，只处理 Gradable=True 的图像")
    ap.add_argument("--no-disc-root", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（调试用）")
    args = ap.parse_args()
    adapter = AutomorphAdapter(args.m2_dir)
    adapter.run(args.out, mode=args.mode, use_disc_root=not args.no_disc_root,
                resize=args.resize, gradable_csv=args.gradable_csv, limit=args.limit)
