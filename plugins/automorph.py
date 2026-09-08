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

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))

from RetiFlow.infer import process_single, _angle_stats
from RetiFlow.config import Config
from RetiFlow.detect.centroid import binary_centroid, gaussian_center, vote_centroid


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

    def get_vessel_skeleton(self, name):
        """返回血管骨架（binary_vessel）。"""
        return self._read("binary_skeleton", name, sub="binary_vessel")

    def get_disc_center(self, name, robust=True):
        """返回视盘质心 (y, x)，来自 optic_disc_cup/raw 的 R 通道。"""
        p = self.dc_dir / "raw" / f"{name}.png"
        if not p.is_file():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return binary_centroid(img[..., 2] > 0, robust=robust)  # R 通道 = disc

    def get_gaussian_center(self, skeleton):
        """返回高斯密度质心 (y, x)。"""
        return gaussian_center(skeleton)

    # -- 运行 ------------------------------------------------------------- #
    def run(self, out_dir, mode="skeleton", use_disc_root=True, config=None,
            root_yx=None):
        """对每张图跑 RetiFlow 分叉检测。

        mode: 'skeleton'（用 AutoMorph 骨架）或 'mask'（用掩码提取中心线）。
        use_disc_root: 用视盘质心作为 root。
        root_yx: 手动指定 root (y, x)，覆盖视盘质心。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = config or Config()
        names = self.list_images()
        results = []
        for i, name in enumerate(names):
            print(f"[{i+1}/{len(names)}] {name}")
            sub = out_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            try:
                if mode == "skeleton":
                    a, v = self.get_skeletons(name)
                else:
                    a, v = self.get_masks(name)
                if a is None or v is None:
                    raise FileNotFoundError(f"missing A/V for {name}")

                # root：三个质心候选投票，防止单个离谱
                disc_robust = self.get_disc_center(name, robust=True) if use_disc_root else None
                disc_simple = self.get_disc_center(name, robust=False) if use_disc_root else None
                gauss_center = self.get_gaussian_center(a)
                voted = self.vote_centroid([disc_robust, disc_simple, gauss_center])
                r = root_yx or voted
                if root_yx is None and voted is not None:
                    print(f"  voted root: {voted} "
                          f"(disc_robust={disc_robust}, disc_simple={disc_simple}, gauss={gauss_center})")

                # 分别跑 A 和 V
                res_a = process_single(a, cfg, sub / "A", r, is_mask=(mode == "mask"))
                res_v = process_single(v, cfg, sub / "V", r, is_mask=(mode == "mask"))

                entry = {
                    "image": name,
                    "mode": mode,
                    "root_yx": [int(r[0]), int(r[1])] if r else None,
                    "disc_robust": [round(disc_robust[0], 1), round(disc_robust[1], 1)]
                                   if disc_robust else None,
                    "disc_simple": [round(disc_simple[0], 1), round(disc_simple[1], 1)]
                                   if disc_simple else None,
                    "gauss_center": [round(gauss_center[0], 1), round(gauss_center[1], 1)]
                                    if gauss_center else None,
                    "voted_center": [round(voted[0], 1), round(voted[1], 1)]
                                     if voted else None,
                    "root_source": "manual" if root_yx else "vote",
                    "A": res_a["RBAD"],
                    "V": res_v["RBAD"],
                }
                results.append(entry)
                with (sub / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                print(f"  A: {res_a['RBAD']['count']}  V: {res_v['RBAD']['count']}  root: {entry['root_source']}")
            except Exception as e:
                results.append({"image": name, "error": str(e)})
                print(f"  ERROR: {e}")

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
    ap.add_argument("--mode", choices=("skeleton", "mask"), default="skeleton")
    ap.add_argument("--no-disc-root", action="store_true")
    args = ap.parse_args()
    adapter = AutomorphAdapter(args.m2_dir)
    adapter.run(args.out, mode=args.mode, use_disc_root=not args.no_disc_root)
