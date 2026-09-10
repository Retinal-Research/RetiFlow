# -*- coding: utf-8 -*-
"""
AutoMorph 角度实验汇总：生成 run_summary.json（M3 风格）+ angle_metrics.json。

读取 Angle 输出目录下每张图的 summary.json，聚合性能指标和角度指标。

用法：
  python -m RetiFlow.plugins.automorph_summary \
    --angle-dir <.../Results/Angle> \
    --out <.../Results/Angle/run_summary.json>
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))


def _angle_stats(angles):
    """角度统计。angles: list of float。"""
    if not angles:
        return {"count": 0}
    a = np.asarray(angles, dtype=float)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 3),
        "median": round(float(np.median(a)), 3),
        "std": round(float(a.std()), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
    }


def _histogram(angles, bins=range(20, 125, 5)):
    """角度直方图。"""
    if not angles:
        return {}
    a = np.asarray(angles, dtype=float)
    hist, edges = np.histogram(a, bins=list(bins))
    return {f"{edges[i]}-{edges[i+1]}": int(hist[i]) for i in range(len(hist))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angle-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    angle_dir = Path(args.angle_dir)
    out = args.out or (angle_dir / "run_summary.json")

    t0 = time.perf_counter()
    per_image = []
    a_angles, v_angles = [], []
    errors = 0
    for sub in sorted(angle_dir.iterdir()):
        if not sub.is_dir():
            continue
        sj = sub / "summary.json"
        if not sj.is_file():
            continue
        try:
            entry = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            errors += 1
            continue
        per_image.append(entry)
        a = entry.get("A", {})
        v = entry.get("V", {})
        a_angles.extend(a.get("angles", []))
        v_angles.extend(v.get("angles", []))

    wall_s = time.perf_counter() - t0
    n = len(per_image)
    a_counts = [e.get("A", {}).get("count", 0) for e in per_image]
    v_counts = [e.get("V", {}).get("count", 0) for e in per_image]

    summary = {
        "module": "RetiFlow-Angle",
        "data_root": str(angle_dir.parent),
        "input_dir": str(angle_dir),
        "output_dir": str(angle_dir),
        "mode": per_image[0].get("mode") if per_image else None,
        "image_count": n,
        "error_count": errors,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "summary_wall_s": round(wall_s, 3),
        "angle_metrics": {
            "A": {
                "bifurcation_count": {
                    "total": int(sum(a_counts)),
                    "per_image_mean": round(float(np.mean(a_counts)), 2),
                    "per_image_std": round(float(np.std(a_counts)), 2),
                    "min": int(min(a_counts)) if a_counts else 0,
                    "max": int(max(a_counts)) if a_counts else 0,
                },
                "angle": _angle_stats(a_angles),
                "angle_histogram": _histogram(a_angles),
            },
            "V": {
                "bifurcation_count": {
                    "total": int(sum(v_counts)),
                    "per_image_mean": round(float(np.mean(v_counts)), 2),
                    "per_image_std": round(float(np.std(v_counts)), 2),
                    "min": int(min(v_counts)) if v_counts else 0,
                    "max": int(max(v_counts)) if v_counts else 0,
                },
                "angle": _angle_stats(v_angles),
                "angle_histogram": _histogram(v_angles),
            },
            "combined": {
                "bifurcation_count": {
                    "total": int(sum(a_counts) + sum(v_counts)),
                    "per_image_mean": round(float(np.mean([a + v for a, v in zip(a_counts, v_counts)])), 2),
                },
                "angle": _angle_stats(a_angles + v_angles),
            },
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n输出: {out}")


if __name__ == "__main__":
    main()
