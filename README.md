# RetiFlow

> Improved Retinal Branching Angle Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RetiFlow** is an end-to-end pipeline for retinal artery/vein (A/V) segmentation and
bifurcation angle measurement. It takes a color fundus image and produces
per-vessel-type bifurcation detections with angles and branch directions, robust
to the discontinuous vessel masks that plague real-world segmentation.

---

## Three Key Improvements

### 1. A/V maps measured separately
Arteries and veins are segmented and measured **independently**, not as a single
vessel tree. Each class gets its own bifurcation statistics (count, angle
distribution), enabling artery-specific and vein-specific clinical analysis.

### 2. Probability-mask completion + centerline extraction
Instead of hard-thresholding each A/V channel (which breaks thin vessels), the
continuous **probability maps** are used to complete the A/V masks. A
probability-cost **centerline** is then extracted as the skeleton — more precise
and continuous than naive threshold-and-thin.

### 3. Endpoint bridging
Disconnected vessel fragments are **actively repaired**: each endpoint is traced
back along its parent arm, its tangent is estimated, and facing endpoints are
bridged (or an endpoint lands on a foreign branch body) under strict geometric
and BV-support checks. The original RBAD angle logic then runs on the repaired,
continuous skeleton.

---

## Pipeline

```
Fundus image
  → RRWNet segmentation (A/V/BV probability maps)
  → v3 probability-path completion (BV defines vessel domain, A/V define class)
  → two-pass RBAD (original fast_keypoints + all-island traversal + endpoint bridge)
  → bifurcation points + angles + branch directions
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`, `scikit-image`,
`scipy`, `matplotlib`.

RRWNet weights (e.g. `rrwnet_HRF_0.pth`) must be reachable; point to them with
`--weights`.

---

## Usage

### From a fundus image

```bash
python -m RBAD_v2.infer \
  --image <fundus_image> \
  --weights rrwnet_HRF_0.pth \
  --out <output_dir>
```

### From saved probability maps (skip RRWNet)

```bash
python -m RBAD_v2.infer \
  --prob-dir <dir with seg_probabilities.npz or seg_A/V/BV.png> \
  --out <output_dir>
```

---

## Parameters

All parameters live in `config.py`; key ones are overridable on the command line.

### Segmentation (`SegmentationConfig`)
| Option | Default | Description |
| --- | --- | --- |
| `--iterations` | 5 | RRWNet recursive refinement passes (1 = single, 5 = loop) |
| `thred` | 25 | Enhancement ROI threshold |

### v3 Completion (`CompletionConfig`)
| Option | Default | Description |
| --- | --- | --- |
| `--max-bridge-length` | 50 | Max bridge path length (0 disables) |
| `--max-opposite-run` | 50 | Max consecutive opposite-class pixels on a shared path |
| `--bridge-passes` | 5 | Bridge search passes (0 disables) |
| `low_bv` / `high_bv` | 0.25 / 0.50 | BV hysteresis thresholds |

### Endpoint Bridge (`EndpointBridgeConfig`)
| Option | Default | Description |
| --- | --- | --- |
| `--max-distance` | 40 | Max added path length |
| `--max-angle` | 35 | Max endpoint tangent-to-target deviation (deg) |
| `--passes` | 2 | Repair passes (0 disables) |
| `min_bv_mean` | 0.25 | Min mean BV probability along an accepted route |

### RBAD Angle (`RbadConfig`)
| Option | Default | Description |
| --- | --- | --- |
| `--tail` | 15 | Branch tracing length |
| `angle_min` / `angle_max` | 20 / 120 | Accepted angle range |

---

## Example

`examples/02_200228_200228_L_mac/` is a complete run on a real fundus image.

### A/V bifurcation overlays (with branch directions)

| Artery (A) | Vein (V) |
| --- | --- |
| ![A overlay](examples/02_200228_200228_L_mac/overlay_A_aligned.png) | ![V overlay](examples/02_200228_200228_L_mac/overlay_V_aligned.png) |

### Combined A/V/BV map (original RRWNet color convention)

A = magenta, V = cyan, BV = blue, crossing = white.

![AV/BV combined](examples/02_200228_200228_L_mac/AV_BV_combined.png)

### Completed masks and centerlines

| A mask | V mask | A centerline | V centerline |
| --- | --- | --- | --- |
| ![A mask](examples/02_200228_200228_L_mac/A_mask.png) | ![V mask](examples/02_200228_200228_L_mac/V_mask.png) | ![A centerline](examples/02_200228_200228_L_mac/A_centerline.png) | ![V centerline](examples/02_200228_200228_L_mac/V_centerline.png) |

### Example result (02_200228_200228_L_mac)
| Class | Bifurcations | Mean angle |
| --- | --- | --- |
| A | 27 | 75.6° |
| V | 44 | 73.6° |

---

## Performance

Measured on an RTX GPU, 608×608 input, single image:

| Stage | Latency | Throughput |
| --- | --- | --- |
| RRWNet inference (it5) | 85 ms | 11.7 fps |
| Segmentation (incl. preprocessing) | ~0.8 s | 1.2 fps |
| v3 completion (masks) | ~1.2 s | 0.8 fps |
| Centerlines | ~0.6 s | — |
| Two-pass RBAD | ~1.2 s | — |
| **End-to-end total** | **~3.7 s** | **0.27 fps** |

**Bottleneck**: v3 completion + centerlines + two-pass RBAD (~3 s, ~80% of total).
RRWNet inference itself is fast (85 ms).

### Detection comparison (A/V bifurcation counts)
| Method | A | V |
| --- | --- | --- |
| Original RBAD (single root) | 7 | 0 |
| Local detection (3-branch) | 14 | 32 |
| **Two-pass (original + all-island + bridge)** | **27** | **44** |

The two-pass method keeps the original RBAD angle logic while solving the
discontinuity problem via all-island traversal and endpoint bridging.

---

## Future Work

- **Performance**: vectorize/Cython the v3 completion and centerline Python loops
  (3–5× speedup expected); cache the first RBAD pass and recompute only bridged
  regions; reuse the loaded model across a batch.
- **Accuracy**: use an optic-disc mask (`--disc-mask`) instead of the Gaussian
  density heuristic for the root; validate `max-distance`/`max-angle` on more
  images; add multi-scale angle stability.
- **Robustness**: batch-validate on larger datasets (e.g. MobileLab, 1426 images);
  handle optic-disc, crossing, and low-contrast cases.
- **Features**: add box-counting fractal dimension as a global, break-immune
  feature; output parent→daughter directions for blood-flow analysis.

---

## Repository Layout

```
RetiFlow/
├── infer.py              # End-to-end inference entry point
├── config.py             # Parameter control
├── detect/               # Bifurcation detection
│   ├── two_pass.py       # Two-pass (original RBAD + all-island + bridge)
│   ├── endpoint_bridge.py# Endpoint bridging
│   ├── local_bifurcation.py  # Local detection (alternative)
│   └── utils.py          # Original RBAD fast_keypoints
├── completion/           # v3 probability-path completion
│   ├── completion.py
│   └── centerline.py
├── examples/             # Example outputs
└── README.md
```

---

## Citation

If you use RetiFlow in your research, please cite this repository and the
original RBAD benchmark:

```bibtex
@inproceedings{wang2024rbad,
  title={RBAD: A dataset and benchmark for retinal vessels branching angle detection},
  author={Wang, Hao and Zhu, Wenhui and Qin, Jiayou and Li, Xin and Dumitrascu, Oana and Chen, Xiwen and Qiu, Peijie and Razi, Abolfazl and Wang, Yalin},
  booktitle={2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgements

RetiFlow builds on two open-source projects:

- **RBAD** — the original retinal branching angle detection benchmark and
  algorithm: <https://github.com/Retinal-Research/RBAD>
- **RRWNet** — the recursive refinement network for retinal artery/vein
  segmentation: <https://github.com/j-morano/rrwnet>

We thank the authors of both projects for making their work publicly available.
