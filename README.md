# RetiFlow

> Improved version of Retinal Branching Angle Detection algorithm

端到端流水线：眼底图 → RRWNet 多任务分割（A/V/BV）→ v3 概率路径补全 → 两遍法 RBAD 分叉检测。

核心解决两个问题：
1. **A/V 掩码不连续**导致分叉检测失效 → 用连续概率图补全 A/V 骨架。
2. **原版 RBAD 依赖全局连通**（单 root 遍历，断点即大面积丢失）→ 两遍法（全岛遍历 + 端点桥接）。

## 流水线

```
眼底图
  → RRWNet 分割 (A/V/BV 三通道概率图)
  → v3 概率路径补全 (BV 决定血管域，A/V 概率决定归属，桥接断口)
  → 两遍法 RBAD (原版 fast_keypoints + 全岛遍历 + 端点桥接)
  → 分叉点 + 角度 + 方向可视化
```

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch`、`torchvision`、`numpy`、`opencv-python`、`scikit-image`、`scipy`、`matplotlib`。

RRWNet 权重（`rrwnet_HRF_0.pth` 等）需放在可访问路径，用 `--weights` 指定。

## 用法

### 从眼底图推理

```bash
python -m RBAD_v2.infer \
  --image <眼底图路径> \
  --weights rrwnet_HRF_0.pth \
  --out <输出目录>
```

### 从已存概率图推理（跳过 RRWNet）

```bash
python -m RBAD_v2.infer \
  --prob-dir <含 seg_probabilities.npz 或 seg_A/V/BV.png 的目录> \
  --out <输出目录>
```

## 参数控制

所有参数集中在 `config.py`，命令行可覆盖关键参数。

### 分割（`SegmentationConfig`）
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--iterations` | 5 | RRWNet 递归精炼次数（1=单次，5=循环5次） |
| `thred` | 25 | 增强预处理 ROI 阈值 |

### v3 补全（`CompletionConfig`）
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--max-bridge-length` | 50 | 最大桥接路径长度（0 禁用） |
| `--max-opposite-run` | 50 | 共享路径最大连续反类像素数 |
| `--bridge-passes` | 5 | 桥接搜索轮数（0 禁用） |
| `low_bv` / `high_bv` | 0.25 / 0.50 | BV 滞回阈值 |

### 端点桥接（`EndpointBridgeConfig`）
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--max-distance` | 40 | 最大新增路径长度 |
| `--max-angle` | 35 | 端点切线到目标最大偏差（度） |
| `--passes` | 2 | 修复轮数（0 禁用） |
| `min_bv_mean` | 0.25 | 路径最小平均 BV 概率 |

### RBAD 角度（`RbadConfig`）
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--tail` | 15 | 分支追踪长度 |
| `angle_min` / `angle_max` | 20 / 120 | 保留角度范围 |

## 示例

`examples/02_200228_200228_L_mac/` 是完整输出示例（真实眼底图）。

### 输出文件
| 文件 | 说明 |
| --- | --- |
| `overlay_A_aligned.png` | A 分叉 + 两条子支方向箭头 + 角度（红） |
| `overlay_V_aligned.png` | V 分叉 + 方向箭头 + 角度（蓝） |
| `AV_BV_combined.png` | A(粉)/V(青)/BV(蓝) 组合图（原版 rrwnet 映射） |
| `A_mask.png` / `V_mask.png` | 补全后的二值掩码 |
| `A_centerline.png` / `V_centerline.png` | 连续中心线 |
| `enhanced_background.png` | 模型实际看到的增强图（overlay 背景） |
| `summary.json` | 分叉统计、参数、BV root |

### 示例结果（02_200228_200228_L_mac）
| 掩码 | 分叉数 | 角度 mean |
| --- | --- | --- |
| A | 27 | 75.6° |
| V | 44 | 73.6° |

## 性能报告

实测（RTX GPU，608×608 输入，单张）：

| 阶段 | 延迟 | 吞吐 |
| --- | --- | --- |
| RRWNet 纯推理（it5） | 85 ms | 11.7 fps |
| 分割（含预处理） | ~0.8 s | 1.2 fps |
| v3 补全（掩码） | ~1.2 s | 0.8 fps |
| 中心线 | ~0.6 s | — |
| 两遍法 RBAD | ~1.2 s | — |
| **端到端总计** | **~3.7 s** | **0.27 fps** |

**瓶颈**：v3 补全 + 中心线 + 两遍法 RBAD（~3s，占 80%），RRWNet 推理本身很快（85ms）。

### 各检测方案对比（A/V 分叉数）
| 方法 | A | V |
| --- | --- | --- |
| 原版 RBAD（单 root） | 7 | 0 |
| 局部检测（3分支要求） | 14 | 32 |
| **两遍法（原版+全岛+桥接）** | **27** | **44** |

两遍法用原版 RBAD 的角度逻辑，通过全岛遍历 + 端点桥接解决断点问题，A/V 分叉数最高。

## 未来优化方向

1. **性能**：
   - v3 补全和中心线有大量 Python 循环，可向量化或 Cython 化，预计提速 3-5 倍。
   - 两遍法跑两遍原版 RBAD，可缓存第一遍结果，只对桥接区域重算。
   - 批量推理时复用模型加载，摊薄分割开销。

2. **精度**：
   - 用视盘 mask（`--disc-mask`）替代高斯密度启发式找 root，更符合解剖学。
   - 端点桥接的 `max-distance` / `max-angle` 需在更多图上验证，避免错误连接。
   - 分叉角目前用原版 RBAD 的 child 选择，可加多尺度角度稳定性（参考 angle_v2）。

3. **鲁棒性**：
   - 在更大数据集（MobileLab 1426 张）上批量验证失败率和参数敏感性。
   - 处理视盘区域、交叉点、低对比度血管等难例。

4. **功能**：
   - 加盒计数分形维数作为全局特征（对断点完全免疫，与分叉角互补）。
   - 输出分叉方向（parent → daughters）用于血流方向分析。

## 目录结构

```
RBAD_v2/
├── infer.py              # 主推理入口
├── config.py             # 参数控制
├── detect/               # 分叉检测
│   ├── two_pass.py       # 两遍法（原版 RBAD + 全岛 + 桥接）
│   ├── endpoint_bridge.py# 端点桥接
│   ├── local_bifurcation.py  # 局部检测（备选）
│   └── utils.py          # 原版 RBAD fast_keypoints
├── completion/           # v3 概率路径补全
│   ├── completion.py
│   └── centerline.py
├── examples/             # 示例输出
└── README.md
```
