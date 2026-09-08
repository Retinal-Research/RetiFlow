# -*- coding: utf-8 -*-
"""RetiFlow 全部参数控制。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class CompletionConfig:
    """v3 概率路径 A/V 补全参数（距离单位为概率图像素）。"""
    low_bv: float = 0.25         # BV 滞回低阈值
    high_bv: float = 0.50        # BV 滞回高阈值
    min_bv_area: int = 20        # 最小 BV 连通域面积
    semantic_margin: float = 0.12
    min_anchor_area: int = 4
    max_bridge_length: float = 50.0   # 最大桥接路径长度（0 禁用）
    max_opposite_run: int = 50        # 共享路径上最大连续反类像素数
    min_facing_cosine: float = 0.1
    direction_radius: float = 10.0
    bridge_passes: int = 5            # 桥接搜索轮数（0 禁用）
    strict_direction_check: bool = False


@dataclass
class EndpointBridgeConfig:
    """端点桥接（两遍法修复）参数。"""
    max_distance: float = 40.0    # 最大新增路径长度
    max_angle_deg: float = 35.0   # 端点切线到目标最大偏差
    lookback: float = 12.0        # 沿父臂回溯距离
    min_parent_length: float = 4.0
    min_bv_mean: float = 0.25     # 路径最小平均 BV 概率
    low_bv: float = 0.10
    max_unsupported_run: int = 8  # 最大连续低 BV 像素数
    passes: int = 2               # 修复轮数（0 禁用）
    allow_body_landing: bool = True
    min_body_angle_deg: float = 25.0


@dataclass
class RbadConfig:
    """原版 RBAD 角度检测参数。"""
    tail: int = 15                # 分支追踪长度
    child_min_dist: int = 5
    angle_min: float = 20.0
    angle_max: float = 120.0


@dataclass
class Config:
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    endpoint_bridge: EndpointBridgeConfig = field(default_factory=EndpointBridgeConfig)
    rbad: RbadConfig = field(default_factory=RbadConfig)

    def to_dict(self):
        return asdict(self)
