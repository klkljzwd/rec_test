"""Fusion 配置加载（自包含）。

把本模型 config.py 的 MODEL_CONFIG 合并成完整 cfg，结构与其它模型逐字段等价。
"""
from __future__ import annotations

from models._shared.config_utils import _apply_overrides

from .config import MODEL_CONFIG


def load(overrides: dict | None = None) -> dict:
    m = MODEL_CONFIG
    cfg = {
        "model": "fusion",
        "model_params": dict(m["model_params"]),
        "features": dict(m["features"]),
        "pipeline": dict(m["pipeline"]),
    }
    for k in ("topk", "seed", "watch_frac", "out"):
        cfg[k] = cfg["pipeline"][k]
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg
