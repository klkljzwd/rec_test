"""配置合并工具 + JSON 入口（模型自治架构的共享基建）。

本模块只放与具体模型无关的合并工具：深覆盖、深路径 --param 覆盖、JSON 文件加载。
各模型的实际配置在 models/<name>/config.py，加载在 models/<name>/load.py。

load_from_json 是 JSON config 入口：读 JSON 的 model 名 → 调对应模型 load.py
得基础 cfg → 用 JSON 其余字段深覆盖 → --param 覆盖。优先级：
  CLI --param > JSON 文件 > 模型 config.py > （无中央默认，模型全量自包含）。
"""
from __future__ import annotations
import copy
import importlib
import json


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_nested(d: dict, keys: list[str], val):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """深路径覆盖：{'model_params.lr':0.1, 'features.neg':30, 'topk':5}。"""
    for path, val in overrides.items():
        parts = path.split(".")
        if parts[0] in ("model", "topk", "seed", "watch_frac", "out", "mode", "datadir"):
            cfg[parts[0]] = val
            if parts[0] in ("topk", "seed", "watch_frac", "out"):
                cfg["pipeline"][parts[0]] = val
            continue
        if parts[0] == "model_params":
            _set_nested(cfg["model_params"], parts[1:], val)
        elif parts[0] == "features":
            _set_nested(cfg["features"], parts[1:], val)
        elif parts[0] == "pipeline":
            _set_nested(cfg["pipeline"], parts[1:], val)
            if len(parts) == 2 and parts[1] in ("topk", "seed", "watch_frac", "out"):
                cfg[parts[1]] = val
        else:
            raise KeyError(f"无法识别的覆盖路径: {path}")
    return cfg


def load_by_name(model_name: str, overrides: dict | None = None) -> dict:
    """调 models.<model_name>.load.load(overrides) 得完整 cfg。"""
    try:
        load_mod = importlib.import_module(f"models.{model_name}.load")
    except ModuleNotFoundError:
        from models.registry import list_models
        raise KeyError(f"未知模型 '{model_name}'，可用: {list_models()}")
    return load_mod.load(overrides)


def load_from_json(path: str, overrides: dict | None = None) -> dict:
    """从 JSON 文件加载配置：JSON model 名 → 模型 load.py 得基础 cfg → JSON 深覆盖 → --param。

    JSON 格式（完整 cfg 子集）：
      {"model": "xgb_ranker", "mode": ..., "datadir": ...,
       "model_params": {...}, "features": {...}, "pipeline": {...}, "topk": ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)
    model_name = file_cfg.get("model")
    if not model_name:
        raise ValueError("JSON config 缺少 'model' 字段")
    cfg = load_by_name(model_name)  # 基础 cfg（来自模型 config.py）
    for key, val in file_cfg.items():
        if key == "model":
            continue
        if key in ("model_params", "features", "pipeline") and isinstance(val, dict):
            cfg[key] = _deep_update(cfg.get(key, {}), val)
        elif key in ("topk", "seed", "watch_frac", "out", "mode", "datadir"):
            cfg[key] = val
            if key in ("topk", "seed", "watch_frac", "out"):
                cfg["pipeline"][key] = val
        else:
            cfg[key] = val
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg
