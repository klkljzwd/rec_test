"""模型注册表。config 用名字引用模型，dispatcher 通过 get_runner 取入口。

并存两套注册：
  - 类注册（_REGISTRY）：候选表模型 RankerModel 子类，get_model 取实例。
  - runner 注册（_RUNNERS）：name -> run(cfg)->metrics，dispatcher 调用入口。

候选表模型（XGB/LGB）两者都有：类注册供 get_model 实例化，runner 注册供
dispatcher 路由。未来端到端序列模型可只提供 runner（不实现 RankerModel）。
"""
from __future__ import annotations

_REGISTRY: dict[str, type] = {}        # 旧：类注册（RankerModel 子类）
_RUNNERS: dict[str, callable] = {}     # 新：name -> run(cfg)->metrics


# --------------------------------------------------------------------------- #
# 类注册（候选表模型）
# --------------------------------------------------------------------------- #
def register(cls):
    """类装饰器：把 RankerModel 子类按其 name 注册。"""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls} 缺少 name 属性")
    _REGISTRY[name] = cls
    return cls


def get_model(name: str, params: dict | None = None):
    """按名字实例化一个 RankerModel。"""
    if name not in _REGISTRY:
        raise ValueError(f"未知模型类 '{name}'，可用: {sorted(_REGISTRY)}")
    return _REGISTRY[name](params)


# --------------------------------------------------------------------------- #
# runner 注册（dispatcher 入口）
# --------------------------------------------------------------------------- #
def register_runner(name: str):
    """装饰器：把 run(cfg)->metrics 注册为模型 name 的统一入口。"""
    def deco(fn):
        _RUNNERS[name] = fn
        return fn
    return deco


def get_runner(name: str):
    """按名字取 runner。找不到直接抛错（用户选了不回退，强制每模型有 run.py）。"""
    if name not in _RUNNERS:
        raise KeyError(f"未知 runner '{name}'，可用: {sorted(_RUNNERS)}")
    return _RUNNERS[name]


def list_models() -> list[str]:
    """类名 ∪ runner 名，去重排序。"""
    return sorted(set(_REGISTRY) | set(_RUNNERS))
