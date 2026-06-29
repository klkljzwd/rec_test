"""模型注册表。config 用名字引用模型，pipeline 通过 get_model 取实例。"""
from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(cls):
    """类装饰器：把 RankerModel 子类按其 name 注册。"""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls} 缺少 name 属性")
    _REGISTRY[name] = cls
    return cls


def get_model(name: str, params: dict | None = None):
    """按名字实例化一个模型。"""
    if name not in _REGISTRY:
        raise ValueError(f"未知模型 '{name}'，可用: {list(_REGISTRY)}")
    return _REGISTRY[name](params)


def list_models() -> list[str]:
    return sorted(_REGISTRY)
