"""统一调度：cfg["model"] -> get_runner -> run(cfg) -> metrics。

不回退旧 pipeline（用户选择强制每个模型都有 run.py）。找不到 runner 直接抛错，
逼出"某模型漏写 run.py"的问题，而非静默走旧路径。
"""
from __future__ import annotations

from .registry import get_runner


def run_experiment(cfg: dict) -> dict:
    """统一调度入口：返回标准 metrics dict。"""
    runner = get_runner(cfg["model"])
    return runner(cfg)
