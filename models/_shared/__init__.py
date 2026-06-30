"""模型层共享内核：评估原语 + 实验日志。

`eval_core` 是评估口径统一的根基（约束 A）：所有 run.py 必须调这里的
`ndcg_at_k`/`eval_ndcg`/`topk_predictions`/`make_metrics`，不得自写 NDCG。
`logging` 提供 config 指纹与 JSONL 实验记录，供 agent 与人复盘。
"""
from __future__ import annotations

from . import eval_core
from . import logging

__all__ = ["eval_core", "logging"]
