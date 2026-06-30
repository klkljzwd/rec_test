"""实验日志 + config 指纹。

`config_fingerprint`: 对影响数值的字段做稳定哈希，用于去重与横向对比。
  排除 out/datadir/timestamp 等不影响结果的字段，保证同实验同指纹。

`log_experiment`: 每次实验把 {timestamp, config_fingerprint, cfg, metrics,
  model, mode} 追加到 experiments/log.jsonl。JSONL 追加不覆盖，agent 与人
  可据此复盘历次实验。
"""
from __future__ import annotations
import hashlib
import json
import os
import time


# 影响数值结果的 cfg 顶层字段（其余如 out/datadir 不进指纹）
_FINGERPRINT_KEYS = ("model", "model_params", "features", "pipeline",
                     "topk", "seed", "mode")


def config_fingerprint(cfg: dict) -> str:
    """对影响数值的 cfg 字段做稳定哈希，返回 sha1 前 12 位。"""
    sub = {k: cfg.get(k) for k in _FINGERPRINT_KEYS if k in cfg}
    blob = json.dumps(sub, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def log_experiment(cfg: dict, metrics: dict,
                   path: str = "experiments/log.jsonl") -> str:
    """追加一条实验记录到 JSONL，返回写入路径。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_fingerprint": config_fingerprint(cfg),
        "model": cfg.get("model"),
        "mode": cfg.get("mode"),
        "cfg": cfg,
        "metrics": metrics,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path
