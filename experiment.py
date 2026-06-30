"""程序化实验入口（agent 对接用）。

agent 的对接契约：
    import experiment
    metrics = experiment.run("experiments/cfg.json")
    # 或: metrics = experiment.run(config_dict)

run(config) 把 config（路径或 dict）跑出 metrics，同时：
  1. 返回 metrics dict（程序化直接用，无序列化损失）。
  2. 写一份结果 JSON 到 experiments/results/<fingerprint>.json（跨进程读取）。
  3. 追加一条到 experiments/log.jsonl（人/agent 复盘历次实验）。

调用链（直接，无间接转出）：
    experiment.run
      -> models._shared.config_utils.load_from_json / load_by_name  合并 cfg
      -> models.dispatcher.run_experiment           路由到 models/<name>/run.py
      -> models._shared.logging.log_experiment      落盘

agent 不碰 argparse、不解析 stdout——直接 import 这个函数。
CLI(run.py) 也复用本模块的 run_from_cfg，保证两条入口走同一条路。
"""
from __future__ import annotations
import json
import os
import time

from models import dispatcher
from models._shared.logging import config_fingerprint, log_experiment
from models._shared.config_utils import (
    load_from_json, load_by_name, _deep_update, _apply_overrides,
)


def run(cfg_input, overrides: dict | None = None) -> dict:
    """跑一个实验，返回 metrics dict。

    Args:
        cfg_input: str（JSON config 文件路径）或 dict（已合并的 cfg）。
        overrides: 可选，深路径覆盖（如 {"features.candidate_k": 200}），
                   优先级最高，等价于 CLI 的 --param。

    Returns:
        metrics dict，如 {"ndcg@10": 0.504, "succ_ndcg": 0.587,
        "n_recalled": 3436, "n_users": 4000}。submit 模式含 {"out","n_rows"}。

    副作用：写 experiments/results/<fingerprint>.json，追加 experiments/log.jsonl。
    """
    cfg = _resolve_cfg(cfg_input, overrides)
    return run_from_cfg(cfg)


def run_from_cfg(cfg: dict) -> dict:
    """已合并 cfg → 跑 → 返回 metrics，并落盘结果文件 + 日志。

    CLI(run.py) 与 run() 都走这里，保证同一条路径。
    """
    metrics = dispatcher.run_experiment(cfg)   # 直接调 dispatcher，无间接转出
    _write_result(cfg, metrics)
    log_experiment(cfg, metrics)
    return metrics


def _resolve_cfg(cfg_input, overrides) -> dict:
    if isinstance(cfg_input, str):
        cfg = load_from_json(cfg_input, overrides)
    elif isinstance(cfg_input, dict):
        cfg = load_by_name(cfg_input.get("model", "xgb_ranker"))
        # 用 dict 内容深覆盖（与 load_from_json 同语义）
        for key, val in cfg_input.items():
            if key == "model":
                continue
            if key in ("model_params", "features", "pipeline") and isinstance(val, dict):
                cfg[key] = _deep_update(cfg.get(key, {}), val)
            elif key in ("topk", "seed", "watch_frac", "out", "mode", "datadir"):
                cfg[key] = val
                if key in ("topk", "seed", "watch_frac", "out") and "pipeline" in cfg:
                    cfg["pipeline"][key] = val
            else:
                cfg[key] = val
        if overrides:
            cfg = _apply_overrides(cfg, overrides)
    else:
        raise TypeError(f"cfg_input 应为 str(JSON路径) 或 dict，得到 {type(cfg_input)}")
    # 兜底默认
    cfg.setdefault("datadir", "data/A推荐")
    cfg.setdefault("mode", "holdout")
    return cfg


def _write_result(cfg: dict, metrics: dict) -> str:
    """把本次结果写成独立 JSON 文件，便于 agent 跨进程读取。

    路径：experiments/results/<fingerprint>.json。同配置重跑覆盖同一文件。
    """
    fp = config_fingerprint(cfg)
    out_dir = os.path.join("experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fp}.json")
    record = {
        "config_fingerprint": fp,
        "model": cfg.get("model"),
        "mode": cfg.get("mode"),
        "metrics": metrics,
        "cfg": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return path


if __name__ == "__main__":
    # 命令行直跑：python experiment.py experiments/cfg.json
    import sys
    if len(sys.argv) < 2:
        print("用法: python experiment.py <config.json> [key=val ...]")
        sys.exit(1)
    path = sys.argv[1]
    ov = {}
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        ov[k] = v
    m = run(path, ov or None)
    print(json.dumps(m, ensure_ascii=False, default=float))
