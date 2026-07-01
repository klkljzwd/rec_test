"""推荐任务实验入口。

模型名即实验名：--model <模型名> 直接取 models/<name>/config.py 的 MODEL_CONFIG
+ FEATURES/PIPELINE 默认值合并后跑。无 EXPERIMENTS 间接层。

用法：
  # 验证（可信 holdout NDCG，本地≈线上）
  python run.py --mode holdout

  # 生成提交
  python run.py --mode submit --out submissions/A3.csv

  # 命令行覆盖任意参数（深路径，可多次）
  python run.py --mode holdout \
      --param model_params.lr=0.03 --param model_params.max_depth=8 \
      --param features.candidate_k=100

  # JSON 配置文件入口（agent 友好，可叠加 --param）
  python run.py --config experiments/xxx.json --mode holdout

  # 列出所有模型 / 查看合并配置
  python run.py --list
  python run.py --show

参数路径：model_params.* / features.* / pipeline.* / topk / seed / out。
加新模型→models/建目录(model/data/run/config)+@register+@register_runner。
"""
from __future__ import annotations
import argparse
import json
import sys

import models
import experiment
from models._shared.config_utils import load_by_name, load_from_json
from models._shared.logging import log_experiment


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data/A推荐")
    ap.add_argument("--mode", choices=["holdout", "submit"], default="holdout")
    ap.add_argument("--model", default="xgb_ranker", help="models/<name>/ 里的模型名")
    ap.add_argument("--out", help="提交输出路径（submit 模式）")
    ap.add_argument("--param", action="append", default=[],
                    help="深路径覆盖，如 model_params.lr=0.1 / features.neg=30（可多次）")
    ap.add_argument("--config", help="JSON 配置文件路径（覆盖默认 config，可再叠加 --param）")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                    help="计算设备：auto(有GPU用GPU)/cpu/cuda。仅深度学习模型(如deepfm)生效，默认 auto")
    ap.add_argument("--list", action="store_true", help="列出所有可用模型")
    ap.add_argument("--show", action="store_true", help="打印合并后配置不运行")
    args = ap.parse_args()

    if args.list:
        print("可用模型:", models.list_models())
        return

    overrides = {}
    for kv in args.param:
        k, v = kv.split("=", 1)
        overrides[k] = _coerce(v)

    if args.config:
        cfg = load_from_json(args.config, overrides)
    else:
        cfg = load_by_name(args.model, overrides)
    cfg["datadir"] = args.datadir
    cfg["mode"] = args.mode
    if args.out:
        cfg["out"] = args.out
    if args.device:
        # 透传到 model_params.device，供深度学习模型(如 deepfm)读取。
        # 树模型(xgb/lgb)忽略此参数。
        cfg.setdefault("model_params", {})["device"] = args.device

    if args.show:
        # 强制 UTF-8 输出（Windows 控制台默认 GBK 会让中文路径写坏），
        # 保证 --show > cfg.json 后能被 --config / experiment.run 读回。
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
        return

    _print_run_config(cfg, overrides)
    result = experiment.run_from_cfg(cfg)   # CLI 与程序化入口走同一条路
    print("[run] result:", json.dumps(result, ensure_ascii=False, default=float))


def _print_run_config(cfg: dict, overrides: dict):
    """跑之前打印本次配置：模型/模式 + 关键特征参数 + 模型参数 + 覆盖项。"""
    f = cfg["features"]
    mp = cfg["model_params"]
    p = cfg["pipeline"]
    aligned = f["train_candidate_k"] == f["candidate_k"]
    print("=" * 60)
    print(f"[config] model={cfg['model']}  mode={cfg['mode']}  seed={cfg['seed']}  topk={cfg['topk']}")
    print(f"[config] features: candidate_k={f['candidate_k']} "
          f"train_candidate_k={f['train_candidate_k']}{'(难度对齐)' if aligned else ''} "
          f"hard_ratio={f['hard_negative_ratio']} collab={f['collab']} "
          f"outer={f['outer_folds']}/{f['outer_fold']} inner={f['inner_folds']}")
    print(f"[config] recall_weights: {f.get('score_weights', {})}")
    mp_str = " ".join(f"{k}={v}" for k, v in mp.items() if k != "verbose_eval")
    print(f"[config] model_params: {mp_str}")
    print(f"[config] pipeline: watch_frac={p['watch_frac']} out={cfg['out']}")
    if overrides:
        print(f"[config] 命令行覆盖: {overrides}")
    print("=" * 60)


if __name__ == "__main__":
    main()
