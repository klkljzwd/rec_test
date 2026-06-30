"""XGB run：holdout/submit 训练+评估+产出统一 metrics。

逻辑与 pipeline._run_holdout / _run_submit 逐行对应，只是评估原语换成
eval_core（函数体等价）。返回统一结构 metrics dict。

@register_runner("xgb_ranker") 注册到 dispatcher。
"""
from __future__ import annotations
import os
import time

import pandas as pd

from models import get_model
from models.registry import register_runner
from models._shared import eval_core

from . import data as D


@register_runner("xgb_ranker")
def run(cfg: dict) -> dict:
    """统一入口：吃 cfg，返回标准 metrics dict。"""
    t0 = time.time()
    mode = cfg.get("mode", "holdout")
    feat_cfg = dict(cfg.get("features", {}))
    feat_cfg["seed"] = cfg.get("seed", 42)
    topk = cfg.get("topk", 10)
    model_params = dict(cfg.get("model_params", {}))
    model_params.setdefault("seed", cfg.get("seed", 42))
    model = get_model("xgb_ranker", model_params)

    if mode == "holdout":
        return _holdout(cfg, model, feat_cfg, topk, t0)
    elif mode == "submit":
        return _submit(cfg, model, feat_cfg, topk, t0)
    raise ValueError(f"未知 mode='{mode}'，应为 'holdout' 或 'submit'")


def _holdout(cfg, model, feat_cfg, topk, t0):
    print(f"[xgb] mode=holdout | 构建可信 holdout 特征...")
    tr, va, feat_cols, cat_cols, tg, vg = D.build_holdout(cfg["datadir"], feat_cfg)
    print(f"      train={tr.shape} val={va.shape} feat={len(feat_cols)} "
          f"cat={len(cat_cols)} ({time.time()-t0:.0f}s)")

    watch_frac = cfg.get("watch_frac", cfg.get("pipeline", {}).get("watch_frac", 0.2))
    fit_df, watch_df, fit_groups, watch_groups = D.split_watch(tr, tg, watch_frac)
    print(f"[xgb] 训练模型 (fit={len(fit_groups)}组 watch={len(watch_groups)}组 early-stop)...")
    model.fit(fit_df, watch_df, feat_cols, cat_cols, fit_groups, watch_groups)
    print(f"      done ({time.time()-t0:.0f}s)")

    print(f"[xgb] val NDCG@{topk} (val 未参与训练/选模)...")
    scores = model.predict_scores(va, feat_cols)
    overall, recalled, n_users, succ_ndcg = eval_core.eval_ndcg(scores, va, topk)
    _print_report(overall, recalled, n_users, succ_ndcg, topk, t0)
    _print_importance(model, feat_cols)
    return eval_core.make_metrics(overall, succ_ndcg, recalled, n_users, topk)


def _submit(cfg, model, feat_cfg, topk, t0):
    print(f"[xgb] mode=submit | 全量训练特征 (inner OOF)...")
    tr, feat_cols, cat_cols, tg = D.build_oof(cfg["datadir"], feat_cfg)
    print(f"      train={tr.shape} ({time.time()-t0:.0f}s)")

    print("[xgb] 训练模型 (全量, 无 early-stop)...")
    model.fit(tr, None, feat_cols, cat_cols, tg, None)
    print(f"      done ({time.time()-t0:.0f}s)")

    print("[xgb] 构建 test 候选特征 (全量统计)...")
    te, _, _, _ = D.build_test(cfg["datadir"], feat_cfg)
    print(f"      test={te.shape} ({time.time()-t0:.0f}s)")

    print("[xgb] 打分取 top-10 并对齐 sample_submission...")
    scores = model.predict_scores(te, feat_cols)
    pred = eval_core.topk_predictions(scores, te, topk)

    sub = pd.read_csv(os.path.join(cfg["datadir"], "sample_submission.csv"), dtype=str)
    out = sub[["uid"]].merge(pred, on="uid", how="left")
    out["prediction"] = out["prediction"].fillna(sub["prediction"])
    out_path = cfg.get("out", "submissions/submission.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    lens = out["prediction"].str.split(",").apply(len)
    print(f"[xgb] 写出 {out_path}: {len(out)} 行 | 每行iid数 {lens.min()}~{lens.max()} "
          f"| 缺失 {out['prediction'].isna().sum()} | 总耗时 {time.time()-t0:.0f}s")
    return {"out": out_path, "n_rows": len(out)}


def _print_report(overall, recalled, n_users, succ_ndcg, topk, t0):
    print(f"\n{'='*50}")
    print(f"验证集 NDCG@{topk} (全体用户, 含召回失败): {overall:.5f}")
    print(f"  召回成功 {recalled}/{n_users} ({recalled/n_users:.1%}) | "
          f"召回失败 {n_users-recalled} (NDCG=0)")
    print(f"  NDCG@{topk} (仅召回成功, 纯排序): {succ_ndcg:.5f}")
    print(f"{'='*50}\n总耗时 {time.time()-t0:.0f}s")


def _print_importance(model, feat_cols):
    imp = model.feature_importance()
    if not imp:
        return
    if "f0" in imp:
        imp = {feat_cols[int(k[1:])]: v for k, v in imp.items() if k.startswith("f")}
    items = sorted(imp.items(), key=lambda x: -x[1])[:12]
    print("Top-12 特征重要性(gain):")
    for name, v in items:
        print(f"  {name}: {v:.1f}")
