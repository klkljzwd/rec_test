"""推荐任务复用流水线。

把"数据构建 → 训练 → 验证/提交"固定下来，方法（模型 + 特征）通过 config 注入。

两种模式：
  mode="holdout" : build_holdout_features 出可信 train/val → 训练(带 early-stop)
                   → val NDCG@10。用于开发、对比方法。本地分数≈线上(已验证 0.242↔0.2476)。
  mode="submit"  : build_oof_features(全量训练) + build_test_features(全量统计)
                   → 训练 → 生成对齐 sample_submission 的提交 csv。

config 字段：
  datadir, mode, model(模型名), model_params, features(传给 build_features 的参数),
  topk, seed, out(提交路径), verbose.

换方法的三种方式（互不干扰 pipeline）：
  1. 换模型   : 在 models/ 加一个 RankerModel 子类 + @register，config["model"] 改名。
  2. 加特征   : 改 build_features.py 的 _feature_schema + _assemble_features（信号/组装
                已分离，加特征只动这两处），pipeline 自动透传新 feat_cols/cat_cols。
  3. 调参     : 改 config["model_params"] / config["features"]。
"""
from __future__ import annotations
import math
import os
import time

import numpy as np
import pandas as pd

import build_features as bf
from models import get_model, list_models


# --------------------------------------------------------------------------- #
# 评估原语（与模型无关，全 pipeline 复用）
# --------------------------------------------------------------------------- #
def ndcg_at_k(pred_iids, target_iid, k=10):
    """单用户 NDCG@k：target 在 top-k 内则 1/log2(rank+2)，否则 0。"""
    for i, iid in enumerate(pred_iids[:k]):
        if iid == target_iid:
            return 1.0 / math.log2(i + 2)
    return 0.0


def eval_ndcg(scores: np.ndarray, df: pd.DataFrame, k=10):
    """在带 label 的候选表上算 NDCG@k。

    scores: 与 df 行对齐的预测分。
    返回 (overall_ndcg, n_recalled, n_users, succ_ndcg)。
      - overall: 全体用户均值（召回失败的用户 NDCG=0）。
      - n_recalled: target 进入候选的用户数。
      - succ_ndcg: 仅召回成功用户的均值（纯排序能力）。
    """
    g = df.assign(_score=scores)
    ndcgs, succ = [], []
    for uid, sub in g.groupby("uid", sort=False):
        if not (sub["label"] == 1).any():
            ndcgs.append(0.0)
            continue
        top = sub.sort_values("_score", ascending=False)["iid"].head(k).tolist()
        tgt = sub[sub["label"] == 1]["iid"].iloc[0]
        sc = ndcg_at_k(top, tgt, k)
        ndcgs.append(sc)
        succ.append(sc)
    return (float(np.mean(ndcgs)), len(succ), len(ndcgs),
            float(np.mean(succ)) if succ else 0.0)


def topk_predictions(scores: np.ndarray, df: pd.DataFrame, k=10) -> pd.DataFrame:
    """由分数取每用户 top-k iid，返回 (uid, prediction) 表。"""
    g = df.assign(_score=scores)
    rows = []
    for uid, sub in g.groupby("uid", sort=False):
        top = sub.sort_values("_score", ascending=False)["iid"].head(k).tolist()
        if len(top) < k:  # 召回不足补齐（罕见）
            top += [iid for iid in sub["iid"] if iid not in top][:k - len(top)]
        rows.append((uid, ",".join(top)))
    return pd.DataFrame(rows, columns=["uid", "prediction"])


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_experiment(cfg: dict):
    t0 = time.time()
    mode = cfg.get("mode", "holdout")
    model_name = cfg["model"]
    feat_cfg = dict(cfg.get("features", {}))
    feat_cfg["seed"] = cfg.get("seed", 42)
    topk = cfg.get("topk", 10)
    # 全局 seed 同步注入模型(若 model_params 未显式设 seed)，保证数据划分与模型随机性一致
    model_params = dict(cfg.get("model_params", {}))
    model_params.setdefault("seed", cfg.get("seed", 42))
    model = get_model(model_name, model_params)

    if mode == "holdout":
        return _run_holdout(cfg, model, feat_cfg, topk, t0)
    elif mode == "submit":
        return _run_submit(cfg, model, feat_cfg, topk, t0)
    else:
        raise ValueError(f"未知 mode='{mode}'，应为 'holdout' 或 'submit'")


def _run_holdout(cfg, model, feat_cfg, topk, t0):
    print(f"[pipeline] mode=holdout model={cfg['model']} | 构建可信 holdout 特征...")
    tr, va, feat_cols, cat_cols, tg, vg = bf.build_holdout_features(
        cfg["datadir"], **_holdout_kwargs(feat_cfg))
    print(f"           train={tr.shape} val={va.shape} feat={len(feat_cols)} "
          f"cat={len(cat_cols)} ({time.time()-t0:.0f}s)")

    # 从 train 按整 group 切出独立 watch 做 early-stop；val 仅用于最终评估，
    # 不参与模型选择 -> 避免"用 val early-stop 又在 val 报分"的乐观泄漏。
    watch_frac = cfg.get("watch_frac", cfg.get("pipeline", {}).get("watch_frac", 0.2))
    n_watch_grp = int(len(tg) * watch_frac)
    n_fit_grp = len(tg) - n_watch_grp
    fit_rows = int(sum(tg[:n_fit_grp]))
    fit_df = tr.iloc[:fit_rows].reset_index(drop=True)
    watch_df = tr.iloc[fit_rows:].reset_index(drop=True)
    fit_groups = list(tg[:n_fit_grp])
    watch_groups = list(tg[n_fit_grp:])
    print(f"[pipeline] 训练模型 (fit={n_fit_grp}组 watch={n_watch_grp}组 early-stop)...")
    model.fit(fit_df, watch_df, feat_cols, cat_cols, fit_groups, watch_groups)
    print(f"           done ({time.time()-t0:.0f}s)")

    print("[pipeline] val NDCG@10 (val 未参与训练/选模)...")
    scores = model.predict_scores(va, feat_cols)
    overall, recalled, n_users, succ_ndcg = eval_ndcg(scores, va, topk)
    _print_report(overall, recalled, n_users, succ_ndcg, topk, t0)
    _print_importance(model, feat_cols)
    return {"ndcg": overall, "ndcg_success": succ_ndcg,
            "n_recalled": recalled, "n_users": n_users}


def _run_submit(cfg, model, feat_cfg, topk, t0):
    print(f"[pipeline] mode=submit model={cfg['model']} | 全量训练特征 (inner OOF)...")
    tr, feat_cols, cat_cols, tg = bf.build_oof_features(cfg["datadir"], **_oof_kwargs(feat_cfg))
    print(f"           train={tr.shape} ({time.time()-t0:.0f}s)")

    print("[pipeline] 训练模型 (全量, 无 early-stop)...")
    model.fit(tr, None, feat_cols, cat_cols, tg, None)
    print(f"           done ({time.time()-t0:.0f}s)")

    print("[pipeline] 构建 test 候选特征 (全量统计)...")
    te, _, _, _ = bf.build_test_features(cfg["datadir"], **_test_kwargs(feat_cfg))
    print(f"           test={te.shape} ({time.time()-t0:.0f}s)")

    print("[pipeline] 打分取 top-10 并对齐 sample_submission...")
    scores = model.predict_scores(te, feat_cols)
    pred = topk_predictions(scores, te, topk)

    sub = pd.read_csv(os.path.join(cfg["datadir"], "sample_submission.csv"), dtype=str)
    out = sub[["uid"]].merge(pred, on="uid", how="left")
    out["prediction"] = out["prediction"].fillna(sub["prediction"])
    out_path = cfg.get("out", "submissions/submission.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    lens = out["prediction"].str.split(",").apply(len)
    print(f"[pipeline] 写出 {out_path}: {len(out)} 行 | 每行iid数 {lens.min()}~{lens.max()} "
          f"| 缺失 {out['prediction'].isna().sum()} | 总耗时 {time.time()-t0:.0f}s")
    return {"out": out_path, "n_rows": len(out)}


# --------------------------------------------------------------------------- #
# config → build_features 参数映射（隔离两边的参数命名差异）
# --------------------------------------------------------------------------- #
def _holdout_kwargs(f):
    return dict(outer_folds=f.get("outer_folds", 5), outer_fold=f.get("outer_fold", 0),
                inner_folds=f.get("inner_folds", 4),
                candidate_k=f.get("candidate_k", 200), seed=f.get("seed", 42),
                collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200),
                train_candidate_k=f.get("train_candidate_k", 200),
                hard_negative_ratio=f.get("hard_negative_ratio", 0.75))


def _oof_kwargs(f):
    return dict(n_folds=f.get("n_folds", 4), seed=f.get("seed", 42),
                collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200),
                train_candidate_k=f.get("train_candidate_k", 200),
                hard_negative_ratio=f.get("hard_negative_ratio", 0.75))


def _test_kwargs(f):
    return dict(candidate_k=f.get("candidate_k", 200), collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200))


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
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
    # xgboost 返回 f0..fN -> 映射回特征名
    if "f0" in imp:
        imp = {feat_cols[int(k[1:])]: v for k, v in imp.items() if k.startswith("f")}
    items = sorted(imp.items(), key=lambda x: -x[1])[:12]
    print("Top-12 特征重要性(gain):")
    for name, v in items:
        print(f"  {name}: {v:.1f}")
