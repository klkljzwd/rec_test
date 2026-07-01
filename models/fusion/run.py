"""Fusion run：xgb + lgb 分数融合，holdout/submit 训练+评估+产出统一 metrics。

meta-runner：不实现 RankerModel，复用 XGBRanker/LGBRanker 的 fit/predict_scores，
候选特征只构建一次（两模型同特征流程）。评估走 eval_core（口径统一）。
@register_runner("fusion") 注册到 dispatcher。

融合是当前唯一不与 htarget/repeat 协同信号冲突的正交增益来源（模型多样性，
非新增冗余信号）。两模型分数尺度不同，故默认 RRF（rank-based 尺度无关）。
"""
from __future__ import annotations
import os
import time

import numpy as np
import pandas as pd

from models import get_model
from models.registry import register_runner
from models._shared import eval_core

from . import data as D


@register_runner("fusion")
def run(cfg: dict) -> dict:
    """统一入口：吃 cfg，返回标准 metrics dict。"""
    t0 = time.time()
    mode = cfg.get("mode", "holdout")
    feat_cfg = dict(cfg.get("features", {}))
    feat_cfg["seed"] = cfg.get("seed", 42)
    topk = cfg.get("topk", 10)
    mp = dict(cfg.get("model_params", {}))
    method = mp.get("method", "rrf")
    weights = mp.get("weights", [0.5, 0.5])
    rrf_k = mp.get("rrf_k", 60)
    seed = cfg.get("seed", 42)

    # 两子模型各自取内嵌超参 + seed
    xgb_params = dict(mp.get("xgb", {}))
    xgb_params.setdefault("seed", seed)
    lgb_params = dict(mp.get("lgb", {}))
    lgb_params.setdefault("seed", seed)

    if mode == "holdout":
        return _holdout(cfg, feat_cfg, topk, t0, method, weights, rrf_k, xgb_params, lgb_params)
    elif mode == "submit":
        return _submit(cfg, feat_cfg, topk, t0, method, weights, rrf_k, xgb_params, lgb_params)
    raise ValueError(f"未知 mode='{mode}'，应为 'holdout' 或 'submit'")


def _fuse(sx, sl, df, method, weights, rrf_k):
    """融合两模型分数。sx/sl 与 df 行对齐，df 含 uid 列按 group 排列。"""
    g = pd.DataFrame({"uid": df["uid"].to_numpy(), "sx": sx, "sl": sl})
    wx, wl = float(weights[0]), float(weights[1])
    if method == "sum":
        # 各模型分数在用户 group 内 min-max 归一后加权求和
        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else s * 0.0
        nx = g.groupby("uid", sort=False)["sx"].transform(_norm)
        nl = g.groupby("uid", sort=False)["sl"].transform(_norm)
        return (wx * nx + wl * nl).to_numpy(np.float32)
    # 默认 rrf：各模型 group 内降序排名，fused = sum w/(rrf_k + rank)
    rx = g.groupby("uid", sort=False)["sx"].rank(ascending=False, method="average")
    rl = g.groupby("uid", sort=False)["sl"].rank(ascending=False, method="average")
    fused = wx / (rrf_k + rx) + wl / (rrf_k + rl)
    return fused.to_numpy(np.float32)


def _holdout(cfg, feat_cfg, topk, t0, method, weights, rrf_k, xgb_params, lgb_params):
    print(f"[fusion] mode=holdout | method={method} weights={weights} rrf_k={rrf_k}")
    print(f"[fusion] 构建可信 holdout 特征（一次，两模型共享）...")
    tr, va, feat_cols, cat_cols, tg, vg = D.build_holdout(cfg["datadir"], feat_cfg)
    print(f"      train={tr.shape} val={va.shape} feat={len(feat_cols)} "
          f"cat={len(cat_cols)} ({time.time()-t0:.0f}s)")

    watch_frac = cfg.get("watch_frac", cfg.get("pipeline", {}).get("watch_frac", 0.2))
    fit_df, watch_df, fit_groups, watch_groups = D.split_watch(tr, tg, watch_frac)

    print(f"[fusion] 训练 xgb (fit={len(fit_groups)}组 watch={len(watch_groups)}组)...")
    xgb_model = get_model("xgb_ranker", xgb_params)
    xgb_model.fit(fit_df, watch_df, feat_cols, cat_cols, fit_groups, watch_groups)
    print(f"      xgb done ({time.time()-t0:.0f}s)")

    print(f"[fusion] 训练 lgb (同 fit/watch)...")
    lgb_model = get_model("lgb_ranker", lgb_params)
    lgb_model.fit(fit_df, watch_df, feat_cols, cat_cols, fit_groups, watch_groups)
    print(f"      lgb done ({time.time()-t0:.0f}s)")

    sx = xgb_model.predict_scores(va, feat_cols)
    sl = lgb_model.predict_scores(va, feat_cols)
    # 对照：两单模型各自 NDCG（确认两基线，便于诊断融合是否优于各自）
    ox, _, _, _ = eval_core.eval_ndcg(sx, va, topk)
    ol, _, _, _ = eval_core.eval_ndcg(sl, va, topk)
    print(f"[fusion] 单模型 val NDCG@{topk}: xgb={ox:.5f} lgb={ol:.5f}")

    fused = _fuse(sx, sl, va, method, weights, rrf_k)
    overall, recalled, n_users, succ_ndcg = eval_core.eval_ndcg(fused, va, topk)
    print(f"[fusion] 融合后 val NDCG@{topk}={overall:.5f} succ={succ_ndcg:.5f} "
          f"(xgb={ox:.5f} lgb={ol:.5f})")
    return eval_core.make_metrics(overall, succ_ndcg, recalled, n_users, topk,
                                  extra={"xgb_ndcg": ox, "lgb_ndcg": ol})


def _submit(cfg, feat_cfg, topk, t0, method, weights, rrf_k, xgb_params, lgb_params):
    print(f"[fusion] mode=submit | method={method} weights={weights} rrf_k={rrf_k}")
    print("[fusion] 全量训练特征 (inner OOF，一次)...")
    tr, feat_cols, cat_cols, tg = D.build_oof(cfg["datadir"], feat_cfg)
    print(f"      train={tr.shape} ({time.time()-t0:.0f}s)")

    print("[fusion] 训练 xgb (全量, 无 early-stop)...")
    xgb_model = get_model("xgb_ranker", xgb_params)
    xgb_model.fit(tr, None, feat_cols, cat_cols, tg, None)
    print(f"      xgb done ({time.time()-t0:.0f}s)")

    print("[fusion] 训练 lgb (全量, 无 early-stop)...")
    lgb_model = get_model("lgb_ranker", lgb_params)
    lgb_model.fit(tr, None, feat_cols, cat_cols, tg, None)
    print(f"      lgb done ({time.time()-t0:.0f}s)")

    print("[fusion] 构建 test 候选特征 (全量统计)...")
    te, _, _, _ = D.build_test(cfg["datadir"], feat_cfg)
    print(f"      test={te.shape} ({time.time()-t0:.0f}s)")

    sx = xgb_model.predict_scores(te, feat_cols)
    sl = lgb_model.predict_scores(te, feat_cols)
    fused = _fuse(sx, sl, te, method, weights, rrf_k)
    pred = eval_core.topk_predictions(fused, te, topk)

    sub = pd.read_csv(os.path.join(cfg["datadir"], "sample_submission.csv"), dtype=str)
    out = sub[["uid"]].merge(pred, on="uid", how="left")
    out["prediction"] = out["prediction"].fillna(sub["prediction"])
    out_path = cfg.get("out", "submissions/submission.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    lens = out["prediction"].str.split(",").apply(len)
    print(f"[fusion] 写出 {out_path}: {len(out)} 行 | 每行iid数 {lens.min()}~{lens.max()} "
          f"| 缺失 {out['prediction'].isna().sum()} | 总耗时 {time.time()-t0:.0f}s")
    return {"out": out_path, "n_rows": len(out)}
