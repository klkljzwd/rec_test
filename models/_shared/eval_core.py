"""评估原语（约束 A：口径统一）。

候选表口径（1正+N负强塞）的 `ndcg_at_k`/`eval_ndcg`/`topk_predictions` 从
pipeline.py 原样搬入，函数体逐行等价（约束 B：不破坏现有分数）。

`eval_predictions` 是模型无关口径，供未来序列模型（gru4rec 等）直接产
{uid: [iid]} 调用，无需候选表；内部复用 `ndcg_at_k`，与候选表口径共享同一
NDCG 公式，保证横向可比。

折划分不在本模块（留数据层 build_features._make_folds），本模块只负责
"给定预测与 target 算指标"。
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd


def ndcg_at_k(pred_iids, target_iid, k=10):
    """单用户 NDCG@k：target 在 top-k 内则 1/log2(rank+2)，否则 0。

    与 pipeline.py 原版逐行等价。
    """
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

    与 pipeline.py 原版逐行等价。
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
    """由分数取每用户 top-k iid，返回 (uid, prediction) 表。

    与 pipeline.py 原版逐行等价。
    """
    g = df.assign(_score=scores)
    rows = []
    for uid, sub in g.groupby("uid", sort=False):
        top = sub.sort_values("_score", ascending=False)["iid"].head(k).tolist()
        if len(top) < k:  # 召回不足补齐（罕见）
            top += [iid for iid in sub["iid"] if iid not in top][:k - len(top)]
        rows.append((uid, ",".join(top)))
    return pd.DataFrame(rows, columns=["uid", "prediction"])


def eval_predictions(preds: dict, targets: dict, k=10):
    """模型无关口径：吃 {uid: [iid,...]} 预测 与 {uid: target_iid}，返回统一 metrics。

    供未来序列模型（gru4rec 等）使用——它们产 top-k iid 列表而非候选表分数。
    内部复用 ndcg_at_k，与候选表口径共享同一 NDCG 公式，保证横向可比。

    返回 dict: {ndcg@<k>, recall@<k>, succ_ndcg, n_recalled, n_users}。
    """
    ndcgs, succ, recalled = [], [], 0
    for uid, tgt in targets.items():
        pred = preds.get(uid, [])
        if tgt in pred[:k]:
            recalled += 1
            sc = ndcg_at_k(pred, tgt, k)
            ndcgs.append(sc)
            succ.append(sc)
        else:
            ndcgs.append(0.0)
    n_users = len(targets)
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{k}": recalled / n_users if n_users else 0.0,
        "succ_ndcg": float(np.mean(succ)) if succ else 0.0,
        "n_recalled": recalled,
        "n_users": n_users,
    }


def make_metrics(overall: float, succ_ndcg: float, recalled: int,
                 n_users: int, topk: int, extra: dict | None = None) -> dict:
    """统一 metrics 结构组装器：所有 run.py 产出同结构 dict。

    候选表模型用 overall/succ/recalled 填；序列模型用 eval_predictions 的结果填。
    """
    m = {
        f"ndcg@{topk}": overall,
        "succ_ndcg": succ_ndcg,
        "n_recalled": recalled,
        "n_users": n_users,
    }
    if extra:
        m.update(extra)
    return m
