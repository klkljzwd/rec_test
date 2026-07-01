"""Fusion 模型配置（全量自包含）。

xgb + lgb 分数融合：不实现 RankerModel，只提供 run.py（meta-runner），复用
XGBRanker/LGBRanker 的 fit/predict_scores，候选特征只构建一次。两模型超参各自
全量内嵌（xgb/lgb 子配置），features/pipeline 与 xgb_ranker 一致（同数据流程）。

融合方法（model_params.method）：
  - "rrf"（默认）: reciprocal rank fusion。各模型在用户 group 内排名，fused =
    sum_m w_m / (rrf_k + rank_m)。rank-based，尺度无关，最稳健。
  - "sum": 各模型分数在用户 group 内 min-max 归一后加权求和。

模型多样性来自两模型本身的差异（xgb rank:ndcg max_depth=5 lr=0.06 vs
lgb lambdarank max_depth=6 num_leaves=63 lr=0.05），不新增冗余信号，是当前
唯一不与 htarget/repeat 协同信号冲突的正交增益来源。
"""
MODEL_CONFIG = {
    "model_params": {
        "method": "rrf",          # rrf | sum
        "rrf_k": 60,              # RRF 常数（标准 60）
        "weights": [0.5, 0.5],    # [xgb, lgb] 权重
        "xgb": {
            "n_estimators": 606, "lr": 0.06, "max_depth": 5,
            "subsample": 0.8, "colsample": 0.8, "min_child_weight": 1.0,
            "reg_lambda": 1.0, "early_stopping": 50, "verbose_eval": 0,
        },
        "lgb": {
            "n_estimators": 1000, "lr": 0.05, "max_depth": 6, "num_leaves": 63,
            "subsample": 0.8, "colsample": 0.8, "min_child_weight": 1.0,
            "reg_lambda": 1.0, "early_stopping": 50, "verbose_eval": 0,
        },
    },
    "features": {
        "candidate_k": 25,
        "train_candidate_k": 50,
        "hard_negative_ratio": 1.0,
        "collab": "auto",
        "ease_lambda": 250.0,
        "ease_max_items": 1500,
        "itemknn_k": 200,
        "outer_folds": 10,
        "outer_fold": 0,
        "inner_folds": 4,
        "n_folds": 4,
    },
    "pipeline": {
        "topk": 10,
        "seed": 42,
        "watch_frac": 0.2,
        "out": "submissions/submission.csv",
    },
}
