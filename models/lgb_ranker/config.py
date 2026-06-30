"""LGB 模型配置（全量自包含）。

包含使用 LGB 完成推荐任务整个流程的所有可调参数。cat_cols(u_cat/i_cat)在 LGBM
中走 categorical_feature，而非当数值。features/pipeline 值与原中央 config.py
默认逐字一致（约束 B）。
"""
MODEL_CONFIG = {
    "model_params": {
        "n_estimators": 1000,
        "lr": 0.05,
        "max_depth": 6,
        "num_leaves": 63,        # 2^max_depth 上限，可独立调
        "subsample": 0.8,
        "colsample": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "early_stopping": 50,
        "verbose_eval": 50,
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
