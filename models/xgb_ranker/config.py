"""XGB 模型配置（全量自包含）。

包含使用 XGB 完成推荐任务整个流程的所有可调参数：
  - model_params：XGB 超参
  - features：候选/负采样/协同/折划分（数据构建参数）
  - pipeline：topk/seed/watch_frac/out（流程参数）

本目录 load.py 负责把 MODEL_CONFIG 合并成完整 cfg。加新模型照此建 config.py +
load.py，无需改任何中央文件。features/pipeline 值与原中央 config.py 默认逐字
一致（约束 B：不破坏现有分数）。
"""
MODEL_CONFIG = {
    "model_params": {
        "n_estimators": 3000,
        "lr": 0.06,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "early_stopping": 300,
        "verbose_eval": 50,
    },
    "features": {
        "candidate_k": 25,
        "train_candidate_k": 50,
        "hard_negative_ratio": 1.0,
        "score_weights": {
            "pop": 2.0,
            "target_prior": 0.0,
            "repeat": 30.0,
            "collab": 2.0,
            "markov": 1.0,
            "htarget": 30.0,
            "user_cond": 15.0,
        },
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
        "watch_frac": 0.1,
        "out": "submissions/submission.csv",
    },
}
