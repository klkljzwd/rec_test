"""DeepFM 模型配置（全量自包含）。

非序列深度学习推荐模型：FM（低阶二阶交叉）+ DNN（高阶），pointwise BCE。
塞进 RankerModel 候选表接口，走与 XGB/LGB 相同的 OOF/holdout 无泄漏协议，
可信可比。features/pipeline 与 xgb_ranker 一致（同数据流程）。
"""
MODEL_CONFIG = {
    "model_params": {
        "embedding_dim": 8,         # 每个 cat 列的 embedding 维度
        "hidden_dims": [64, 64],  # DNN 隐藏层
        "lr": 0.001,
        "batch_size": 4096,
        "epochs": 20,
        "early_stopping": 3,        # watch ndcg@10 连续不升的 epoch 数
        "dropout": 0.2,
        "device": "auto",           # auto(有GPU用GPU)/cpu/cuda，可被 run.py --device 覆盖
        "seed": 42,
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
