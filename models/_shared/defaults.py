"""流程/特征参数的参考默认值。

新模型 config.py 应**全量显式写**自己的 features/pipeline（不从本文件继承），
本文件仅作参考模板，方便新模型拷贝后按需修改。运行时不参与合并——每个模型
的 load.py 直接用自己的 config.py 全量内容。

值与原中央 config.py 的 FEATURES/PIPELINE 逐字一致（约束 B）。
"""

# 流程通用参数（参考默认）
PIPELINE = {
    "topk": 10,                # NDCG@k / 提交每用户取 top-k
    "seed": 42,
    "watch_frac": 0.2,         # holdout 从 train 切独立 watch 的比例(early-stop 用)
    "out": "submissions/submission.csv",  # submit 模式默认输出
}

# 特征构建参数（参考默认）
FEATURES = {
    # 候选与负采样
    "candidate_k": 25,        # val/test 召回候选数
    "train_candidate_k": 50,  # 训练group大小(1正+其余负)；与candidate_k同值即难度对齐
    "hard_negative_ratio": 1.0,  # 负样本中难负(召回top)占比，其余为随机负
    "score_weights": {        # 候选生成信号权重，可用深路径 --param 单独覆盖
        "pop": 2.0,
        "target_prior": 6.0,
        "repeat": 20.0,
        "collab": 2.0,
        "markov": 1.0,
        "htarget": 30.0,
        "user_cond": 15.0,
    },
    # 协同信号
    "collab": "auto",          # auto|ease|itemknn；auto 按 ease_max_items 选
    "ease_lambda": 250.0,
    "ease_max_items": 1500,    # auto 模式商品数超过此值走 itemknn(避免稠密求逆)
    "itemknn_k": 200,          # itemknn 每行保留 top-k 相似度
    # 折划分
    "outer_folds": 10,         # holdout 外层折数
    "outer_fold": 0,           # holdout 用第几个外层折做验证(0 起)
    "inner_folds": 4,          # holdout 内层 OOF 折数
    "n_folds": 4,              # submit 模式全量 OOF 折数
}
