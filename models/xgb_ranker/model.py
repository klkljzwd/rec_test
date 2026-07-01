"""XGBoost LambdaMART 排序器 (rank:ndcg)。

使用原生 ``xgb.train``；holdout 有 early stopping 时，预测只使用
``best_iteration`` 之前的树，submit 无验证集时使用全部训练轮次。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from models.base import RankerModel
from models.registry import register


@register
class XGBRanker(RankerModel):
    name = "xgb_ranker"

    def fit(self, tr_df, va_df, feat_cols, cat_cols, tr_groups, va_groups):
        p = self.params
        Xtr = tr_df[feat_cols].to_numpy(np.float32)
        ytr = tr_df["label"].to_numpy(np.float32)
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dtr.set_group(np.asarray(tr_groups, np.int64))

        params = dict(
            objective="rank:ndcg", eval_metric="ndcg@10",
            learning_rate=p.get("lr", 0.05), max_depth=p.get("max_depth", 6),
            subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample", 0.8),
            min_child_weight=p.get("min_child_weight", 1.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            seed=p.get("seed", 42), verbosity=0, tree_method="hist",
        )
        evals = [(dtr, "train")]
        es = None
        if va_df is not None:
            Xva = va_df[feat_cols].to_numpy(np.float32)
            yva = va_df["label"].to_numpy(np.float32)
            dva = xgb.DMatrix(Xva, label=yva)
            dva.set_group(np.asarray(va_groups, np.int64))
            evals.append((dva, "val"))
            es = p.get("early_stopping", 50)
        self.model = xgb.train(
            params, dtr, num_boost_round=p.get("n_estimators", 606),
            evals=evals, early_stopping_rounds=es,
            verbose_eval=p.get("verbose_eval", 50))

    def predict_scores(self, df, feat_cols):
        X = df[feat_cols].to_numpy(np.float32)
        dmatrix = xgb.DMatrix(X)
        # xgb.train 的 early stopping 只记录最佳轮次，不会裁掉后续树；
        # Booster.predict 默认使用完整模型，需显式限制到 best_iteration。
        # best_iteration 从 0 开始，而 iteration_range 的右端为开区间。
        best_iteration = getattr(self.model, "best_iteration", None)
        if best_iteration is not None:
            return self.model.predict(
                dmatrix,
                iteration_range=(0, best_iteration + 1),
            )
        return self.model.predict(dmatrix)

    def feature_importance(self):
        gain = self.model.get_score(importance_type="gain")
        # xgboost 默认列名 f0..fN，无法直接映射回特征名；由 pipeline 用 feat_cols 索引转换
        return gain
