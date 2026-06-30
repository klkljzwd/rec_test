"""XGBoost LambdaMART 排序器 (rank:pairwise)。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from .base import RankerModel
from .registry import register


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
        return self.model.predict(xgb.DMatrix(X))

    def feature_importance(self):
        gain = self.model.get_score(importance_type="gain")
        # xgboost 默认列名 f0..fN，无法直接映射回特征名；由 pipeline 用 feat_cols 索引转换
        return gain
