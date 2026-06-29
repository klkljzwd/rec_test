"""LightGBM LambdaRank 排序器 (lambdarank)。

与 xgb_ranker 同接口(RankerModel)，pipeline 无需改动即可切换。
关键差异：cat_cols(u_cat/i_cat 整数编码)在此真正以类别特征喂给 LGBM
(categorical_feature)，而非像 XGBoost 那样当连续数值——这是 CLAUDE.md
里"LGBM 需 categorical_feature"的设计意图。

特征矩阵由 build_features 构建为 float32，cat_cols 列里存的是整数码；
故 fit/predict 前先把 cat_cols 列转 int32 再交给 LGBM（LGBM 要求类别列
为非负整数/ category dtype，float 会按连续值处理）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import RankerModel
from .registry import register


@register
class LGBRanker(RankerModel):
    name = "lgb_ranker"

    def _prepare(self, df: pd.DataFrame, feat_cols: list[str],
                 cat_cols: list[str]) -> pd.DataFrame:
        """取特征列，并把 cat_cols 转 int32 以激活 LGBM 类别语义。"""
        X = df[feat_cols].copy()
        cats = [c for c in cat_cols if c in X.columns]
        if cats:
            X[cats] = X[cats].astype(np.int32)
        return X

    def fit(self, tr_df, va_df, feat_cols, cat_cols, tr_groups, va_groups):
        self._cat_cols = cat_cols  # predict 复用
        p = self.params
        Xtr = self._prepare(tr_df, feat_cols, cat_cols)
        ytr = tr_df["label"].to_numpy(np.float32)
        dtr = lgb.Dataset(
            Xtr, label=ytr, group=np.asarray(tr_groups, np.int64),
            categorical_feature=[c for c in cat_cols if c in Xtr.columns] or "auto",
        )

        # XGB 风格参数名映射到 LGBM 原生键名，便于复用同一套调参习惯；
        # 同时支持 num_leaves 等 LGBM 专属参数覆盖。
        max_depth = p.get("max_depth", 6)
        params = {
            "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
            "learning_rate": p.get("lr", 0.05),
            "max_depth": max_depth,
            "num_leaves": p.get("num_leaves", min(2 ** max_depth, 63)),
            "bagging_fraction": p.get("subsample", 0.8), "bagging_freq": 1,
            "feature_fraction": p.get("colsample", 0.8),
            "min_child_weight": p.get("min_child_weight", 1.0),
            "lambda_l2": p.get("reg_lambda", 1.0),
            "seed": p.get("seed", 42), "verbosity": -1,
            "label_gain": [0, 1],  # 二元相关性(label 只 0/1)
        }

        valid_sets, valid_names = [dtr], ["train"]
        if va_df is not None:
            Xva = self._prepare(va_df, feat_cols, cat_cols)
            yva = va_df["label"].to_numpy(np.float32)
            dva = lgb.Dataset(
                Xva, label=yva, group=np.asarray(va_groups, np.int64),
                categorical_feature=[c for c in cat_cols if c in Xva.columns] or "auto",
                reference=dtr,
            )
            valid_sets.append(dva)
            valid_names.append("val")

        es = p.get("early_stopping", 50)
        callbacks = [lgb.log_evaluation(p.get("verbose_eval", 50))]
        if va_df is not None and es:
            callbacks.append(lgb.early_stopping(es, verbose=False))

        self.model = lgb.train(
            params, dtr, num_boost_round=p.get("n_estimators", 1000),
            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        )

    def predict_scores(self, df, feat_cols):
        X = self._prepare(df, feat_cols, getattr(self, "_cat_cols", []))
        # 用 early-stop 选出的最佳迭代数；无验证集时取全部轮数
        n = self.model.best_iteration if hasattr(self.model, "best_iteration") else -1
        return self.model.predict(X, num_iteration=n)

    def feature_importance(self):
        # LGBM 返回与列名对齐的 gain，无需 f0..fN 映射(pipeline 直接用)
        names = self.model.feature_name()
        gain = self.model.feature_importance(importance_type="gain")
        return {n: float(v) for n, v in zip(names, gain)}
