"""排序器模型统一接口。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class RankerModel(ABC):
    """所有排序模型的基类。

    子类需实现:
        fit(tr_df, va_df, feat_cols, cat_cols, tr_groups, va_groups)
        predict_scores(df, feat_cols) -> np.ndarray

    约定:
      - 输入 DataFrame 含列: feat_cols(特征) + "label"(0/1) + "uid"/"iid"。
      - groups 是每用户的样本数列表(LambdaRank 的 group 参数)。
      - va_df 可为 None(提交模式无验证集 early-stop)。
      - predict_scores 返回与 df 行对齐的一维分数，越大越靠前。
    """

    name = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = None

    @abstractmethod
    def fit(self, tr_df: pd.DataFrame, va_df: pd.DataFrame | None,
            feat_cols: list[str], cat_cols: list[str],
            tr_groups: list[int], va_groups: list[int] | None):
        ...

    @abstractmethod
    def predict_scores(self, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
        ...

    def feature_importance(self) -> dict[str, float]:
        """可选：返回 {特征名: 重要性}。子类按需覆盖。"""
        return {}
