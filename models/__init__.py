"""模型层：排序器抽象 + 注册表。

设计：
  - RankerModel 统一接口：fit(train_df, val_df, ...) / predict_scores(df)。
    模型只负责"给每个 (用户,候选) 打分"；排序、取 top-k、算 NDCG 全在
    pipeline 层复用，与具体模型无关。
  - 每个模型一个文件，通过 @register 注册到 registry，config 用名字引用。
  - 加新模型 = 新建一个文件实现 RankerModel + @register，不改 pipeline。

模型接收 DataFrame（而非裸 numpy），以便自行处理 categorical_feature 等
列级语义；feat_cols / cat_cols 由 pipeline 传入。
"""
from .base import RankerModel
from .registry import register, get_model, list_models
# 导入各模型实现，触发 @register 注册
from . import xgb_ranker  # noqa: F401
from . import lgb_ranker  # noqa: F401

__all__ = ["RankerModel", "register", "get_model", "list_models"]
