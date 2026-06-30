"""模型层：排序器抽象 + 注册表 + 统一调度。

设计：
  - RankerModel 统一接口（候选表模型）：fit/predict_scores，类注册到 _REGISTRY。
  - 每模型一个目录（model.py/data.py/run.py），run.py 是统一入口，runner 注册
    到 _RUNNERS，dispatcher 路由。
  - 加新模型 = 新建目录 + 三文件 + @register(类)/@register_runner(run)，不改
    dispatcher/pipeline。

模型接收 DataFrame（而非裸 numpy），以便自行处理 categorical_feature 等
列级语义；feat_cols / cat_cols 由 run.py 经 data.py 传入。
"""
from .base import RankerModel
from .registry import (
    register, get_model, list_models,
    register_runner, get_runner,
)
# 导入各模型实现，触发 @register（类）+ @register_runner（runner）
from .xgb.model import XGBRanker  # noqa: F401
from .lgb.model import LGBRanker  # noqa: F401
from .xgb import run as _xgb_run  # noqa: F401  触发 @register_runner("xgb_ranker")
from .lgb import run as _lgb_run  # noqa: F401  触发 @register_runner("lgb_ranker")
from .dispatcher import run_experiment

__all__ = [
    "RankerModel", "register", "get_model", "list_models",
    "register_runner", "get_runner", "run_experiment",
]
