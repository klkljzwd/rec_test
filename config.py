"""推荐任务统一配置。

结构：PIPELINE(流程通用) + FEATURES(特征构建) + MODELS(每模型一套全部参数)。
模型名即实验名：run.py --exp <模型名> 直接取该模型的全部参数 + FEATURES/PIPELINE
默认值合并后跑。无 EXPERIMENTS 间接层。

加载：get_config(model_name, overrides) -> 完整 cfg，传给 pipeline.run_experiment。
覆盖优先级：命令行 --param > MODELS[name] 的覆盖 > FEATURES/PIPELINE 默认。

加新模型：models/ 建 RankerModel 子类 + @register，这里 MODELS 加一项(可内嵌
features/pipeline 子块覆盖默认)。
命令行覆盖：run.py --param model_params.lr=0.1 / features.candidate_k=200。
"""
from __future__ import annotations
import copy

# --------------------------------------------------------------------------- #
# 流程通用参数
# --------------------------------------------------------------------------- #
PIPELINE = {
    "topk": 10,                # NDCG@k / 提交每用户取 top-k
    "seed": 42,
    "watch_frac": 0.2,         # holdout 从 train 切独立 watch 的比例(early-stop 用)
    "out": "submissions/submission.csv",  # submit 模式默认输出
}

# --------------------------------------------------------------------------- #
# 特征构建参数
# --------------------------------------------------------------------------- #
FEATURES = {
    # 候选与负采样
    "candidate_k": 20,        # val/test 召回候选数
    "train_candidate_k": 50,  # 训练group大小(1正+其余负)；与candidate_k同值即难度对齐
    "hard_negative_ratio": 0.75,  # 负样本中难负(召回top)占比，其余为随机负
    # 协同信号
    "collab": "auto",          # auto|ease|itemknn；auto 按 ease_max_items 选
    "ease_lambda": 250.0,
    "ease_max_items": 1500,    # auto 模式商品数超过此值走 itemknn(避免稠密求逆)
    "itemknn_k": 200,          # itemknn 每行保留 top-k 相似度
    # 折划分
    "outer_folds": 5,          # holdout 外层折数
    "outer_fold": 0,           # holdout 用第几个外层折做验证(0 起)
    "inner_folds": 4,          # holdout 内层 OOF 折数
    "n_folds": 4,              # submit 模式全量 OOF 折数
}

# --------------------------------------------------------------------------- #
# 各模型配置（key = 模型名，对应 models/registry）。
# 每个模型块含 model_params，可选内嵌 features / pipeline 子块覆盖默认。
# --------------------------------------------------------------------------- #
MODELS = {
    "xgb_ranker": {
        "model_params": {
            "n_estimators": 606,
            "lr": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "early_stopping": 50,
            "verbose_eval": 50,
        },
        # features/pipeline 用 FEATURES/PIPELINE 默认(已含 train_candidate_k=200 难度对齐)
    },
    # 示例：加新模型时照此添加（model_params 必填，features/pipeline 可选覆盖）
    # "lgb_ranker": {
    #     "model_params": {"n_estimators": 1000, "lr": 0.05, ...},
    # },
}


# --------------------------------------------------------------------------- #
# 解析与合并
# --------------------------------------------------------------------------- #
def list_models() -> list[str]:
    return sorted(MODELS)


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_config(model_name: str, overrides: dict | None = None) -> dict:
    """合并出完整配置：模型名取 MODELS[name]，与 FEATURES/PIPELINE 默认合并。

    返回: {model, model_params, features, pipeline, topk, seed, watch_frac, out}。
    """
    if model_name not in MODELS:
        raise KeyError(f"未知模型 '{model_name}'，可用: {list_models()}")
    m = copy.deepcopy(MODELS[model_name])
    cfg = {
        "model": model_name,
        "model_params": _deep_update(m.get("model_params", {}), m.get("model_params_override", {})),
        "features": _deep_update(FEATURES, m.get("features")),
        "pipeline": _deep_update(PIPELINE, m.get("pipeline")),
    }
    # 顶层快捷字段(便于 pipeline 直接读)
    for k in ("topk", "seed", "watch_frac", "out"):
        cfg[k] = cfg["pipeline"][k]
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """深路径覆盖：{'model_params.lr':0.1, 'features.neg':30, 'topk':5}。"""
    for path, val in overrides.items():
        parts = path.split(".")
        if parts[0] in ("model", "topk", "seed", "watch_frac", "out", "mode", "datadir"):
            cfg[parts[0]] = val
            if parts[0] in ("topk", "seed", "watch_frac", "out"):
                cfg["pipeline"][parts[0]] = val
            continue
        if parts[0] == "model_params":
            _set_nested(cfg["model_params"], parts[1:], val)
        elif parts[0] == "features":
            _set_nested(cfg["features"], parts[1:], val)
        elif parts[0] == "pipeline":
            _set_nested(cfg["pipeline"], parts[1:], val)
            if len(parts) == 2 and parts[1] in ("topk", "seed", "watch_frac", "out"):
                cfg[parts[1]] = val
        else:
            raise KeyError(f"无法识别的覆盖路径: {path}")
    return cfg


def _set_nested(d: dict, keys: list[str], val):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


if __name__ == "__main__":
    import json
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "xgb_ranker"
    print(f"=== 模型 {name} 合并后配置 ===")
    print(json.dumps(get_config(name), ensure_ascii=False, indent=2, default=str))
