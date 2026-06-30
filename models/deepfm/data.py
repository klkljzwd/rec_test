"""DeepFM 候选表数据处理：与 xgb_ranker/data.py 完全相同。

候选表模型共用同一套 build_features 产出，故映射逻辑与 split_watch 一致。
注意：holdout_kwargs/oof_kwargs 的兜底默认值不能"顺手修正"成模型 config.py
的 features 默认；实际值由 load() 合并后的 feat_cfg 提供。
"""
from __future__ import annotations

import build_features as bf


def holdout_kwargs(f: dict) -> dict:
    return dict(outer_folds=f.get("outer_folds", 5), outer_fold=f.get("outer_fold", 0),
                inner_folds=f.get("inner_folds", 4),
                candidate_k=f.get("candidate_k", 200), seed=f.get("seed", 42),
                collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200),
                train_candidate_k=f.get("train_candidate_k", 200),
                hard_negative_ratio=f.get("hard_negative_ratio", 0.75))


def oof_kwargs(f: dict) -> dict:
    return dict(n_folds=f.get("n_folds", 4), seed=f.get("seed", 42),
                collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200),
                train_candidate_k=f.get("train_candidate_k", 200),
                hard_negative_ratio=f.get("hard_negative_ratio", 0.75))


def test_kwargs(f: dict) -> dict:
    return dict(candidate_k=f.get("candidate_k", 200), collab_method=f.get("collab", "auto"),
                ease_lambda=f.get("ease_lambda", 250.0),
                ease_max_items=f.get("ease_max_items", 1500),
                itemknn_k=f.get("itemknn_k", 200))


def build_holdout(datadir, feat_cfg):
    """调 bf.build_holdout_features，返回 (tr, va, feat_cols, cat_cols, tg, vg)。"""
    return bf.build_holdout_features(datadir, **holdout_kwargs(feat_cfg))


def build_oof(datadir, feat_cfg):
    """调 bf.build_oof_features，返回 (tr, feat_cols, cat_cols, tg)。"""
    return bf.build_oof_features(datadir, **oof_kwargs(feat_cfg))


def build_test(datadir, feat_cfg):
    """调 bf.build_test_features，返回 (te, feat_cols, cat_cols, tg)。"""
    return bf.build_test_features(datadir, **test_kwargs(feat_cfg))


def split_watch(tr, tg, watch_frac):
    """从 train 按整 group 切出独立 watch 做 early-stop；val 不参与选模。

    返回 (fit_df, watch_df, fit_groups, watch_groups)。
    """
    n_watch_grp = int(len(tg) * watch_frac)
    n_fit_grp = len(tg) - n_watch_grp
    fit_rows = int(sum(tg[:n_fit_grp]))
    fit_df = tr.iloc[:fit_rows].reset_index(drop=True)
    watch_df = tr.iloc[fit_rows:].reset_index(drop=True)
    fit_groups = list(tg[:n_fit_grp])
    watch_groups = list(tg[n_fit_grp:])
    return fit_df, watch_df, fit_groups, watch_groups
