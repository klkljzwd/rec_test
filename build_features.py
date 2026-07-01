"""推荐任务：无泄漏 OOF、可信 holdout 与测试特征构建。

本文件明确区分三种用途：

1. ``build_oof_features``
   为全部训练用户生成 cross-fitted 特征，用于训练最终二级排序器。每个用户的
   target 相关统计只来自其他折。该表不能再随意切分后宣称得到无偏验证分数。

2. ``build_holdout_features``
   先做 Outer train/validation；Outer train 内部再做 Inner OOF 以训练排序器，
   Outer validation 特征只使用 Outer train 统计，适合可信离线 NDCG 评估。

3. ``build_test_features``
   用全部训练用户重建统计，为测试用户生成真实候选及与 OOF 完全一致的特征列。

默认候选生成融合 popularity、repeat、协同、Markov、HTarget 和用户特征条件
分布；不会过滤历史商品，因为本任务允许重复推荐。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import core.feature_core as core


DEFAULT_SCORE_WEIGHTS = {
    "pop": 2.0,
    "target_prior": 6.0,
    "repeat": 20.0,
    "collab": 2.0,
    "markov": 1.0,
    "htarget": 30.0,
    "user_cond": 15.0,
}


def _feature_schema(ucols, icols):
    feat_cols = (
        [
            "in_hist",
            "count",
            "count_norm",
            "count_log_norm",
            "is_last_item",
            "last_position_norm",
            "run_count",
            "run_count_norm",
            "popularity",
            "is_known_target",
            "target_freq",
            "target_freq_log",
            "repeat",
            "collab_score",
            "markov",
            "htarget",
        ]
        + [f"u_{column}" for column in ucols]
        + [f"i_{column}" for column in icols]
        + [f"hist_{column}_share" for column in icols]
        + [f"last_{column}_match" for column in icols]
        + ["ufeat_target_cond"]
        + [f"ucond_{column}" for column in ucols]
        + ["lastcat_target_cond"]
    )
    cat_cols = [f"u_{column}" for column in ucols]
    cat_cols += [f"i_{column}" for column in icols if column.startswith("i_cat")]
    return feat_cols, cat_cols


def _load_context(datadir):
    data = core.load_data(datadir)
    train, test = data["train"], data["test"]
    user, item = data["user"], data["item"]
    items, iid2idx, n_item = data["items"], data["iid2idx"], data["n_item"]

    records, uids, runs = [], [], []
    columns = ["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"]
    for uid, target, raw, dedup, counts in train[columns].itertuples(index=False, name=None):
        records.append((core._split(dedup), core._parse_counts(counts), target or None))
        uids.append(uid)
        runs.append(core._to_runs(core._split(raw)))

    valid = [
        idx
        for idx, (_, _, target) in enumerate(records)
        if target is not None and target in iid2idx
    ]
    ucols, user_raw, user_encoded = core.prepare_user_features(user)
    icols = [column for column in item.columns if column != "iid"]
    item_features = (
        item[icols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
        if icols
        else np.zeros((n_item, 0), dtype=np.float32)
    )
    test_lengths = np.asarray(
        [len(core._split(value)) for value in test["item_seq_dedup"]],
        dtype=np.int32,
    )
    feat_cols, cat_cols = _feature_schema(ucols, icols)
    return {
        "data": data,
        "train": train,
        "test": test,
        "user": user,
        "item": item,
        "items": items,
        "item_ids": np.asarray(items, dtype=object),
        "iid2idx": iid2idx,
        "n_item": n_item,
        "records": records,
        "uids": uids,
        "runs": runs,
        "valid": valid,
        "ucols": ucols,
        "icols": icols,
        "user_raw": user_raw,
        "user_encoded": user_encoded,
        "item_features": item_features,
        "test_lengths": test_lengths,
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
    }


def _make_folds(indices, n_folds, seed):
    if n_folds < 2:
        raise ValueError("n_folds 必须至少为 2")
    indices = np.asarray(indices, dtype=np.int64).copy()
    if len(indices) < n_folds:
        raise ValueError(f"有效用户数({len(indices)})少于折数({n_folds})")
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def _build_fold_stats(
    context,
    train_indices,
    collab_method,
    ease_lambda,
    ease_max_items,
    itemknn_k,
):
    records = [context["records"][idx] for idx in train_indices]
    uids = [context["uids"][idx] for idx in train_indices]
    n_item, iid2idx = context["n_item"], context["iid2idx"]

    pop = core.build_popularity(records, n_item, iid2idx)
    interaction = core.build_user_item_matrix(records, n_item, iid2idx)
    collab, selected = core.build_collab(
        interaction,
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
    )
    markov = core.build_markov(records, n_item, iid2idx)
    htarget = core.build_htarget(records, n_item, iid2idx)
    user_cond = core.build_userfeat_target_cond(
        records,
        uids,
        context["user_raw"],
        context["ucols"],
        n_item,
        iid2idx,
    )
    target_pool = np.asarray(
        sorted(
            {
                iid2idx[target]
                for _, _, target in records
                if target is not None and target in iid2idx
            }
        ),
        dtype=np.int64,
    )
    # target 先验：商品作为训练 target 的频次（与 popularity 曝光度互补）。
    # 按折统计 -> 验证折 target 不进自身特征，无泄漏。
    target_count = core.build_target_freq(records, n_item, iid2idx)
    target_count_max = int(target_count.max()) if target_count.size else 0
    return {
        "pop": pop,
        "collab": collab,
        "markov": markov,
        "htarget": htarget,
        "user_cond": user_cond,
        "target_pool": target_pool,
        "target_count": target_count,
        "target_count_max": target_count_max,
        "selected_collab": selected,
    }


def _user_full_scores(ded, counts, context, stats, decay=0.95):
    """计算单用户在全商品集合上的四路历史信号。"""
    n_item = context["n_item"]
    items, iid2idx = context["items"], context["iid2idx"]
    collab, markov, htarget = stats["collab"], stats["markov"], stats["htarget"]
    zero = np.zeros(n_item, dtype=np.float32)
    idxs = [iid2idx[iid] for iid in ded if iid in iid2idx]
    if not idxs:
        return zero, zero.copy(), zero.copy(), zero.copy()

    length = len(idxs)
    repeat = np.zeros(n_item, dtype=np.float32)
    for position, item_idx in enumerate(idxs):
        weight = decay ** (length - 1 - position) * math.sqrt(counts.get(items[item_idx], 1))
        repeat[item_idx] = max(repeat[item_idx], weight)
    maximum = float(repeat.max())
    if maximum > 0:
        repeat /= maximum

    collaborative = np.zeros(n_item, dtype=np.float32)
    for item_idx in dict.fromkeys(idxs):
        weight = math.log1p(counts.get(items[item_idx], 1))
        if sp.issparse(collab):
            row = collab.getrow(item_idx)
            collaborative[row.indices] += weight * row.data
        else:
            collaborative += weight * collab[item_idx]
    minimum, maximum = float(collaborative.min()), float(collaborative.max())
    if maximum > minimum:
        collaborative = (collaborative - minimum) / (maximum - minimum)

    markov_score = np.zeros(n_item, dtype=np.float32)
    for rank, item_idx in enumerate(reversed(idxs[-3:])):
        row = markov.getrow(item_idx)
        markov_score[row.indices] += (0.7**rank) * row.data
    maximum = float(markov_score.max())
    if maximum > 0:
        markov_score /= maximum

    htarget_score = np.zeros(n_item, dtype=np.float32)
    for position, item_idx in enumerate(idxs):
        weight = decay ** (length - 1 - position) * math.sqrt(counts.get(items[item_idx], 1))
        row = htarget.getrow(item_idx)
        htarget_score[row.indices] += weight * row.data
    maximum = float(htarget_score.max())
    if maximum > 0:
        htarget_score /= maximum

    return repeat, collaborative, markov_score, htarget_score


def _user_condition_score(user_values, context, stats, normalize=True):
    score = np.zeros(context["n_item"], dtype=np.float32)
    if user_values is None:
        return score
    for column, value in zip(context["ucols"], user_values):
        distribution = stats["user_cond"].get((column, value))
        if distribution is not None:
            score += distribution
    if normalize:
        maximum = float(score.max())
        if maximum > 0:
            score /= maximum
    return score


def _candidate_sequence_features(candidates, ded, counts, context):
    """候选在截断历史中的近期性与跨 run 重复特征。

    ``ded`` 是 run-level 序列：连续重复已压成一个位置，但同一商品在非连续
    run 中可多次出现。相比总次数 ``count``，run_count 和最后出现位置能区分
    "一次连续刷很多次"与"跨多个阶段反复回来"两种复购模式。
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_iids = [context["items"][idx] for idx in candidates]
    size = len(candidates)
    if not ded:
        zeros = np.zeros(size, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()

    last_position: dict[str, int] = {}
    run_counts: dict[str, int] = {}
    for position, iid in enumerate(ded):
        if iid not in context["iid2idx"]:
            continue
        last_position[iid] = position
        run_counts[iid] = run_counts.get(iid, 0) + 1

    length = max(len(ded), 1)
    maximum_runs = max(run_counts.values(), default=1)
    last_iid = ded[-1]
    is_last = np.asarray([iid == last_iid for iid in candidate_iids], dtype=np.float32)
    # 未出现候选为 0；历史中越靠近末尾越接近 1。
    position_norm = np.asarray(
        [(last_position[iid] + 1) / length if iid in last_position else 0.0
         for iid in candidate_iids],
        dtype=np.float32,
    )
    run_count = np.asarray([run_counts.get(iid, 0) for iid in candidate_iids], dtype=np.float32)
    run_count_norm = run_count / float(maximum_runs)
    return is_last, position_norm, run_count, run_count_norm


def _history_item_profile_features(candidates, ded, counts, context):
    """用户历史商品属性分布与候选属性的匹配度。

    对每个 item 元数据列输出两类特征：
      - hist_<col>_share：历史加权质量中，与候选属性值相同的占比；
      - last_<col>_match：候选与最后一个历史商品的属性是否相同。

    权重使用 sqrt(count)，既保留复购强度，又避免极端 count 完全支配画像。
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    n_candidates, n_columns = len(candidates), len(context["icols"])
    shares = np.zeros((n_candidates, n_columns), dtype=np.float32)
    last_matches = np.zeros((n_candidates, n_columns), dtype=np.float32)
    if not ded or n_columns == 0:
        return shares, last_matches

    # 同一商品只进入画像一次，强度由累计 count 表示。
    history_indices = list(dict.fromkeys(
        context["iid2idx"][iid] for iid in ded if iid in context["iid2idx"]
    ))
    if not history_indices:
        return shares, last_matches

    history_weights = np.asarray(
        [math.sqrt(counts.get(context["items"][idx], 1)) for idx in history_indices],
        dtype=np.float32,
    )
    total_weight = float(history_weights.sum())
    candidate_values = context["item_features"][candidates]
    history_values = context["item_features"][history_indices]
    last_idx = context["iid2idx"].get(ded[-1])

    for column_idx in range(n_columns):
        mass_by_value: dict[float, float] = {}
        for value, weight in zip(history_values[:, column_idx], history_weights):
            key = float(value)
            mass_by_value[key] = mass_by_value.get(key, 0.0) + float(weight)
        if total_weight > 0:
            shares[:, column_idx] = np.asarray(
                [mass_by_value.get(float(value), 0.0) / total_weight
                 for value in candidate_values[:, column_idx]],
                dtype=np.float32,
            )
        if last_idx is not None:
            last_value = context["item_features"][last_idx, column_idx]
            last_matches[:, column_idx] = (candidate_values[:, column_idx] == last_value)

    return shares, last_matches


def _user_condition_components(user_values, candidates, context, stats):
    """拆出每个用户属性对应的 target 条件先验，避免只保留求和结果。"""
    result = np.zeros((len(candidates), len(context["ucols"])), dtype=np.float32)
    if user_values is None:
        return result
    for column_idx, (column, value) in enumerate(zip(context["ucols"], user_values)):
        distribution = stats["user_cond"].get((column, value))
        if distribution is not None:
            result[:, column_idx] = distribution[candidates]
    return result


def _all_item_scores(ded, counts, uid, context, stats, score_weights=None):
    weights = DEFAULT_SCORE_WEIGHTS if score_weights is None else score_weights
    repeat, collaborative, markov, htarget = _user_full_scores(ded, counts, context, stats)
    user_cond = _user_condition_score(context["user_raw"].get(uid), context, stats, normalize=True)
    # target 先验参与候选生成，而不只作为 ranker 特征。统计严格来自当前训练折；
    # log 归一减弱头部 target 的垄断，同时让冷启动用户优先覆盖已知 target 池。
    target_prior = np.zeros(context["n_item"], dtype=np.float32)
    target_count_max = float(stats["target_count_max"])
    if target_count_max > 0:
        target_prior = (
            np.log1p(stats["target_count"].astype(np.float32))
            / math.log1p(target_count_max)
        )
    score = (
        weights["pop"] * stats["pop"]
        + weights.get("target_prior", 0.0) * target_prior
        + weights["repeat"] * repeat
        + weights["collab"] * collaborative
        + weights["markov"] * markov
        + weights["htarget"] * htarget
        + weights["user_cond"] * user_cond
    )
    return score.astype(np.float32, copy=False)


def _topk(score, k):
    k = min(max(int(k), 1), len(score))
    if k == len(score):
        return np.argsort(-score, kind="stable").astype(np.int64)
    indices = np.argpartition(-score, k - 1)[:k]
    return indices[np.argsort(-score[indices], kind="stable")].astype(np.int64)


def _generate_candidates(ded, counts, uid, context, stats, k, score_weights=None):
    return _topk(_all_item_scores(ded, counts, uid, context, stats, score_weights), k)


def _assemble_features(candidates, ded, counts, uid, context, stats):
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_iids = [context["items"][idx] for idx in candidates]
    maximum_count = max(counts.values(), default=1)
    in_history = np.asarray([iid in counts for iid in candidate_iids], dtype=np.float32)
    count = np.asarray([counts.get(iid, 0) for iid in candidate_iids], dtype=np.float32)
    count_norm = count / maximum_count
    count_log_norm = np.log1p(count) / math.log1p(maximum_count)
    is_last, last_position_norm, run_count, run_count_norm = _candidate_sequence_features(
        candidates, ded, counts, context)

    components = core.candidate_components(
        ded,
        counts,
        candidates,
        context["iid2idx"],
        context["items"],
        stats["collab"],
        stats["markov"],
        stats["htarget"],
    )
    repeat, collaborative, markov, htarget, last_cond = components
    user_encoded = context["user_encoded"].get(uid)
    if user_encoded is None:
        user_features = np.full((len(candidates), len(context["ucols"])), -1.0, dtype=np.float32)
    else:
        user_features = np.broadcast_to(user_encoded, (len(candidates), len(context["ucols"])))
    user_cond = _user_condition_score(
        context["user_raw"].get(uid),
        context,
        stats,
        normalize=False,
    )[candidates]
    user_cond_components = _user_condition_components(
        context["user_raw"].get(uid), candidates, context, stats)
    history_profile, last_profile_match = _history_item_profile_features(
        candidates, ded, counts, context)

    # target 先验三列：是否曾作训练 target / 频次(归一) / 频次(log 归一)。
    tc = stats["target_count"][candidates].astype(np.float32)
    tc_max = float(stats["target_count_max"])
    is_known_target = (tc > 0).astype(np.float32)
    if tc_max > 0:
        target_freq = tc / tc_max
        target_freq_log = np.log1p(tc) / np.log1p(tc_max)
    else:
        target_freq = np.zeros_like(tc)
        target_freq_log = np.zeros_like(tc)

    columns = [
        in_history[:, None],
        count[:, None],
        count_norm[:, None],
        count_log_norm[:, None],
        is_last[:, None],
        last_position_norm[:, None],
        run_count[:, None],
        run_count_norm[:, None],
        stats["pop"][candidates, None],
        is_known_target[:, None],
        target_freq[:, None],
        target_freq_log[:, None],
        repeat[:, None],
        collaborative[:, None],
        markov[:, None],
        htarget[:, None],
        user_features,
        context["item_features"][candidates],
        history_profile,
        last_profile_match,
        user_cond[:, None],
        user_cond_components,
        last_cond[:, None],
    ]
    result = np.hstack(columns).astype(np.float32, copy=False)
    if result.shape[1] != len(context["feat_cols"]):
        raise RuntimeError(
            f"特征列错位: matrix={result.shape[1]} schema={len(context['feat_cols'])}"
        )
    return result


def _sample_negatives(
    positive,
    ranked_candidates,
    target_pool,
    n_item,
    neg,
    rng,
    hard_ratio,
):
    """从真实召回候选采难负样本，并以训练折target池/全商品补足。"""
    if neg <= 0:
        return np.empty(0, dtype=np.int64)
    hard_count = min(neg, max(0, int(round(neg * hard_ratio))))
    selected: list[int] = []
    used = {int(positive)}
    for candidate in ranked_candidates:
        candidate = int(candidate)
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
            if len(selected) >= hard_count:
                break

    random_pool = [int(idx) for idx in target_pool if int(idx) not in used]
    rng.shuffle(random_pool)
    for candidate in random_pool:
        selected.append(candidate)
        used.add(candidate)
        if len(selected) >= neg:
            break

    if len(selected) < neg:
        catalog = np.arange(n_item, dtype=np.int64)
        rng.shuffle(catalog)
        for candidate in catalog:
            candidate = int(candidate)
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                if len(selected) >= neg:
                    break
    if len(selected) < neg:
        raise ValueError(f"合法负样本不足: need={neg}, got={len(selected)}")
    return np.asarray(selected[:neg], dtype=np.int64)


def _build_oof_from_context(
    context,
    source_indices,
    n_folds,
    seed,
    collab_method,
    ease_lambda,
    ease_max_items,
    itemknn_k,
    train_candidate_k,
    hard_negative_ratio,
):
    """构建 cross-fitted 训练特征。

    每用户 group = train_candidate_k 个候选 = 1 个正样本 + (train_candidate_k-1) 个负样本。
    负样本构成由 hard_negative_ratio 控制：该比例从真实召回 top 取难负，其余从训练折
    target_pool 取随机负(不够用全商品兜底)。train_candidate_k 与 val/test 的 candidate_k
    取同值即"难度对齐"。无新旧模式之分，参数设什么就是什么。
    """
    source_indices = [int(idx) for idx in source_indices]
    folds = _make_folds(source_indices, n_folds, seed)
    group_size = train_candidate_k
    neg = group_size - 1
    n_item = context["n_item"]
    total_rows = len(source_indices) * group_size
    features = np.empty((total_rows, len(context["feat_cols"])), dtype=np.float32)
    labels = np.empty(total_rows, dtype=np.float32)
    output_uids = np.empty(total_rows, dtype=object)
    output_iids = np.empty(total_rows, dtype=object)
    fold_ids = np.empty(total_rows, dtype=np.int16)
    groups = []
    cursor = 0

    for fold_id, validation_indices in enumerate(folds):
        validation_set = {int(idx) for idx in validation_indices}
        train_indices = [idx for idx in source_indices if idx not in validation_set]
        stats = _build_fold_stats(
            context,
            train_indices,
            collab_method,
            ease_lambda,
            ease_max_items,
            itemknn_k,
        )
        rng = np.random.RandomState(seed * 1000 + fold_id)
        for record_idx in validation_indices:
            record_idx = int(record_idx)
            _, _, target = context["records"][record_idx]
            uid = context["uids"][record_idx]
            target_idx = context["iid2idx"][target]
            history_length = int(rng.choice(context["test_lengths"]))
            ded, counts = core.truncate_runs(context["runs"][record_idx], history_length)

            # 召回足够多的候选作为难负来源池(至少要能凑够 neg 个难负)
            ranked = _generate_candidates(
                ded, counts, uid, context, stats, min(max(neg, 1), n_item))
            negatives = _sample_negatives(
                target_idx, ranked, stats["target_pool"], n_item,
                neg, rng, hard_negative_ratio)
            candidates = np.concatenate(([target_idx], negatives)).astype(np.int64)
            candidates = candidates[rng.permutation(len(candidates))]
            group_labels = (candidates == target_idx).astype(np.float32)
            group_features = _assemble_features(candidates, ded, counts, uid, context, stats)

            end = cursor + group_size
            features[cursor:end] = group_features
            labels[cursor:end] = group_labels
            output_uids[cursor:end] = uid
            output_iids[cursor:end] = context["item_ids"][candidates]
            fold_ids[cursor:end] = fold_id
            groups.append(group_size)
            cursor = end
        print(
            f"[OOF] fold {fold_id + 1}/{n_folds} | collab={stats['selected_collab']} "
            f"train={len(train_indices)} val={len(validation_indices)} "
            f"group_size={group_size} hard_ratio={hard_negative_ratio}",
            flush=True,
        )

    frame = pd.DataFrame(features, columns=context["feat_cols"])
    frame["uid"] = output_uids
    frame["iid"] = output_iids
    frame["label"] = labels
    frame["fold_id"] = fold_ids
    return frame, list(context["feat_cols"]), list(context["cat_cols"]), groups


def build_oof_features(
    datadir,
    n_folds=5,
    seed=42,
    collab_method="auto",
    ease_lambda=250.0,
    ease_max_items=1500,
    itemknn_k=200,
    train_candidate_k=200,
    hard_negative_ratio=0.75,
):
    """为最终排序器训练生成全量 cross-fitted 特征。

    每用户 group=train_candidate_k(1正+其余负)，负样本难/随机比由 hard_negative_ratio 控制。
    """
    start = time.perf_counter()
    context = _load_context(datadir)
    result = _build_oof_from_context(
        context,
        context["valid"],
        n_folds,
        seed,
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
        train_candidate_k,
        hard_negative_ratio,
    )
    print(f"[OOF] 完成 rows={len(result[0])} elapsed={time.perf_counter() - start:.1f}s")
    return result


def _build_validation_features(
    context,
    validation_indices,
    stats,
    candidate_k,
    seed,
):
    """不强塞正样本，为Outer validation构建真实候选及标签。"""
    candidate_k = min(candidate_k, context["n_item"])
    total_rows = len(validation_indices) * candidate_k
    features = np.empty((total_rows, len(context["feat_cols"])), dtype=np.float32)
    labels = np.zeros(total_rows, dtype=np.float32)
    output_uids = np.empty(total_rows, dtype=object)
    output_iids = np.empty(total_rows, dtype=object)
    history_lengths = np.empty(total_rows, dtype=np.int32)
    recall_ranks = np.empty(total_rows, dtype=np.int32)
    groups = []
    rng = np.random.RandomState(seed)
    cursor = 0

    for record_idx in validation_indices:
        record_idx = int(record_idx)
        _, _, target = context["records"][record_idx]
        uid = context["uids"][record_idx]
        target_idx = context["iid2idx"][target]
        length = int(rng.choice(context["test_lengths"]))
        ded, counts = core.truncate_runs(context["runs"][record_idx], length)
        candidates = _generate_candidates(ded, counts, uid, context, stats, candidate_k)
        group_features = _assemble_features(candidates, ded, counts, uid, context, stats)
        end = cursor + candidate_k
        features[cursor:end] = group_features
        labels[cursor:end] = (candidates == target_idx).astype(np.float32)
        output_uids[cursor:end] = uid
        output_iids[cursor:end] = context["item_ids"][candidates]
        history_lengths[cursor:end] = length
        recall_ranks[cursor:end] = np.arange(candidate_k, dtype=np.int32)
        groups.append(candidate_k)
        cursor = end

    frame = pd.DataFrame(features, columns=context["feat_cols"])
    frame["uid"] = output_uids
    frame["iid"] = output_iids
    frame["label"] = labels
    frame["history_len"] = history_lengths
    frame["recall_rank"] = recall_ranks
    return frame, groups


def build_holdout_features(
    datadir,
    outer_folds=5,
    outer_fold=0,
    inner_folds=4,
    candidate_k=200,
    seed=42,
    collab_method="auto",
    ease_lambda=250.0,
    ease_max_items=1500,
    itemknn_k=200,
    train_candidate_k=200,
    hard_negative_ratio=0.75,
):
    """生成一套无二级泄漏的排序器训练集与Outer validation集。

    训练group=train_candidate_k(1正+其余负,难/随机比=hard_negative_ratio)；
    val候选数=candidate_k。两者取同值即难度对齐。
    """
    context = _load_context(datadir)
    outer = _make_folds(context["valid"], outer_folds, seed)
    if outer_fold < 0 or outer_fold >= len(outer):
        raise ValueError(f"outer_fold应在[0,{len(outer)-1}]内")
    validation_indices = [int(idx) for idx in outer[outer_fold]]
    validation_set = set(validation_indices)
    train_indices = [idx for idx in context["valid"] if idx not in validation_set]

    train_result = _build_oof_from_context(
        context,
        train_indices,
        inner_folds,
        seed + 101,
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
        train_candidate_k,
        hard_negative_ratio,
    )
    stats = _build_fold_stats(
        context,
        train_indices,
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
    )
    validation_frame, validation_groups = _build_validation_features(
        context,
        validation_indices,
        stats,
        candidate_k,
        seed + 202,
    )
    train_frame, feat_cols, cat_cols, train_groups = train_result
    print(
        f"[holdout] outer_fold={outer_fold} train_users={len(train_indices)} "
        f"val_users={len(validation_indices)} val_recall@{candidate_k}="
        f"{validation_frame.groupby('uid', sort=False)['label'].max().mean():.5f}",
        flush=True,
    )
    return (
        train_frame,
        validation_frame,
        feat_cols,
        cat_cols,
        train_groups,
        validation_groups,
    )


def build_test_features(
    datadir,
    candidate_k=200,
    collab_method="auto",
    ease_lambda=250.0,
    ease_max_items=1500,
    itemknn_k=200,
):
    """使用全训练统计构建测试候选特征，列结构与 OOF 完全一致。"""
    start = time.perf_counter()
    context = _load_context(datadir)
    stats = _build_fold_stats(
        context,
        context["valid"],
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
    )
    candidate_k = min(candidate_k, context["n_item"])
    test = context["test"]
    total_rows = len(test) * candidate_k
    features = np.empty((total_rows, len(context["feat_cols"])), dtype=np.float32)
    output_uids = np.empty(total_rows, dtype=object)
    output_iids = np.empty(total_rows, dtype=object)
    recall_ranks = np.empty(total_rows, dtype=np.int32)
    groups = []
    cursor = 0

    columns = ["uid", "item_seq_dedup", "item_seq_counts"]
    for order, (uid, dedup_text, count_text) in enumerate(
        test[columns].itertuples(index=False, name=None),
        start=1,
    ):
        ded = core._split(dedup_text)
        counts = core._parse_counts(count_text)
        candidates = _generate_candidates(ded, counts, uid, context, stats, candidate_k)
        group_features = _assemble_features(candidates, ded, counts, uid, context, stats)
        end = cursor + candidate_k
        features[cursor:end] = group_features
        output_uids[cursor:end] = uid
        output_iids[cursor:end] = context["item_ids"][candidates]
        recall_ranks[cursor:end] = np.arange(candidate_k, dtype=np.int32)
        groups.append(candidate_k)
        cursor = end
        if order % 2000 == 0:
            print(f"[test] {order}/{len(test)} users", flush=True)

    frame = pd.DataFrame(features, columns=context["feat_cols"])
    frame["uid"] = output_uids
    frame["iid"] = output_iids
    frame["recall_rank"] = recall_ranks
    print(
        f"[test] rows={len(frame)} collab={stats['selected_collab']} "
        f"elapsed={time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return frame, list(context["feat_cols"]), list(context["cat_cols"]), groups


def evaluate_recall(
    datadir,
    n_folds=5,
    k=10,
    seed=42,
    collab_method="auto",
    ease_lambda=250.0,
    ease_max_items=1500,
    itemknn_k=200,
):
    """按OOF协议评价真实候选Recall；允许重复商品并使用冷启动用户特征。"""
    start = time.perf_counter()
    context = _load_context(datadir)
    folds = _make_folds(context["valid"], n_folds, seed)
    buckets: dict[str, list[float]] = {}
    all_hits: list[float] = []

    for fold_id, validation_indices in enumerate(folds):
        validation_set = {int(idx) for idx in validation_indices}
        train_indices = [idx for idx in context["valid"] if idx not in validation_set]
        stats = _build_fold_stats(
            context,
            train_indices,
            collab_method,
            ease_lambda,
            ease_max_items,
            itemknn_k,
        )
        rng = np.random.RandomState(seed * 1000 + fold_id)
        for record_idx in validation_indices:
            record_idx = int(record_idx)
            _, _, target = context["records"][record_idx]
            uid = context["uids"][record_idx]
            target_idx = context["iid2idx"][target]
            length = int(rng.choice(context["test_lengths"]))
            ded, counts = core.truncate_runs(context["runs"][record_idx], length)
            candidates = _generate_candidates(ded, counts, uid, context, stats, k)
            hit = float(target_idx in set(int(idx) for idx in candidates))
            bucket = f"L={length}" if length <= 3 else "L>=4"
            buckets.setdefault(bucket, []).append(hit)
            all_hits.append(hit)
        print(f"[recall] fold {fold_id + 1}/{n_folds} done", flush=True)

    overall = float(np.mean(all_hits)) if all_hits else 0.0
    bucket_result = {key: float(np.mean(values)) for key, values in sorted(buckets.items())}
    details = "  ".join(
        f"{key}:{value:.4f}(n{len(buckets[key])})" for key, value in bucket_result.items()
    )
    print(
        f"[recall] Recall@{k}={overall:.5f} | {details} | {time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return overall, bucket_result


def _write_frame(frame, output_path):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception as exc:
        fallback = os.path.splitext(output_path)[0] + ".csv"
        frame.to_csv(fallback, index=False)
        print(f"[提示] parquet失败({exc})，改存CSV: {fallback}")
        return fallback


def _write_metadata(data_path, feat_cols, cat_cols, groups, artifact_type):
    metadata_path = os.path.splitext(data_path)[0] + ".meta.json"
    metadata = {
        "artifact_type": artifact_type,
        "feature_columns": feat_cols,
        "categorical_columns": cat_cols,
        "groups": groups,
        "n_groups": len(groups),
        "n_rows": int(sum(groups)),
    }
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data/A推荐")
    parser.add_argument("--out", default="features/train_features_oof.parquet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collab", choices=["auto", "ease", "itemknn"], default="auto")
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--train-candidate-k", type=int, default=200,
                        help="训练group大小(1正+其余负)；与candidate-k同值即难度对齐")
    parser.add_argument("--hard-negative-ratio", type=float, default=0.75,
                        help="负样本中难负(召回top)占比，其余随机负")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--holdout", action="store_true", help="生成Outer holdout训练/验证特征")
    mode.add_argument("--test-features", action="store_true", help="生成正式测试候选特征")
    parser.add_argument("--outer-fold", type=int, default=0, help="holdout使用的outer fold，0开始")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--recall", action="store_true", help="额外运行OOF Recall@K")
    parser.add_argument("--recall-k", type=int, default=10)
    args = parser.parse_args()

    common = {
        "collab_method": args.collab,
    }
    if args.test_features:
        frame, feat_cols, cat_cols, groups = build_test_features(
            args.datadir,
            candidate_k=args.candidate_k,
            **common,
        )
        path = _write_frame(frame, args.out)
        meta = _write_metadata(path, feat_cols, cat_cols, groups, "test_features")
        print(f"[完成] test={path} metadata={meta}")
    elif args.holdout:
        train_frame, val_frame, feat_cols, cat_cols, train_groups, val_groups = build_holdout_features(
            args.datadir,
            outer_folds=args.folds,
            outer_fold=args.outer_fold,
            inner_folds=args.inner_folds,
            candidate_k=args.candidate_k,
            seed=args.seed,
            train_candidate_k=args.train_candidate_k,
            hard_negative_ratio=args.hard_negative_ratio,
            **common,
        )
        root, extension = os.path.splitext(args.out)
        extension = extension or ".parquet"
        train_path = _write_frame(train_frame, root + "_train" + extension)
        val_path = _write_frame(val_frame, root + "_val" + extension)
        _write_metadata(train_path, feat_cols, cat_cols, train_groups, "holdout_train")
        _write_metadata(val_path, feat_cols, cat_cols, val_groups, "holdout_validation")
        print(f"[完成] holdout train={train_path} val={val_path}")
    else:
        frame, feat_cols, cat_cols, groups = build_oof_features(
            args.datadir,
            n_folds=args.folds,
            seed=args.seed,
            train_candidate_k=args.train_candidate_k,
            hard_negative_ratio=args.hard_negative_ratio,
            **common,
        )
        path = _write_frame(frame, args.out)
        meta = _write_metadata(path, feat_cols, cat_cols, groups, "final_ranker_oof_train")
        print(f"[完成] OOF={path} metadata={meta}")

    if args.recall:
        evaluate_recall(
            args.datadir,
            n_folds=args.folds,
            k=args.recall_k,
            seed=args.seed,
            **common,
        )


if __name__ == "__main__":
    main()
