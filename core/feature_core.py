"""推荐任务共享特征核心。

为 build_features.py 提供序列解析、统计矩阵、协同模型和候选特征等底层能力：

1. 训练序列只解析一次，并预先保存 run-length 编码；
2. 每个用户只计算采样候选的 Repeat/Markov/HT 特征；
3. EASE 利用用户向量的稀疏性，按历史商品行求和，不再执行稠密向量乘矩阵；
4. 用户、商品特征预编码为数组，训练结果一次性预分配；
5. 商品规模较大时，auto 模式自动改用稀疏 ItemKNN，避免稠密 EASE 求逆。

通常应运行 build_features.py。直接运行本文件仅用于底层特征性能测试；其全量
训练特征入口不提供OOF隔离，不能用于报告离线验证分数。
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp


def _split(s: str) -> list[str]:
    """把逗号分隔序列解析为列表；空值返回空列表。"""
    return [x for x in str(s).split(",") if x] if s else []


def _parse_counts(s: str) -> dict[str, int]:
    """解析形如 ``item:count,item:count`` 的计数字符串。"""
    out: dict[str, int] = {}
    for tok in _split(s):
        if ":" not in tok:
            continue
        key, value = tok.rsplit(":", 1)
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = 1
    return out


def _to_runs(raw: Iterable[str]) -> list[tuple[str, int]]:
    """将原始序列压缩为 ``[(item, 连续次数), ...]``。"""
    runs: list[tuple[str, int]] = []
    for iid in raw:
        if runs and runs[-1][0] == iid:
            runs[-1] = (iid, runs[-1][1] + 1)
        else:
            runs.append((iid, 1))
    return runs


def truncate_runs(runs: list[tuple[str, int]], length: int):
    """保留最近 ``length`` 个 run，并重建 dedup 序列与累计次数。"""
    if length <= 0 or not runs:
        return [], {}
    tail = runs[-length:]
    ded = [iid for iid, _ in tail]
    counts: dict[str, int] = {}
    for iid, count in tail:
        counts[iid] = counts.get(iid, 0) + count
    return ded, counts


def load_data(datadir: str):
    train = pd.read_csv(os.path.join(datadir, "train.csv"), dtype=str).fillna("")
    test = pd.read_csv(os.path.join(datadir, "test.csv"), dtype=str).fillna("")
    item = pd.read_csv(os.path.join(datadir, "item.csv"), dtype=str).fillna("")
    user = pd.read_csv(os.path.join(datadir, "user.csv"), dtype=str).fillna("")
    items = item["iid"].tolist()
    iid2idx = {iid: idx for idx, iid in enumerate(items)}
    return {
        "train": train,
        "test": test,
        "item": item,
        "user": user,
        "items": items,
        "iid2idx": iid2idx,
        "n_item": len(items),
    }


def build_popularity(records, n_item, iid2idx):
    pop = np.zeros(n_item, dtype=np.float32)
    for _, counts, _ in records:
        for iid, count in counts.items():
            idx = iid2idx.get(iid)
            if idx is not None:
                pop[idx] += count
    maximum = float(pop.max())
    if maximum > 0:
        pop /= maximum
    return pop


def build_target_prior(records, n_item, iid2idx):
    is_target = np.zeros(n_item, dtype=np.float32)
    for _, _, target in records:
        idx = iid2idx.get(target) if target else None
        if idx is not None:
            is_target[idx] = 1.0
    return is_target


def build_target_freq(records, n_item, iid2idx):
    """每个商品作为训练 target 出现的次数（target 先验频率）。

    与 build_popularity（历史序列曝光度）不同：这里统计商品"真正成为 target"的
    频次。A 榜仅 ~235 个 item 曾作训练 target，故该信号近二值且强——可救 44% 非
    repeat target 与 35% 冷启动用户（这两类 repeat/markov/htarget 全失效）。

    必须按 OOF 折统计（只含训练折的 target），验证折 target 不得进自身特征；
    test 用全量训练统计（test target 隐藏，无泄漏）。这些隔离由 _build_fold_stats
    的按折构建天然保证。
    """
    freq = np.zeros(n_item, dtype=np.int32)
    for _, _, target in records:
        idx = iid2idx.get(target) if target else None
        if idx is not None:
            freq[idx] += 1
    return freq


def build_user_item_matrix(records, n_item, iid2idx):
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for user_idx, (_, counts, _) in enumerate(records):
        for iid, count in counts.items():
            item_idx = iid2idx.get(iid)
            if item_idx is None:
                continue
            rows.append(user_idx)
            cols.append(item_idx)
            values.append(math.log1p(count))
    return sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(len(records), n_item),
        dtype=np.float32,
    )


def build_ease(X: sp.csr_matrix, lam: float):
    """构建稠密 EASE 矩阵；仅适合商品规模较小的情况。"""
    gram = (X.T @ X).toarray().astype(np.float64, copy=False)
    gram.flat[:: gram.shape[0] + 1] += lam
    precision = sla.inv(gram, overwrite_a=True, check_finite=False)
    weights = -precision / np.diag(precision)[None, :]
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def _prune_topk_rows(matrix: sp.csr_matrix, topk: int):
    """每行仅保留绝对值最大的 top-k 个稀疏元素。"""
    if topk <= 0:
        return matrix
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_cols = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_values.size > topk:
            keep = np.argpartition(np.abs(row_values), -topk)[-topk:]
            row_cols = row_cols[keep]
            row_values = row_values[keep]
        rows.extend([row_idx] * len(row_cols))
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())
    return sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=matrix.shape,
        dtype=np.float32,
    )


def build_itemknn(X: sp.csr_matrix, topk: int):
    """构建稀疏 item-item cosine 相似度，作为大商品池的协同信号。"""
    gram = (X.T @ X).tocsr().astype(np.float32)
    norm = np.sqrt(np.maximum(gram.diagonal(), 1e-12)).astype(np.float32)
    inv_norm = 1.0 / norm
    similarity = sp.diags(inv_norm) @ gram @ sp.diags(inv_norm)
    similarity = similarity.tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    return _prune_topk_rows(similarity, topk)


def build_collab(X, method, ease_lambda, ease_max_items, itemknn_k):
    """按商品规模选择 EASE 或稀疏 ItemKNN。"""
    selected = method
    if method == "auto":
        selected = "ease" if X.shape[1] <= ease_max_items else "itemknn"
    if selected == "ease":
        return build_ease(X, ease_lambda), "ease"
    if selected == "itemknn":
        return build_itemknn(X, itemknn_k), "itemknn"
    raise ValueError(f"未知协同方法: {method}")


def _normalize_sparse_rows(matrix: sp.csr_matrix):
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return (sp.diags(1.0 / row_sum) @ matrix).tocsr()


def build_markov(records, n_item, iid2idx):
    """使用 COO 批量构建转移矩阵，避免逐元素修改 LIL。"""
    rows: list[int] = []
    cols: list[int] = []
    for ded, _, target in records:
        idxs = [iid2idx[iid] for iid in ded if iid in iid2idx]
        rows.extend(idxs[:-1])
        cols.extend(idxs[1:])
        target_idx = iid2idx.get(target) if target else None
        if idxs and target_idx is not None:
            rows.append(idxs[-1])
            cols.append(target_idx)
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n_item, n_item)).tocsr()
    return _normalize_sparse_rows(matrix)


def build_htarget(records, n_item, iid2idx):
    rows: list[int] = []
    cols: list[int] = []
    for ded, _, target in records:
        target_idx = iid2idx.get(target) if target else None
        if target_idx is None:
            continue
        history = {iid2idx[iid] for iid in ded if iid in iid2idx}
        rows.extend(history)
        cols.extend([target_idx] * len(history))
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n_item, n_item)).tocsr()
    return _normalize_sparse_rows(matrix)


def prepare_user_features(user_df: pd.DataFrame):
    ucols = [column for column in user_df.columns if column != "uid"]
    factor_maps = {
        column: {value: idx for idx, value in enumerate(sorted(user_df[column].unique()))}
        for column in ucols
    }
    raw_lookup: dict[str, tuple[str, ...]] = {}
    encoded_lookup: dict[str, np.ndarray] = {}
    columns = ["uid"] + ucols
    for row in user_df[columns].itertuples(index=False, name=None):
        uid, values = row[0], tuple(row[1:])
        raw_lookup[uid] = values
        encoded_lookup[uid] = np.asarray(
            [factor_maps[column].get(value, -1) for column, value in zip(ucols, values)],
            dtype=np.float32,
        )
    return ucols, raw_lookup, encoded_lookup


def build_userfeat_target_cond(
    records,
    uids,
    user_raw,
    ucols,
    n_item,
    iid2idx,
):
    """统计 ``(用户特征列, 值) -> target分布``。"""
    distributions: dict[tuple[str, str], np.ndarray] = {}
    for (_, _, target), uid in zip(records, uids):
        target_idx = iid2idx.get(target) if target else None
        values = user_raw.get(uid)
        if target_idx is None or values is None:
            continue
        for column, value in zip(ucols, values):
            key = (column, value)
            if key not in distributions:
                distributions[key] = np.zeros(n_item, dtype=np.float32)
            distributions[key][target_idx] += 1.0
    for values in distributions.values():
        total = float(values.sum())
        if total > 0:
            values /= total
    return distributions


def _accumulate_sparse_rows(matrix, weighted_rows):
    """累加稀疏矩阵的若干行，返回 ``列->分数`` 字典及最大分数。"""
    scores: dict[int, float] = {}
    for row_idx, weight in weighted_rows:
        row = matrix.getrow(row_idx)
        for col_idx, value in zip(row.indices, row.data):
            scores[int(col_idx)] = scores.get(int(col_idx), 0.0) + weight * float(value)
    maximum = max(scores.values(), default=0.0)
    return scores, maximum


def candidate_components(ded, counts, candidates, iid2idx, items, collab, T, HT, decay=0.95):
    """只为当前候选计算协同特征，避免每用户创建多个全商品向量。"""
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_iids = [items[idx] for idx in candidates]
    idxs = [iid2idx[iid] for iid in ded if iid in iid2idx]
    size = len(candidates)
    zeros = np.zeros(size, dtype=np.float32)
    if not idxs:
        return zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

    # Repeat：按最近位置和累计次数计算，候选查询使用字典。
    repeat_by_item: dict[int, float] = {}
    length = len(idxs)
    for position, item_idx in enumerate(idxs):
        score = decay ** (length - 1 - position) * math.sqrt(counts.get(items[item_idx], 1))
        repeat_by_item[item_idx] = max(repeat_by_item.get(item_idx, 0.0), score)
    repeat_max = max(repeat_by_item.values(), default=0.0)
    repeat = np.asarray(
        [repeat_by_item.get(int(idx), 0.0) / repeat_max if repeat_max else 0.0 for idx in candidates],
        dtype=np.float32,
    )

    # 协同分：EASE仅累加历史对应的矩阵行，复杂度由 O(n_item^2) 降为 O(L*n_item)。
    unique_idxs = list(dict.fromkeys(idxs))
    if sp.issparse(collab):
        weighted = [(idx, math.log1p(counts.get(items[idx], 1))) for idx in unique_idxs]
        collab_scores, collab_max = _accumulate_sparse_rows(collab, weighted)
        collaborative = np.asarray(
            [collab_scores.get(int(idx), 0.0) / collab_max if collab_max else 0.0 for idx in candidates],
            dtype=np.float32,
        )
    else:
        full_scores = np.zeros(collab.shape[1], dtype=np.float32)
        for idx in unique_idxs:
            full_scores += math.log1p(counts.get(items[idx], 1)) * collab[idx]
        minimum, maximum = float(full_scores.min()), float(full_scores.max())
        if maximum > minimum:
            full_scores = (full_scores - minimum) / (maximum - minimum)
        collaborative = full_scores[candidates].astype(np.float32, copy=False)

    # Markov：最近三个历史 item 的转移分布。
    markov_rows = [(idx, 0.7**rank) for rank, idx in enumerate(reversed(idxs[-3:]))]
    markov_scores, markov_max = _accumulate_sparse_rows(T, markov_rows)
    markov = np.asarray(
        [markov_scores.get(int(idx), 0.0) / markov_max if markov_max else 0.0 for idx in candidates],
        dtype=np.float32,
    )

    # 历史->target：保留同一 item 多次出现在 dedup 序列中的位置权重。
    htarget_rows = []
    for position, idx in enumerate(idxs):
        weight = decay ** (length - 1 - position) * math.sqrt(counts.get(items[idx], 1))
        htarget_rows.append((idx, weight))
    htarget_scores, htarget_max = _accumulate_sparse_rows(HT, htarget_rows)
    htarget = np.asarray(
        [htarget_scores.get(int(idx), 0.0) / htarget_max if htarget_max else 0.0 for idx in candidates],
        dtype=np.float32,
    )

    # 原实现名为 lastcat，实际是最后一个具体 item 的 HT 行。
    last_row = HT.getrow(idxs[-1])
    last_map = {int(idx): float(value) for idx, value in zip(last_row.indices, last_row.data)}
    last_max = max(last_map.values(), default=0.0)
    last_cond = np.asarray(
        [last_map.get(int(idx), 0.0) / last_max if last_max else 0.0 for idx in candidates],
        dtype=np.float32,
    )

    return repeat, collaborative, markov, htarget, last_cond


def build_one_user_features(
    candidates,
    ded,
    counts,
    pop,
    is_target,
    user_raw_values,
    user_encoded_values,
    ucols,
    item_features,
    ufp,
    components,
    items,
):
    candidates = np.asarray(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    candidate_iids = [items[idx] for idx in candidates]
    maximum_count = max(counts.values(), default=1)

    in_history = np.asarray([iid in counts for iid in candidate_iids], dtype=np.float32)
    count = np.asarray([counts.get(iid, 0) for iid in candidate_iids], dtype=np.float32)
    count_norm = count / maximum_count
    repeat, collaborative, markov, htarget, last_cond = components

    if user_encoded_values is None:
        user_features = np.full((n_candidates, len(ucols)), -1.0, dtype=np.float32)
    else:
        user_features = np.broadcast_to(user_encoded_values, (n_candidates, len(ucols)))

    conditional = np.zeros(n_candidates, dtype=np.float32)
    if user_raw_values is not None:
        for column, value in zip(ucols, user_raw_values):
            distribution = ufp.get((column, value))
            if distribution is not None:
                conditional += distribution[candidates]

    columns = [
        in_history[:, None],
        count[:, None],
        count_norm[:, None],
        pop[candidates, None],
        is_target[candidates, None],
        repeat[:, None],
        collaborative[:, None],
        markov[:, None],
        htarget[:, None],
        user_features,
        item_features[candidates],
        conditional[:, None],
        last_cond[:, None],
    ]
    return np.hstack(columns).astype(np.float32, copy=False)


def build_training_features(
    datadir: str,
    neg: int = 20,
    seed: int = 42,
    collab_method: str = "auto",
    ease_lambda: float = 250.0,
    ease_max_items: int = 1500,
    itemknn_k: int = 200,
):
    start_time = time.perf_counter()
    data = load_data(datadir)
    train, test = data["train"], data["test"]
    user, item = data["user"], data["item"]
    items, iid2idx, n_item = data["items"], data["iid2idx"], data["n_item"]
    item_ids = np.asarray(items, dtype=object)

    # 一次解析所有训练字段，并预先构建 run-length 序列。
    records, uids, runs = [], [], []
    columns = ["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"]
    for uid, target, raw_text, ded_text, count_text in train[columns].itertuples(index=False, name=None):
        records.append((_split(ded_text), _parse_counts(count_text), target or None))
        uids.append(uid)
        runs.append(_to_runs(_split(raw_text)))

    pop = build_popularity(records, n_item, iid2idx)
    is_target = build_target_prior(records, n_item, iid2idx)
    interaction = build_user_item_matrix(records, n_item, iid2idx)
    collab, selected_collab = build_collab(
        interaction,
        collab_method,
        ease_lambda,
        ease_max_items,
        itemknn_k,
    )
    markov = build_markov(records, n_item, iid2idx)
    htarget = build_htarget(records, n_item, iid2idx)

    ucols, user_raw, user_encoded = prepare_user_features(user)
    ufp = build_userfeat_target_cond(records, uids, user_raw, ucols, n_item, iid2idx)

    icols = [column for column in item.columns if column != "iid"]
    item_features = (
        item[icols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        if icols
        else np.zeros((n_item, 0), dtype=np.float32)
    )

    target_pool = np.asarray(
        sorted({iid2idx[target] for _, _, target in records if target in iid2idx}),
        dtype=np.int64,
    )
    if len(target_pool) < 2:
        raise ValueError("训练 target 商品不足两个，无法进行负采样")
    negative_count = min(neg, len(target_pool) - 1)
    negative_pools = {
        int(target): target_pool[target_pool != target]
        for target in target_pool
    }
    valid_indices = [
        idx for idx, (_, _, target) in enumerate(records)
        if target is not None and target in iid2idx
    ]

    test_lengths = np.asarray([len(_split(value)) for value in test["item_seq_dedup"]], dtype=np.int32)
    rng = np.random.RandomState(seed)

    feature_columns = (
        ["in_hist", "count", "count_norm", "popularity", "is_known_target",
         "repeat", "ease", "markov", "htarget"]
        + [f"u_{column}" for column in ucols]
        + [f"i_{column}" for column in icols]
        + ["ufeat_target_cond", "lastcat_target_cond"]
    )
    group_size = 1 + negative_count
    total_rows = len(valid_indices) * group_size
    feature_matrix = np.empty((total_rows, len(feature_columns)), dtype=np.float32)
    labels = np.zeros(total_rows, dtype=np.float32)
    output_uids = np.empty(total_rows, dtype=object)
    output_iids = np.empty(total_rows, dtype=object)

    cursor = 0
    for order, record_idx in enumerate(valid_indices, start=1):
        _, _, target = records[record_idx]
        uid = uids[record_idx]
        target_idx = iid2idx[target]
        negatives = rng.choice(negative_pools[target_idx], size=negative_count, replace=False)
        candidates = np.concatenate(([target_idx], negatives)).astype(np.int64, copy=False)

        history_length = int(rng.choice(test_lengths))
        ded, counts = truncate_runs(runs[record_idx], history_length)
        components = candidate_components(
            ded,
            counts,
            candidates,
            iid2idx,
            items,
            collab,
            markov,
            htarget,
        )
        features = build_one_user_features(
            candidates,
            ded,
            counts,
            pop,
            is_target,
            user_raw.get(uid),
            user_encoded.get(uid),
            ucols,
            item_features,
            ufp,
            components,
            items,
        )

        end = cursor + group_size
        feature_matrix[cursor:end] = features
        labels[cursor] = 1.0
        output_uids[cursor:end] = uid
        output_iids[cursor:end] = item_ids[candidates]
        cursor = end

        if order % 5000 == 0 or order == len(valid_indices):
            elapsed = time.perf_counter() - start_time
            print(f"[特征构建] {order}/{len(valid_indices)} 用户，累计 {elapsed:.1f}s")

    feature_df = pd.DataFrame(feature_matrix, columns=feature_columns)
    feature_df["uid"] = output_uids
    feature_df["iid"] = output_iids
    feature_df["label"] = labels
    print(
        f"[特征构建] 协同方法={selected_collab}, rows={len(feature_df)}, "
        f"features={len(feature_columns)}, total={time.perf_counter() - start_time:.1f}s"
    )
    return feature_df, feature_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data/A推荐")
    parser.add_argument("--neg", type=int, default=20, help="每用户负样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="train_features_core_benchmark.parquet")
    parser.add_argument("--collab", choices=["auto", "ease", "itemknn"], default="auto")
    parser.add_argument("--ease-lambda", type=float, default=250.0)
    parser.add_argument(
        "--ease-max-items",
        type=int,
        default=1500,
        help="auto模式下允许使用稠密EASE的最大商品数；A/B榜默认走稀疏ItemKNN",
    )
    parser.add_argument("--itemknn-k", type=int, default=200)
    args = parser.parse_args()

    frame, columns = build_training_features(
        args.datadir,
        neg=args.neg,
        seed=args.seed,
        collab_method=args.collab,
        ease_lambda=args.ease_lambda,
        ease_max_items=args.ease_max_items,
        itemknn_k=args.itemknn_k,
    )
    print(f"[完成] 特征表形状: {frame.shape}")
    print(f"[完成] 特征列({len(columns)}个): {columns}")
    try:
        frame.to_parquet(args.out, index=False)
        print(f"[完成] 已保存: {args.out}")
    except Exception as exc:
        fallback = os.path.splitext(args.out)[0] + ".csv"
        frame.to_csv(fallback, index=False)
        print(f"[提示] parquet保存失败({exc})，已保存CSV: {fallback}")


if __name__ == "__main__":
    main()
