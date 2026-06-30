"""DeepFM 排序器（非序列深度学习推荐模型）。

FM（低阶二阶交叉）+ DNN（高阶），pointwise BCE。塞进 RankerModel 候选表接口，
与 XGB/LGB 同范式，走相同 OOF/holdout 无泄漏协议。

FM 部分对 cat_cols 学 embedding 并自动学二阶交叉（u_cat×i_cat）——补当前手算
特征体系缺的"用户类别×候选类别交互"。DNN 部分对 cat embedding + 数值特征做
高阶非线性。

关键坑：cat 值可能为 -1（冷启动用户的未知类别码，prepare_user_features 返回
-1），而 Embedding 索引必须非负。fit 时把 -1 映射到 vocab_size 槽位
（总槽位 vocab_size+1），predict 同样映射。

训练：pointwise BCE（label 0/1），按 batch 喂全候选行（不做 listwise——这是
DeepFM 标准与 XGB rank:ndcg 的本质差异，预期可能略弱，诚实实现）。
early-stop：watch 集上算 ndcg@10（调 eval_core），与"val 不参与选模"一致。
"""
from __future__ import annotations
import os

# torch 在 conda 环境常与 numpy 的 OpenMP 冲突，必须先放开否则 import 即 crash。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.base import RankerModel
from models.registry import register
from models._shared import eval_core


class _DeepFMNet(nn.Module):
    """FM + DNN 网络。"""

    def __init__(self, num_num_feats: int, cat_vocab_sizes: dict[str, int],
                 embed_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        # 每个 cat 列一个 embedding：vocab_size+1 槽（末槽给 -1/未知）。
        self.cat_cols = list(cat_vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(vs + 1, embed_dim)
            for c, vs in cat_vocab_sizes.items()
        })
        n_cat_embed = len(self.cat_cols) * embed_dim

        # FM 一阶：数值线性 + 每 cat embedding 一阶（求和到标量）
        self.num_first = nn.Linear(num_num_feats, 1, bias=False)
        self.cat_first = nn.ModuleDict({
            c: nn.Linear(vs + 1, 1, bias=False) for c, vs in cat_vocab_sizes.items()
        })

        # DNN：cat embedding flatten + 数值 → MLP
        in_dim = n_cat_embed + num_num_feats
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.dnn = nn.Sequential(*layers) if layers else nn.Identity()
        self.dnn_out = nn.Linear(in_dim, 1, bias=False)

    def forward(self, num_x: torch.Tensor, cat_x: dict[str, torch.Tensor]) -> torch.Tensor:
        # cat_x: {col: LongTensor[...,]}（已把 -1 映射到 vocab_size）
        embeds = [self.embeddings[c](cat_x[c]) for c in self.cat_cols]  # 各 [B, D]
        # FM 一阶
        first = self.num_first(num_x)  # [B,1]
        for c in self.cat_cols:
            # embedding 一阶：对每个样本取其 cat 对应 embedding 的和
            first = first + self.embeddings[c](cat_x[c]).sum(dim=1, keepdim=True)
        # FM 二阶：(sum e)^2 - sum(e^2)，内积形式
        if embeds:
            sum_emb = torch.stack(embeds, dim=1).sum(dim=1)   # [B, D]
            sum_sq = torch.stack([e * e for e in embeds], dim=1).sum(dim=1)
            second = 0.5 * (sum_emb.pow(2).sum(dim=1, keepdim=True)
                            - sum_sq.sum(dim=1, keepdim=True))  # [B,1]
        else:
            second = torch.zeros(num_x.shape[0], 1, device=num_x.device)

        # DNN
        dnn_in = torch.cat([num_x] + embeds, dim=1) if embeds else num_x
        dnn_logit = self.dnn_out(self.dnn(dnn_in))  # [B,1]
        return (first + second + dnn_logit).squeeze(1)  # [B]


@register
class DeepFM(RankerModel):
    name = "deepfm"

    def _split_feats(self, df: pd.DataFrame, feat_cols: list[str],
                     cat_cols: list[str]):
        """拆数值/类别，cat 值转 LongTensor 并把 -1 映射到各自 vocab_size 槽。"""
        cats = [c for c in cat_cols if c in feat_cols]
        nums = [c for c in feat_cols if c not in cats]
        num_x = df[nums].to_numpy(np.float32) if nums else np.zeros((len(df), 0), np.float32)
        cat_x = {}
        for c in cats:
            v = df[c].to_numpy(np.int64)
            vs = self._vocab[c]
            v = np.where(v < 0, vs, v)   # -1 -> vocab_size 槽
            v = np.clip(v, 0, vs)         # 防越界
            cat_x[c] = v
        return num_x, cat_x, nums, cats

    def fit(self, tr_df, va_df, feat_cols, cat_cols, tr_groups, va_groups):
        p = self.params
        torch.manual_seed(p.get("seed", 42))
        np.random.seed(p.get("seed", 42))

        self._cat_cols = [c for c in cat_cols if c in feat_cols]
        self._feat_cols = list(feat_cols)
        # 算各 cat 列基数（max+1，-1 不计）。fit 内从训练数据算，不改接口。
        self._vocab = {}
        for c in self._cat_cols:
            vals = tr_df[c].to_numpy()
            vals = vals[vals >= 0]
            self._vocab[c] = int(vals.max()) + 1 if len(vals) else 1

        num_num = len([c for c in feat_cols if c not in self._cat_cols])
        self.net = _DeepFMNet(
            num_num_feats=num_num, cat_vocab_sizes=self._vocab,
            embed_dim=p.get("embedding_dim", 8),
            hidden_dims=p.get("hidden_dims", [256, 128]),
            dropout=p.get("dropout", 0.2),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=p.get("lr", 0.001))
        loss_fn = nn.BCEWithLogitsLoss()

        # 准备训练张量
        num_x, cat_x, _, _ = self._split_feats(tr_df, feat_cols, self._cat_cols)
        y = tr_df["label"].to_numpy(np.float32)
        num_t = torch.from_numpy(num_x)
        cat_t = {c: torch.from_numpy(cat_x[c]) for c in cat_x}
        y_t = torch.from_numpy(y)
        n = len(tr_df)
        bs = p.get("batch_size", 4096)
        epochs = p.get("epochs", 20)

        # watch 集评估用（va_df 是从 train 切的独立 watch，不是最终 val）
        self.net.train()
        best_ndcg, best_state, no_improve = -1.0, None, 0
        es = p.get("early_stopping", 3)
        for ep in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                bx = num_t[idx]
                bc = {c: cat_t[c][idx] for c in cat_t}
                by = y_t[idx]
                opt.zero_grad()
                logit = self.net(bx, bc)
                loss = loss_fn(logit, by)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            # watch ndcg@10 early-stop
            msg = f"[deepfm] epoch {ep+1}/{epochs} loss={total_loss/n:.4f}"
            if va_df is not None:
                scores = self._predict_internal(va_df)
                overall, _, _, _ = eval_core.eval_ndcg(scores, va_df, 10)
                msg += f" watch_ndcg@10={overall:.5f}"
                if overall > best_ndcg:
                    best_ndcg = overall
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
            print(msg, flush=True)
            if va_df is not None and es and no_improve >= es:
                print(f"[deepfm] early-stop @ epoch {ep+1} (best watch_ndcg@10={best_ndcg:.5f})")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def _predict_internal(self, df: pd.DataFrame) -> np.ndarray:
        num_x, cat_x, _, _ = self._split_feats(df, self._feat_cols, self._cat_cols)
        with torch.no_grad():
            num_t = torch.from_numpy(num_x)
            cat_t = {c: torch.from_numpy(cat_x[c]) for c in cat_x}
            return self.net(num_t, cat_t).cpu().numpy().astype(np.float32)

    def predict_scores(self, df, feat_cols):
        return self._predict_internal(df)

    def feature_importance(self):
        # DeepFM 无直接 gain；返回 {}（run.py 的 _print_importance 会跳过）
        return {}
