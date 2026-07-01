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

        # FM 一阶：数值特征走线性层；每个类别值拥有独立的标量权重。
        # 这里必须使用 Embedding 按类别索引查表，不能用需要 one-hot 输入的 Linear，
        # 也不能复用二阶 embedding，否则类别主效应会与交互表示相互牵制。
        self.num_first = nn.Linear(num_num_feats, 1, bias=False)
        self.cat_first = nn.ModuleDict({
            c: nn.Embedding(vs + 1, 1) for c, vs in cat_vocab_sizes.items()
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
            # 一维 embedding 直接给出该类别的一阶标量贡献。
            first = first + self.cat_first[c](cat_x[c])
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

    def _resolve_device(self, spec: str) -> torch.device:
        """auto/cpu/cuda -> torch.device。auto 时有 cuda 才用 cuda。"""
        spec = spec or "auto"
        if spec == "cpu":
            return torch.device("cpu")
        if spec == "cuda":
            if not torch.cuda.is_available():
                print("[deepfm] 指定 cuda 但不可用，回退 cpu")
                return torch.device("cpu")
            return torch.device("cuda")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

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
        self._device = self._resolve_device(p.get("device", "auto"))
        print(f"[deepfm] device={self._device}", flush=True)
        self.net = _DeepFMNet(
            num_num_feats=num_num, cat_vocab_sizes=self._vocab,
            embed_dim=p.get("embedding_dim", 8),
            hidden_dims=p.get("hidden_dims", [256, 128]),
            dropout=p.get("dropout", 0.2),
        ).to(self._device)
        opt = torch.optim.Adam(self.net.parameters(), lr=p.get("lr", 0.001))
        loss_fn = nn.BCEWithLogitsLoss()

        # 准备训练张量
        num_x, cat_x, _, _ = self._split_feats(tr_df, feat_cols, self._cat_cols)
        y = tr_df["label"].to_numpy(np.float32)
        num_t = torch.from_numpy(num_x).to(self._device)
        cat_t = {c: torch.from_numpy(cat_x[c]).to(self._device) for c in cat_x}
        y_t = torch.from_numpy(y).to(self._device)
        n = len(tr_df)
        bs = p.get("batch_size", 4096)
        epochs = p.get("epochs", 20)
        loss_kind = p.get("loss", "pointwise")

        # listwise 需按整 group 采样（tr_groups[i] = 第 i 个 group 的行数）。
        # pointwise 沿用整行打乱（向后兼容，且不依赖 group）。
        if loss_kind == "listwise":
            groups = [int(g) for g in tr_groups]
            group_offsets = torch.tensor(
                [0] + [int(np.cumsum(groups)[k]) for k in range(len(groups))],
                dtype=torch.long, device=self._device,
            )
            n_groups = len(groups)
            print(f"[deepfm] loss=listwise (ListNet, 按 group={groups[0]} 采样)", flush=True)
        else:
            print("[deepfm] loss=pointwise (BCE, 整行打乱)", flush=True)

        # watch 集评估用（va_df 是从 train 切的独立 watch，不是最终 val）
        self.net.train()
        best_ndcg, best_state, no_improve = -1.0, None, 0
        es = p.get("early_stopping", 3)
        for ep in range(epochs):
            total_loss = 0.0
            if loss_kind == "listwise":
                gperm = torch.randperm(n_groups, device=self._device)
                # 每 step 取若干整 group；group_size 固定（candidate_k/trian_candidate_k）
                # bs 取每 step 处理的候选行数，反推 group 数。
                grp_per_step = max(1, bs // int(groups[0]))
                for i in range(0, n_groups, grp_per_step):
                    gi = gperm[i:i + grp_per_step]
                    starts = group_offsets[gi]
                    ends = group_offsets[gi + 1]
                    # 整 group 的行索引（group 间连续，故可切片）
                    s, e = int(starts.min()), int(ends.max())
                    bx = num_t[s:e]
                    bc = {c: cat_t[c][s:e] for c in cat_t}
                    by = y_t[s:e]
                    opt.zero_grad()
                    logit = self.net(bx, bc)            # [e-s]
                    # 按 group 拆 logit 做 ListNet：每 group softmax+CE 对正样本。
                    loss = self._listnet_loss(logit, by, starts, ends)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * (e - s)
            else:
                perm = torch.randperm(n, device=self._device)
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
        # no_grad 只关闭梯度，不会关闭 Dropout。watch early-stop 发生在训练循环中，
        # 因此预测前必须临时切到 eval，否则每次 watch NDCG 都带随机噪声；预测后
        # 恢复原训练状态，确保下一 epoch 的 Dropout 继续生效。
        was_training = self.net.training
        self.net.eval()
        try:
            num_x, cat_x, _, _ = self._split_feats(df, self._feat_cols, self._cat_cols)
            with torch.no_grad():
                num_t = torch.from_numpy(num_x).to(self._device)
                cat_t = {c: torch.from_numpy(cat_x[c]).to(self._device) for c in cat_x}
                return self.net(num_t, cat_t).cpu().numpy().astype(np.float32)
        finally:
            if was_training:
                self.net.train()

    def _listnet_loss(self, logit, label, starts, ends):
        """ListNet：每个 group 对 logits 做 softmax，与标签的 softmax 做 CE。

        标签含 1 个正样本(label=1)，故标签分布是 one-hot，CE 退化为
        -log_softmax(logit)[正样本位置] 的负对数似然。按 group 求和后除以 group 数。
        ListNet 是 listwise 排序损失，直接优化 group 内正样本排名，与 NDCG 对齐
        （xgb rank:ndcg 同理），弥补 pointwise BCE 不感知相对顺序的结构缺陷。
        """
        device = logit.device
        total = logit.new_zeros(())
        for k in range(starts.shape[0]):
            s, e = int(starts[k]), int(ends[k])
            g_logit = logit[s:e]                       # [g]
            g_label = label[s:e]                       # [g]
            probs = torch.softmax(g_logit, dim=0)
            mask = g_label > 0.5
            if not bool(mask.any()):
                continue
            pos_prob = probs[mask].sum()               # 正样本分配到的概率
            total = total - torch.log(pos_prob + 1e-12)
        return total / starts.shape[0]

    def predict_scores(self, df, feat_cols):
        return self._predict_internal(df)

    def feature_importance(self):
        # DeepFM 无直接 gain；返回 {}（run.py 的 _print_importance 会跳过）
        return {}
