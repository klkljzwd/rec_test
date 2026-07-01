# 推荐任务相关论文、Kaggle 解题思路与可尝试方案

> 调研日期：2026-07-01  
> 任务：根据用户历史行为预测下一个商品，评价指标为 NDCG@10。  
> 当前数据特征：40,000 个训练用户、10,000 个测试用户、2,156 个商品；约 56.38% 的 target 出现在用户历史中，约 35.15% 的测试用户为空历史，训练集中仅 235 个商品出现过 target。

## 1. 先说结论

与本任务最接近的公开比赛和论文给出的共同结论是：

1. **候选生成通常比继续加深排序模型更重要。** OTTO 和 H&M 的高分方案普遍采用多路候选生成，再用 LightGBM/XGBoost/CatBoost 排序；H&M 第一名复盘明确强调候选策略是关键。
2. **item-to-item 信号要做成多个视角，而不是只有一个 ItemKNN 分数。** 常见做法包括有向共现、不同距离/位置衰减、最近商品转移、Word2Vec、矩阵分解和图 embedding，并把它们对用户历史的 max/sum/mean/加权和作为排序特征。
3. **重复购买与探索新商品最好显式拆开。** RepeatNet 的核心就是先判断 repeat 还是 explore，再分别计算候选概率。本任务 56% target 在历史中，非常适合这个结构。
4. **简单方法在小数据上往往不输 Transformer。** session recommendation 的系统评测以及后续的 Transformer/近邻比较都指出，近邻方法在部分数据集上可以达到或超过复杂深度模型；本任务数据量不大、测试又有 35% 冷启动，不宜把 SASRec/BERT4Rec 当作第一优先级。
5. **当前最值得尝试的是 item embedding/SVD、丰富的共现矩阵、用户历史类别画像、repeat-explore gate，以及 XGB/LGB/DeepFM 融合。**

## 2. Kaggle 与比赛方案

### 2.1 OTTO Multi-Objective Recommender System

OTTO 是典型的 session-based recommendation：根据截断后的会话预测后续点击、加购和购买。它的商品规模远大于本任务，但“短序列、下一行为、多路召回、Top-K 排序”的结构高度相似。

#### 第一名方案

- 原始方案：[Kaggle 1st place discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)
- 中文汇总：[Kaggle OTTO 推荐系统比赛 TOP 方案总结](https://lukan217.github.io/2023/02/12/Kaggle%20Otto%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E6%AF%94%E8%B5%9BTOP%E6%96%B9%E6%A1%88%E6%80%BB%E7%BB%93/)

主要思路：

- 生成约 1,200 个候选，来源包括历史商品、多种共同访问矩阵和神经召回。
- 对共现矩阵分别施加行为类型、时间和位置权重，并进行多跳召回。
- 排序特征包括 session 长度、重复率、商品热度、共现排名、神经召回余弦相似度、候选在 session 中的位置等。
- 使用 LightGBM Ranker，并融合多个不同超参数的模型。

可迁移到本任务：

- 不需要照搬 1,200 个候选；商品只有 2,156 个，可以先构造 top100 候选池，再压缩到 top25/top10。
- 建立多套有向共现矩阵：最近一次转移、相邻 run、距离衰减、只统计 target 转移、类别条件转移。
- 每套矩阵不只输出一个总分，还输出最近商品相似度、历史 max、sum、mean、recency-weighted sum 和召回 rank。

#### 第二名方案

- [Kaggle 2nd place Part 1](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)
- [Kaggle 2nd place Part 2](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382790)

主要思路：

- 大量使用 next-action、有向 item-CF 和不同加权的共现矩阵。
- 使用 Word2Vec 与图 embedding 相似度。
- 对候选与最后一个商品、最近若干商品、全部历史商品的相似度做 max/sum/weighted-sum 聚合。
- CatBoost Ranker、LightGBM、XGBoost 等模型融合。

可迁移到本任务：

- 当前 `collab_score` 只有一个聚合结果，可以拆成多列。
- 增加 `sim_last`、`sim_last3_max`、`sim_hist_mean`、`sim_hist_decay_sum`、`candidate_recall_rank`。
- 用 item2vec/SVD/共现矩阵分别生成以上聚合特征，交给 XGB/LGB/DeepFM 排序。

#### 第三名与规则模型

- [Kaggle 3rd place discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)
- [Rules-only public notebook](https://www.kaggle.com/code/cdeotte/rules-only-model-achieves-lb-590)

主要思路：

- 构建多套规则不同的共同访问矩阵，仅靠规则召回就能得到很强的成绩。
- 排序阶段使用 session、item、session×item 和各共现矩阵分数，再交给 XGBoost。

对本任务的启示：当前已有 Markov、HTarget、ItemKNN，但矩阵视角仍然偏少。继续增加“不同统计口径的矩阵”可能比换更复杂的神经网络更有效。

### 2.2 H&M Personalized Fashion Recommendations

H&M 根据用户历史购买预测未来商品，使用 MAP@12，包含大量冷启动和商品元数据，和本任务的“历史复购 + 用户/商品类别 + 排序”非常接近。

- [H&M 第一名方案](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)
- [H&M 上位方案复盘幻灯片](https://speakerdeck.com/kuto5046/kaggle-h-and-mkonpezhen-rifan-ri)
- [H&M 排名融合示例](https://www.kaggle.com/code/titericz/h-m-ensembling-how-to)

上位方案中的高价值做法：

- 候选来源包含历史购买、近期热门、类别热门、Word2Vec、图 embedding（ProNE）、BPR 和双塔召回。
- 计算用户历史商品与候选商品的 Word2Vec/图 embedding 相似度。
- 使用“用户 × 商品”的动态特征：用户在候选类别上的购买比例、复购率、与上次商品的相似度等。
- 冷启动用户使用按年龄或用户属性分组的近期热门商品，而不是所有人共用一个热门榜。
- 多个候选/特征不同的模型采用 rank-weighted ensemble。

可迁移到本任务：

- 对 8 个用户类别分别统计 target 商品或 target 类别条件分布，当前 `ufeat_target_cond` 可以进一步拆为 8 个独立特征。
- 构建用户历史类别画像，例如各 item category 的占比、熵、主类别和候选类别匹配度。
- 冷启动用户按 `u_cat_01~08` 生成分群 target prior，而不只使用全局 `target_freq`。
- 用 reciprocal-rank 或归一化分数融合 XGB、LGB 和 DeepFM。

### 2.3 Instacart Market Basket Analysis

Instacart 的目标是预测用户下一单会再次购买哪些历史商品，是 repeat recommendation 的经典比赛。

- [公开的高分方案代码与说明](https://github.com/sjvasquez/instacart-basket-prediction)
- [获奖方案概览](https://queirozf.com/entries/winning-solutions-overview-kaggle-instacart-competition)
- [特征工程方案示例](https://github.com/p-mckenzie/Instacart-market-basket-analysis)

常见有效特征：

- 商品全局复购率、用户复购率、用户×商品购买次数。
- 距离上次购买的时间/订单数、连续购买 streak、平均复购周期。
- 用户对商品类别的支持度、商品在用户历史中的位置和近期性。
- 神经网络或矩阵分解先学习表示，再把表示/分数交给 LightGBM 等二级模型。

可迁移到本任务：

- 原数据没有明确日期，但有 raw sequence 和 run-length，可以构造：最后出现距离、最近连续 run 长度、出现间隔均值、最近窗口计数、历史集中度。
- 对 repeat 候选单独构造 `last_position`、`gap_from_last`、`n_runs`、`last_run_length`、`recent_window_count`。

## 3. 相关论文

### 3.1 重复推荐与探索

#### RepeatNet

- 论文：[RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/4408)

核心思想：先预测当前行为属于 repeat 还是 explore；repeat 分支只在历史商品中分配概率，explore 分支在新商品中分配概率，再由 gate 融合。

本任务中的轻量实现：

1. 用一个二分类模型预测 `P(target in history)`。
2. repeat ranker 只排序历史商品；explore ranker 排序非历史商品和冷启动候选。
3. 最终分数：`gate * repeat_score + (1-gate) * explore_score`。
4. gate 特征可用历史长度、唯一商品数、重复率、最大计数占比、最近 run、用户类别和历史类别熵。

#### Repetition and Exploration in Sequential Recommendation

- 论文：[Repetition and Exploration in Sequential Recommendation](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf)

核心启示：不同用户和不同阶段的 repeat/explore 倾向并不相同，应单独评估重复目标与探索目标，而不是只看总体指标。

### 3.2 Item embedding 与协同召回

#### Item2Vec

- 论文：[Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/abs/1603.04259)

核心思想：把用户商品序列当作句子，把商品当作 token，用 Skip-Gram/负采样学习稠密 item embedding。

建议实验：

- 使用 raw run 序列或 dedup 序列，分别测试窗口 3/5/10、维度 16/32/64。
- 生成 `cos(candidate,last_item)`、最近 3 个商品最大相似度、历史加权平均相似度。
- 用历史 embedding 的 recency-weighted mean 作为用户短期表示，与候选做余弦或点积。
- embedding 既用于召回，也作为 XGB/LGB/DeepFM 特征。

#### EASE

- 论文：[Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375)
- ACM 页面：[DOI 10.1145/3308558.3313710](https://dl.acm.org/doi/10.1145/3308558.3313710)

核心思想：对隐式反馈矩阵学习一个闭式解的 item-to-item 线性模型，结构很浅但在 Top-N 推荐上经常很强。

项目已有 EASE 实现，但 `auto` 因商品数大于阈值会选择 ItemKNN。商品数实际只有 2,156，仍可尝试：

- `collab=ease`；`ease_lambda` 测试 50/100/250/500/1000。
- 不只把 EASE 当唯一 collab，而是与 ItemKNN 同时保留为两列分数。
- 对 EASE 分数输出 last/max/sum/decay-sum 多个聚合视角。

#### BPR

- 论文：[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)

核心思想：对每个用户优化“正商品分数高于负商品”的 pairwise ranking objective。

建议用法：不一定替换最终 ranker，可训练 16/32 维 BPR-MF，输出用户-候选分数和 item embedding 相似度，作为召回通道及二级特征。

#### LightGCN / SVD-GCN

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- [SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation](https://arxiv.org/abs/2208.12689)

核心思想：在用户-商品交互图上传播 embedding；SVD-GCN 进一步指出图协同过滤与低秩分解关系紧密。

适配判断：

- 本任务用户与测试用户不重叠，直接学习 user ID embedding 对测试无用。
- item embedding 和 item-item 高阶关系仍然有价值。
- 优先试共现矩阵 Truncated SVD；它比完整 LightGCN 更轻、更稳定，并可对冷启动测试用户通过历史 item embedding 聚合得到用户向量。

### 3.3 序列推荐

#### GRU4Rec

- 论文：[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)

用 GRU 编码 session 并预测下一商品，是 session recommendation 的经典基线。

适配判断：测试序列很短且 35% 为空，GRU4Rec 只能覆盖非冷启动用户，建议作为额外召回/分数特征，不建议一开始替代整个现有流水线。

#### SASRec

- 论文：[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)

使用单向自注意力从历史中选择与下一行为最相关的商品。相比 BERT4Rec，更贴近严格的 next-item 推断。

轻量配置建议：1～2 层、hidden 32/64、2 个 attention heads、最大序列长度 20/50。输出 top100 候选或候选 logits，交给二级 ranker。

#### BERT4Rec

- 论文：[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)
- 复现研究：[A Systematic Review and Replicability Study of BERT4Rec](https://eprints.gla.ac.uk/275645/)

通过随机 mask 商品的 Cloze 任务训练双向 Transformer。复现研究指出训练时长和实现细节会显著影响结果。

适配判断：本任务规模较小、历史长度与测试分布差异大，BERT4Rec 成本和调参风险高于 SASRec，优先级靠后。

#### STAMP

- 论文：[STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://arxiv.org/abs/1803.00794)

同时建模 session 的总体兴趣与最后点击代表的短期兴趣。这个结构与当前 `repeat + last-item HT + history aggregation` 很接近，可以先借鉴其特征思想，而不必完整实现网络。

#### 简单近邻与复杂模型的比较

- [Evaluation of Session-based Recommendation Algorithms](https://arxiv.org/abs/1803.09587)
- [Sequential recommendation: A study on transformers, nearest neighbors and sampled metrics](https://doi.org/10.1016/j.ins.2022.07.079)

两项研究都提醒：在部分 session/sequence 数据上，合理调优的近邻方法可与深度模型持平甚至更强；采样评估也可能改变模型排名。本项目坚持真实候选和 outer holdout 是正确方向。

## 4. 针对当前项目的可执行实验

下面按“预期性价比”排序。提升幅度均应视为实验假设，不应直接根据公开榜反复调参。

| 优先级 | 实验 | 具体实现 | 主要影响 | 计算成本 |
|---|---|---|---|---|
| P0 | 多视角共现/转移矩阵 | 相邻 run、有向距离衰减、last→target、类别条件矩阵；输出 last/max/sum/mean/decay-sum/rank | 召回 + 排序 | 低 |
| P0 | Truncated SVD item embedding | 对 item-item 共现或 user-item 矩阵做 16/32/64 维 SVD | 非 repeat 泛化、冷启动历史聚合 | 低 |
| P0 | Item2Vec | window 3/5/10，dim 16/32/64；加入相似度聚合 | 非 repeat 召回与稠密转移 | 低～中 |
| P0 | 用户历史类别画像 | 历史类别占比/熵/主类别，候选类别匹配、条件 target prior | 用户画像 × 商品交互、冷启动 | 低 |
| P0 | Repeat-Explore gate | 独立预测 repeat 概率，融合 repeat/explore 两套分数 | 直接利用 56% repeat | 低～中 |
| P1 | EASE + ItemKNN 双通道 | 两套分数同时作为特征；EASE λ 网格 | 补充协同视角 | 中 |
| P1 | top100→top25 两阶段压缩 | 先用浅模型/规则保召回，再用主 ranker 排 top25/top10 | 提高 candidate recall | 中 |
| P1 | BPR-MF | 16/32 维，输出召回和相似度特征 | 协同 embedding | 中 |
| P1 | XGB/LGB/DeepFM 融合 | 分数归一化或 reciprocal-rank fusion | 降低模型方差 | 低 |
| P2 | SASRec 轻量召回 | 1～2 层、hidden 32/64，作为额外通道 | 非线性序列模式 | 中～高 |
| P2 | LightGCN | 只保留 item embedding/score，不依赖测试 user ID | 高阶协同 | 中～高 |
| P3 | BERT4Rec | 小模型 masked-item 训练 | 双向序列表达 | 高 |

## 5. 推荐的最小实验队列

### E0：冻结可信基线

- 固定一个 XGB 基线和统一 outer folds。
- 记录 `candidate_recall@25/50/100`、`NDCG@10`、`succ_ndcg`。
- 分桶记录 cold、短历史、repeat target、explore target。

### E1：扩展当前矩阵特征

在现有 Markov/HTarget/ItemKNN 基础上增加：

- `last_item_score`
- `last3_max_score`
- `hist_mean_score`
- `hist_decay_sum_score`
- `source_rank`
- `n_sources_recalled`

先不改变候选集合，只判断排序收益；再把新矩阵加入候选生成判断 recall 收益。

### E2：SVD embedding

- 构建 item-item 共现矩阵。
- Truncated SVD：dim 16/32/64。
- 对每个候选生成 last cosine、history max/mean/decay-weighted cosine。
- 优先选择 outer fold 0、1、2 做快速筛选。

### E3：Item2Vec

- dedup 和 raw-runs 分别训练，避免连续重复商品完全支配窗口。
- window 3/5/10，dim 32 起步。
- 与 SVD 做消融及同时加入。

### E4：类别画像与冷启动 prior

- 每个用户历史类别分布与候选类别的匹配。
- 分别基于 8 个用户属性计算 target prior，不只保留当前总和。
- 冷启动用户用分群 prior + 全局 target prior + popularity。

### E5：Repeat-Explore gate

- 标签：训练 target 是否在截断后的历史中。
- gate 模型可先用 LogisticRegression/LightGBM。
- repeat 分支重点使用 count、last position、run、recency。
- explore 分支重点使用 target prior、用户条件 prior、SVD/Item2Vec、类别匹配。

### E6：融合

- 保留 XGB、LGB、DeepFM 的 OOF 分数。
- 先试简单 rank average，再试权重网格。
- 融合权重只能在 outer validation 上确定。

## 6. 评估与防泄漏要求

1. 所有 target frequency、用户条件 target prior、target 转移矩阵必须只使用对应训练折。
2. Item2Vec/SVD 若使用无标签历史序列，可以使用训练折全部历史；不要把验证 target 当作序列尾部加入训练。
3. 每项实验同时报告候选召回和最终 NDCG，避免“召回提高但排序下降”被掩盖。
4. 至少使用 3 个 outer folds 做初筛；最终候选方案使用全部 10 folds 报均值与标准差。
5. 分桶指标建议：
   - `history_len=0`
   - `history_len=1~3`
   - `history_len>=4`
   - `target_in_history=True/False`
   - `known_target=True/False`
6. 不用单次公开榜结果决定参数；公开榜只用于验证本地与线上方向是否一致。

## 7. 最推荐的实施顺序

如果目标是尽快从约 0.50 冲到更高分，建议顺序为：

1. 多视角共现矩阵及其聚合特征。
2. Truncated SVD item embedding。
3. 用户历史类别画像与分群冷启动 prior。
4. Item2Vec。
5. Repeat-Explore gate。
6. XGB/LGB/DeepFM 融合。
7. EASE 双通道或 BPR。
8. 最后再考虑 SASRec/LightGCN/BERT4Rec。

这个顺序优先复用当前 `build_features.py → ranker` 架构，每一步都能独立做 OOF 消融，也符合 Kaggle 上位方案“先把候选与特征做厚，再考虑复杂模型”的经验。

