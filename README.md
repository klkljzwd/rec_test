# 推荐任务 (REC_TASK)

AFAC2026 比赛推荐子任务：序列推荐 / 下一物品预测，NDCG@10 评测。给定用户历史点击序列预测下一个物品，输出 top-10。约束：单 4090、≤1GB、轻量可复现；本流水线纯 CPU（XGBoost / LightGBM + 稀疏协同），无 GPU 依赖。

- 数据：40000 train / 10000 test / 2156 item。约 56% target 在用户历史里（repeat/recency 是主信号），35% test 用户冷启动（空历史），仅 235 个 item 曾作训练 target。
- 当前基线：本地 holdout NDCG@10 = **0.5043**（xgb_ranker，candidate_k=25/train_candidate_k=50，outer 10 折 fold 0）。
- 本地↔线上已对齐，本地分可作为线上分的代理。

## 目录结构

```
REC_TASK/
├── run.py              # CLI 入口：--model/--mode/--config/--param，调 experiment
├── experiment.py       # 程序化入口（agent 对接）：run(cfg)->metrics + 落盘结果
├── build_features.py   # 特征层：OOF/holdout/test 三入口 + evaluate_recall，信号/组装分离
├── core/
│   └── feature_core.py # 底层信号/解析（序列解析、truncate_runs、EASE/ItemKNN、Markov、HTarget、target_freq 等）
├── models/             # 模型层：每模型一个目录，全量自包含
│   ├── __init__.py     #   导入各模型触发 @register + @register_runner
│   ├── base.py         #   RankerModel 统一接口 fit / predict_scores
│   ├── registry.py     #   双注册：类注册(get_model) + runner 注册(get_runner)
│   ├── dispatcher.py   #   run_experiment(cfg) -> get_runner -> models/<name>/run.py
│   ├── _shared/        #   跨模型共享基建
│   │   ├── defaults.py     # FEATURES/PIPELINE 参考默认（新模型拷贝用，不参与运行时合并）
│   │   ├── config_utils.py # 合并工具 + load_from_json + load_by_name
│   │   ├── eval_core.py    # 评估原语（ndcg_at_k/eval_ndcg/topk_predictions/make_metrics）——口径统一地基
│   │   └── logging.py      # config_fingerprint + log_experiment（experiments/log.jsonl）
│   ├── xgb_ranker/     # XGBoost rank:ndcg（目录名=模型名）
│   │   ├── config.py   #   全量自包含：model_params + features + pipeline
│   │   ├── load.py     #   load(overrides)->cfg（加载在模型目录）
│   │   ├── model.py    #   XGBRanker 架构
│   │   ├── data.py     #   config↔build_features kwargs 映射 + split_watch
│   │   └── run.py      #   holdout/submit 训练+评估+统一 metrics（@register_runner）
│   └── lgb_ranker/     # LightGBM LambdaRank（同构四文件，cat_cols 走 categorical_feature）
├── features/           # 特征产物（parquet + meta.json）
├── experiments/        # log.jsonl（历次实验）+ results/<fingerprint>.json（单次结果）
├── submissions/        # 提交 csv（对齐 sample_submission）
├── data/A推荐/         # 原始 5 个 csv
└── docs/               # 任务说明、学习笔记、论文列表
```

## 架构：每模型自治 + 统一 config

每个模型目录**全量自包含**整个流程的所有可调参数（model_params + features + pipeline），加载函数也在自己目录（`load.py`）。中央无 config.py。

**调用链（直给，无间接）：**
```
agent:  experiment.run(cfg_path)  →  config_utils.load_from_json / load_by_name  →  dispatcher.run_experiment  →  models/<name>/run.py  →  eval_core
CLI:    run.py                    →  experiment.run_from_cfg（与 agent 同一条路）
```

**两条核心约束：**
- **评估口径统一**：NDCG/topk 只走 `models/_shared/eval_core.py`，各 run.py 不许自写。横向比模型的前提。
- **无泄漏**：OOF 协议，每折统计只来自其余折，验证折 target 不进自己特征；holdout 从 train 切独立 watch 做 early-stop，val 不参与选模。

## 用法

```bash
# 验证（可信 holdout NDCG，本地≈线上）
python run.py --mode holdout                          # 默认 xgb_ranker
python run.py --mode holdout --model lgb_ranker

# 生成提交（对齐 sample_submission：uid,prediction=10个iid）
python run.py --mode submit --out submissions/A3.csv

# 临时调参（深路径，可多次）
python run.py --mode holdout \
    --param features.candidate_k=200 \
    --param model_params.max_depth=8

# JSON 配置入口（agent 友好，可叠加 --param）
python run.py --config experiments/xxx.json --mode holdout

# 辅助
python run.py --list          # 列可用模型
python run.py --show          # 打印合并后配置不运行（UTF-8 输出）
```

### agent 对接（程序化入口，不碰 argparse/不解析 stdout）
```python
import experiment
metrics = experiment.run("experiments/cfg.json")          # JSON 路径
metrics = experiment.run({"model": "xgb_ranker", ...})    # dict
metrics = experiment.run("cfg.json", overrides={"features.candidate_k": 200})  # 带覆盖
# 返回 {"ndcg@10":..., "succ_ndcg":..., "n_recalled":..., "n_users":...}
# 副作用：写 experiments/results/<fingerprint>.json + 追加 experiments/log.jsonl
```

## 关键参数（每模型 config.py 的 features/pipeline）

- `candidate_k=25`：val/test 召回候选数。
- `train_candidate_k=50`：训练 group 大小（1正+其余负）。与 candidate_k 同值=难度对齐；当前不对齐（50>25），实测效果好。
- `hard_negative_ratio=1.0`：负样本中难负（召回 top）占比。
- `collab=auto`：A 榜 2156>1500 走稀疏 ItemKNN（非 EASE）；想跑真 EASE 用 `--param features.collab=ease`。
- 折划分：`outer_folds=10/outer_fold=0/inner_folds=4`（holdout）、`n_folds=4`（submit）。
- `watch_frac=0.2`：holdout 从 train 切独立 watch 做 early-stop 的比例。

## 特征（46 列，定义在 build_features._feature_schema + _assemble_features）

- **A 交互**：in_hist/count/count_norm/count_log_norm（主信号，~56% target 在历史）
- **A2 复购近期性**：is_last_item/last_position_norm/run_count/run_count_norm
- **B 全局**：popularity（冷启动兜底）
- **C 协同**：repeat/collab_score(ItemKNN)/markov/htarget（htarget 最强）
- **D target 先验**：is_known_target/target_freq/target_freq_log（商品当过训练 target 的频次，与 popularity 互补；同时以 log 先验参与候选生成，专攻非 repeat/冷启动）
- **E 用户**：u_cat_01~08（cat_cols，LGB 走 categorical_feature）
- **F 商品**：i_cat_01~03/i_bucket_01
- **G 历史类别画像**：每个商品属性的 hist_*_share 与 last_*_match（用户历史分布 × 候选属性）
- **H 冷启动条件**：ufeat_target_cond + 8 个 ucond_* 分解先验 + lastcat_target_cond

## 加新模型

建 `models/<新模型名>/` 目录（目录名=模型名），放 4 个文件：
- `config.py`：全量 `MODEL_CONFIG = {model_params, features, pipeline}`（可从 `_shared/defaults.py` 拷贝改）
- `load.py`：`load(overrides)->cfg`（照 xgb_ranker/load.py 抄）
- `model.py`：模型架构（候选表模型实现 `RankerModel`；端到端序列模型可只提供 run.py）
- `run.py`：`@register_runner("<名>")` 装饰 `run(cfg)->metrics`，调 `eval_core` 评估

在 `models/__init__.py` 加 import 触发注册。**不碰任何其他文件**。候选表模型额外 `@register` 类注册供 `get_model` 实例化。

## 已验证的关键结论（实验沉淀）

- **recall@100≈0.985 / recall@20≈0.828**：target 在规则 top100 池里，瓶颈是"top100→top20 压缩"，不是"找商品"。
- **手调规则权重几乎无效**（+0.06pp）；**兼职 ranker 压缩 +1.1pp**；**专用浅召回模型 ≈ 兼职 ranker**（同批特征下挤不出更多）。模型侧到顶在 recall@20≈0.839。
- **overall = recall@20 × succ_ndcg**（乘法）：recall@20 每提 1pp → overall +0.006，是排序特征收益的 3~6 倍。但 recall 升时 succ 可能降，互相拉扯——必须端到端看 overall。
- **succ 不掉的压缩路径**有价值：ranker 重排 top100→top20 时 succ≈0.59 不降。
- **DeepFM/GRU4Rec 评估**：DeepFM 能塞进 RankerModel 接口但小数据+冷启动+repeat 主导下预期持平~略差；GRU4Rec 与可信 OOF 协议冲突（序列 target 即训练标签，需单独无泄漏序列划分），且 35% 冷启动无序列可用。两者非冲 0.51 的杠杆。
- **冲 0.51 的杠杆在特征侧**：item embedding（攻非 repeat target 的稠密转移泛化）+ 用户画像交互特征，直击 recall@20 那 ~1164 个"在 top100 池里但排不进 top20"的 target。

## 已知改进点（下一步候选）

1. item embedding 特征（item2vec/共现 SVD，全量序列无监督无泄漏）——冲 0.51 主力。
2. 用户画像聚合特征（用户历史类别分布 × 候选类别匹配度）——廉价补充。
3. xgb + lgb 分数融合——模型族不同，融合必涨，尾部 +0.001~0.003。
4. 提 collab 权重（现 2.0 最低）/ 换 EASE——便宜试探。
