# 推荐任务 (REC_TASK)

序列推荐 / 下一物品预测，NDCG@10 评测。本地 holdout NDCG@10 ≈ 0.2457，线上 A2 = 0.2476（已对齐，评估可信）。

## 目录结构

```
REC_TASK/
├── run.py              # 实验入口（config + CLI），改方法主要改这里
├── pipeline.py         # 流程层：数据构建→训练→验证/提交，与模型/特征解耦
├── build_features.py   # 特征层：OOF/holdout/test 三种入口，信号/组装已分离
├── core/
│   └── feature_core.py # 底层信号/解析（序列解析、EASE/ItemKNN、Markov、HTarget 等）
├── models/             # 模型层：注册表机制
│   ├── base.py         #   RankerModel 统一接口 fit / predict_scores
│   ├── registry.py     #   @register / get_model(name)
│   └── xgb_ranker.py   #   XGBoost LambdaMART（已注册 "xgb_ranker"）
├── features/           # 特征产物（parquet + meta.json）
├── submissions/        # 提交 csv（对齐 sample_submission）
├── data/A推荐/         # 原始 5 个 csv
└── docs/               # 任务说明、学习笔记、论文列表
```

## 用法

```bash
# 验证（诚实 NDCG，本地≈线上）
python run.py --mode holdout --exp xgb_default

# 生成提交
python run.py --mode submit --exp xgb_default --out submissions/A2.csv

# 调参（不改代码）
python run.py --mode holdout --param lr=0.1 --param max_depth=8
```

## 改方法（互不干扰 pipeline）

| 改什么 | 改哪里 | pipeline 要动吗 |
|---|---|---|
| 换模型 | `models/` 新建文件，实现 `RankerModel` + `@register`，config `model` 改名 | 否 |
| 加特征 | `build_features.py` 的 `_feature_schema` + `_assemble_features` | 否 |
| 调参 | `run.py` 的 `EXPERIMENTS` 或 `--param` | 否 |

## 关键设计

- **模型只打分**，排序/NDCG/取 top-k 在 pipeline 复用，与具体模型无关。
- **诚实评估**：holdout 从 train 切独立 watch 做 early-stop，val 不参与选模。
- **无泄漏**：OOF 协议，每折统计只来自其余折，验证折 target 不进自己的特征。
