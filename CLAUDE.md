# CLAUDE.md — 推荐任务 (REC_TASK)

AFAC2026 比赛推荐子任务：序列推荐 / 下一物品预测，NDCG@10 评测。工作目录 `C:\Users\26972\Desktop\AFAC2026比赛\REC_TASK`。Shell 是 bash（Windows 下用 Unix 语法、正斜杠）。

## 任务速览
- 给定用户历史点击序列，预测每个用户的下一个目标物品，输出 top-10。
- 评测 NDCG@10。提交格式 `uid,prediction`，prediction=逗号分隔10个iid，顺序对齐 `data/A推荐/sample_submission.csv`。
- A榜：40000 train / 10000 test / 2156 item。~56% target 在用户历史里（repeat/recency 是主信号）；35% test 用户冷启动（空历史）；仅235个item曾作训练target。
- 约束：单4090、≤1GB、轻量可复现；本流水线纯 CPU（XGBoost+稀疏协同），无 GPU 依赖。

## 当前基线（已验证可信）
本地 holdout NDCG@10 = **0.4965**（train_candidate_k=200, hard_ratio=0.75）。
- 线上 A2.csv（旧21候选模式）= 0.2476；线上 A2_pipe.csv（同构200）≈0.49 量级。
- **本地↔线上已对齐**：评估无泄漏可信，本地分可作为线上分的代理。
- 关键跃迁：训练group从21候选→200候选（与val难度对齐）使 NDCG 从 0.2457→0.49。

## 流水线架构（四层解耦）
```
run.py        入口：config + CLI，跑前打印本次参数
  ↓
config.py     统一配置：PIPELINE / FEATURES / MODELS（模型名=实验名）
  ↓
pipeline.py   流程层：数据构建→训练→验证/提交，与模型/特征解耦
  ↓                                        ↓
build_features.py ─→ core/feature_core.py   models/（注册表）
  特征层(OOF/holdout/test)   底层信号/解析     base/registry/xgb_ranker
```

### 各层职责
- **run.py**：解析 CLI → `config.get_config(model)` 合并配置 → 打印配置 → `pipeline.run_experiment(cfg)`。`--param 深路径` 临时覆盖。
- **config.py**：三块 `PIPELINE`/`FEATURES`/`MODELS`。模型名即实验名，`get_config(name)` 合并出 `{model, model_params, features, pipeline}`。加新模型=MODELS加一项。
- **pipeline.py**：`run_experiment` 分 holdout/submit 两模式。模型只调 `fit`/`predict_scores`，排序/NDCG/取topk 在此复用（`eval_ndcg`/`topk_predictions`）。`_holdout_kwargs`/`_oof_kwargs`/`_test_kwargs` 隔离 config↔build_features 的参数命名。
- **build_features.py**：特征层。`_feature_schema`+`_assemble_features` 定义22列特征；`build_oof_features`/`build_holdout_features`/`build_test_features` 三入口；`evaluate_recall` 评端到端召回。依赖 `core.feature_core` 的无状态信号函数。
- **core/feature_core.py**：底层信号（序列解析、run-length截断、EASE/ItemKNN、Markov、HTarget、流行度）。
- **models/**：`base.RankerModel`(fit/predict_scores 抽象) + `registry`(@register/get_model) + 各实现。当前 `xgb_ranker`（XGBoost rank:pairwise）。

## 用法
```bash
# 验证（可信 holdout NDCG，本地≈线上）
python run.py --mode holdout
# 生成提交
python run.py --mode submit --out submissions/A3.csv
# 临时调参（深路径，可多次）
python run.py --param features.train_candidate_k=100 --param features.hard_negative_ratio=1.0
# 列模型 / 看合并配置不跑
python run.py --list
python run.py --show
# 单独构建特征落盘（不训练）
python build_features.py --holdout          # → features/*.parquet + .meta.json
python build_features.py --test-features
python build_features.py --recall           # 端到端 Recall@K
```

## 关键参数（config.py FEATURES）
- `candidate_k=200`：val/test 召回候选数。
- `train_candidate_k=200`：训练group大小（1正+其余负）。与 candidate_k 同值=难度对齐（配置打印标注）。**这是最强杠杆**。
- `hard_negative_ratio=0.75`：负样本中难负（召回top）占比，其余随机负。0.75 比 1.0 略好（混随机负防过拟合）。
- `collab=auto`：协同信号。auto按`ease_max_items=1500`选——A榜2156>1500走稀疏ItemKNN（非EASE）；想跑真EASE用`--param features.collab=ease`（慢，2156能跑）。
- 折划分：`outer_folds=5/outer_fold=0/inner_folds=4`（holdout）、`n_folds=4`（submit）。
- `watch_frac=0.2`：holdout从train切独立watch做early-stop的比例。

## 特征（22列，6组，定义在 _feature_schema/_assemble_features）
- A 交互：in_hist/count/count_norm（主信号，~56% target在历史）
- B 全局：popularity（冷启动兜底）
- C 协同：repeat/collab_score(默认ItemKNN非EASE)/markov/htarget（htarget最强）
- D 用户：u_cat_01~08（整数编码，cat_cols，LGBM需categorical_feature；XGBoost当数值喂）
- E 商品：i_cat_01~03(类别)/i_bucket_01(有序)
- F 冷启动条件：ufeat_target_cond/lastcat_target_cond
- XGBoost重要性：count_norm>htarget>repeat>ufeat>lastcat>in_hist。u_cat靠后→缺用户画像聚合特征。

## 扩展方式（互不干扰 pipeline）
- **加模型**：`models/`建 RankerModel 子类 + `@register`，`config.MODELS`加一项，`--model 新名`跑。
- **加特征**：改 `build_features.py` 的 `_feature_schema`+`_assemble_features`，pipeline 自动透传 feat_cols/cat_cols。
- **调参**：改 config.py 或 `--param 深路径`。

## 关键设计原则（务必遵守）
1. **无泄漏**：OOF协议，每折统计只来自其余折，验证折target不进自己特征。`build_holdout_features`/`build_oof_features` 已隔离；`build_test_features`用全量训练统计（test target隐藏，无泄漏）。
2. **诚实评估**：holdout从train切独立watch做early-stop，**val不参与选模**。否则会得乐观泄漏分（用val做early-stop+报分≈0.37，不可信）。本地0.4965是诚实的。
3. **诚实截断**：训练样本用 `truncate_runs` 按test长度分布截断历史模拟；test直接用自带`item_seq_dedup`（已是最终形态，不截断）。必须run-length重建，不能直接切item_seq_dedup。
4. **候选难度对齐**：训练/验证group同构（都train_candidate_k/candidate_k），否则排序器在易负上学、迁移到难候选打折。
5. **NDCG评估两类口径**：排序器表(1正+N负强塞)只能评NDCG排序力；端到端召回用`evaluate_recall`(不强塞、全商品)。val Recall@200≈0.996（召回很强，瓶颈在排序）。

## 目录结构
```
run.py / config.py / pipeline.py / build_features.py   顶层代码
core/feature_core.py        底层信号
models/                     base.py/registry.py/xgb_ranker.py
features/                   特征产物(parquet+meta.json)
submissions/                提交csv(A2.csv=0.2476旧版, A2_pipe.csv=同构版)
data/A推荐/                 原始5个csv
docs/                       任务说明/学习笔记/论文列表/赛题说明
README.md                   结构速览
```

## 已知改进点（下一步候选）
1. 补用户画像聚合特征（用户历史类别分布×候选类别匹配度）——特征重要性显示u_cat弱，性价比最高。
2. lastcat_target_cond名不副实（是last-item的HT行非类别条件），与htarget冗余。
3. collab_score默认itemknn，A榜可试`collab=ease`对比。
4. hard_negative_ratio 可网格搜索（0.5/0.75/1.0）。
