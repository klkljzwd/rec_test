
## 2026-07-01 20:00:20 — A_regularization_probe

- baseline (config.py MODEL_CONFIG): model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 outer=0/10 watch_frac=0.1
- folds=[0, 1, 2]  baseline_mean_ndcg=0.50404  target=0.51500

| config | fold0 | fold1 | fold2 | ndcg_mean | Δvs_base | succ_mean | best_iters |
|---|---|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51002 | 0.50208 | 0.50004 | 0.50404 | +0.00000 | 0.58940 | 1389,1039,904 |
| min_child_weight=5 (min_child_weight=5.0) | 0.50930 | 0.50213 | 0.50157 | 0.50433 | +0.00029 | 0.58974 | 1093,534,1271 |
| min_child_weight=10 (min_child_weight=10.0) | 0.51072 | 0.50352 | 0.50016 | 0.50480 | +0.00076 | 0.59029 | 1175,921,1135 |
| reg_lambda=3 (reg_lambda=3.0) | 0.51028 | 0.50312 | 0.50070 | 0.50470 | +0.00066 | 0.59017 | 1282,1171,1181 |
| max_depth=4 (max_depth=4) | 0.50826 | 0.50360 | 0.49972 | 0.50386 | -0.00018 | 0.58919 | 1137,1067,1274 |
| max_depth=6 (max_depth=6) | 0.50876 | 0.50373 | 0.49930 | 0.50393 | -0.00011 | 0.58927 | 1094,435,1166 |


## 2026-07-01 20:15:45 — B_recall_lever (FEATURE SWEEP)

- baseline model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- baseline features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 watch_frac=0.1
- folds=[0]  baseline_mean_ndcg=0.51002  target=0.51500

| config | fold0 | ndcg_mean | Δvs_base | succ_mean | recall_mean |
|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51002 | 0.51002 | +0.00000 | 0.59373 | 0.85900 |
| cand_k=50,tc_k=50 (candidate_k=50 train_candidate_k=50) | 0.50768 | 0.50768 | -0.00234 | 0.53951 | 0.94100 |
| cand_k=100,tc_k=50 (candidate_k=100 train_candidate_k=50) | 0.32899 | 0.32899 | -0.18103 | 0.33459 | 0.98325 |
| cand_k=100,tc_k=100 (candidate_k=100 train_candidate_k=100) | 0.51069 | 0.51069 | +0.00067 | 0.51939 | 0.98325 |
| hard_neg=0.5 (hard_negative_ratio=0.5) | 0.50943 | 0.50943 | -0.00059 | 0.59305 | 0.85900 |


## 调参中期小结 (fold0 baseline=0.51002, 3-fold baseline=0.50404, target=0.515)

### 3-fold 已确认（robust，跨折一致为正）
- min_child_weight=10: +0.00076（best_iter 略降但仍充分）—— 最佳模型正则
- reg_lambda=3: +0.00066
- max_depth: 5 为最优点（4 与 6 均略负）-> 保持 5

### fold0 单折结论（待 3-fold 确认）
- train_candidate_k: 25(0.502,太易) < 75(0.506,局部低点) < 50(0.510) ≈ 100(0.5107,+0.0007)。tc_k=100 略好但 2× 慢，收益在噪声内。
- candidate_k: 25 最优。50(succ 崩 0.540, -0.0023)、100 需 tc_k 对齐否则灾难(0.329)。对齐 100 仅 +0.0007 且 2× 慢。-> 保持 25。
- hard_negative_ratio: 1.0 最优(0.5->-0.0006)。-> 保持 1.0。
- itemknn_k: 200 最优(100->-0.0035, 400->-0.0005)。collab=ease 太慢(>12min, 1.6GB)不实用。-> 保持 200。

### 已确认无杠杆的方向
- 高 candidate_k：succ 崩塌 > recall 增益，net 负或中性。README "succ不降" 在本端到端 setup 不成立。
- 单一参数收益均 ≤+0.001，需叠加多参数。

### 待测 (Sweep E)
- subsample / colsample / watch_frac / lr (模型侧采样正则 + 早停可靠性)

## 2026-07-01 21:06:16 — E_train_dynamics

- baseline (config.py MODEL_CONFIG): model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 outer=0/10 watch_frac=0.1
- folds=[0]  baseline_mean_ndcg=0.51002  target=0.51500

| config | fold0 | ndcg_mean | Δvs_base | succ_mean | best_iters |
|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51002 | 0.51002 | +0.00000 | 0.59373 | 1389 |
| subsample=0.7 (subsample=0.7) | 0.50797 | 0.50797 | -0.00205 | 0.59135 | 1150 |
| subsample=0.6 (subsample=0.6) | 0.50822 | 0.50822 | -0.00180 | 0.59164 | 1395 |
| colsample=0.6 (colsample=0.6) | 0.50914 | 0.50914 | -0.00087 | 0.59272 | 1399 |
| colsample=1.0 (colsample=1.0) | 0.50833 | 0.50833 | -0.00169 | 0.59177 | 1200 |
| watch_frac=0.2 (watch_frac=0.2) | 0.51015 | 0.51015 | +0.00013 | 0.59388 | 915 |
| lr=0.04,n=2500 (lr=0.04 n_estimators=2500) | 0.50960 | 0.50960 | -0.00042 | 0.59325 | 1242 |


## 2026-07-01 21:32:28 — G_stack_and_weights (FEATURE SWEEP)

- baseline model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- baseline features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 watch_frac=0.1
- folds=[0]  baseline_mean_ndcg=0.51002  target=0.51500

| config | fold0 | ndcg_mean | Δvs_base | succ_mean | recall_mean |
|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51002 | 0.51002 | +0.00000 | 0.59373 | 0.85900 |
| mcw10+tc100 (train_candidate_k=100 min_child_weight=10.0) | 0.50849 | 0.50849 | -0.00153 | 0.59195 | 0.85900 |
| collab_w8 (score_weights={'collab': 8.0}) | 0.50777 | 0.50777 | -0.00225 | 0.59181 | 0.85800 |
| target_prior6 (score_weights={'target_prior': 6.0}) | 0.50781 | 0.50781 | -0.00221 | 0.59410 | 0.85475 |
| htarget50 (score_weights={'htarget': 50.0}) | 0.50939 | 0.50939 | -0.00063 | 0.59231 | 0.86000 |


## Sweep D/E/F/G 结论补记
- Sweep D (collab): itemknn_k=200 最优(100->-0.0035, 400->-0.0005)；collab=ease 太慢(>12min/1.6GB)不实用。
- Sweep E (train dynamics): subsample=0.8/colsample=0.8/watch_frac=0.1/lr=0.06 均为最优点，任一方向皆劣。
- Sweep F (组合): min_child_weight+reg_lambda 组合 antagonistic（mcw10+l3=-0.0013 < mcw10 单独 +0.0007）。两者同属正则化，叠加过正则。
- Sweep G (异质叠加+权重): mcw10+tc100=-0.0015(underfit,best_iter605)；score_weights(collab_w8/target_prior6/htarget50)均劣。确认 README"手调权重几乎无效"。

## 阶段结论
config.py 参数空间已穷尽。唯一稳健正收益：**min_child_weight=10 单独 +0.00076(3-fold)**。
其余参数均在最优点。Sweep H 验证 min_child_weight/reg_lambda 单独推高是否更优。

## 2026-07-01 21:43:49 — H_push_single

- baseline (config.py MODEL_CONFIG): model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 outer=0/10 watch_frac=0.1
- folds=[0]  baseline_mean_ndcg=0.51002  target=0.51500

| config | fold0 | ndcg_mean | Δvs_base | succ_mean | best_iters |
|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51002 | 0.51002 | +0.00000 | 0.59373 | 1389 |
| mcw=15 (min_child_weight=15.0) | 0.51048 | 0.51048 | +0.00046 | 0.59427 | 1171 |
| mcw=20 (min_child_weight=20.0) | 0.50870 | 0.50870 | -0.00132 | 0.59220 | 1292 |
| mcw=30 (min_child_weight=30.0) | 0.50873 | 0.50873 | -0.00129 | 0.59223 | 1221 |
| lambda=5 (reg_lambda=5.0) | 0.50799 | 0.50799 | -0.00203 | 0.59138 | 1269 |
| lambda=10 (reg_lambda=10.0) | 0.50815 | 0.50815 | -0.00186 | 0.59156 | 1184 |


## 各参数对结果的影响（穷尽实测，3-fold/fold0 标注）

### 指标结构
overall ndcg@10 = recall × succ_ndcg（乘法）。baseline: recall@25=0.859 × succ=0.594 = 0.510。
recall 由 candidate_k 决定（候选集是否含 target）；succ 由 ranker 排序质量决定。
=> 要 overall +0.005(到0.515)，需 succ +0.006(到0.600，recall 不变) 或 recall+succ 联动。succ 受固定特征上界限制，参数调不动 +0.006。

### model_params（XGB 超参）
| 参数 | baseline | 测过 | 最优 | 影响 | 结论 |
|---|---|---|---|---|---|
| min_child_weight | 1.0 | 5,10,15,20,30 | **10** (+0.00076@3f) | 叶最小权重，正则叶分裂；10 平衡，>=20 过正则 succ 降 | **唯一正收益** |
| reg_lambda | 1.0 | 3,5,10 | 3 (+0.00066@3f) | L2；3 略好，>=5 过正则 | 次优但与 mcw 组合 antagonistic |
| max_depth | 5 | 4,6 | 5 | 树深；4 欠拟合，6 过拟合，5 最优 | 保持 5 |
| subsample | 0.8 | 0.6,0.7 | 0.8 | 行采样；越低数据越少越差(0.6<0.7<0.8) | 保持 0.8 |
| colsample | 0.8 | 0.6,1.0 | 0.8 | 列采样；双向皆劣(0.6欠/1.0过拟合) | 保持 0.8 |
| lr | 0.06 | 0.04,0.05,0.08 | 0.06 | 学习率；过低收敛不足，过高不稳 | 保持 0.06 |
| n_estimators | 1400 | 3000(同分) | 1400 | 上限；early_stop 300 下 best_iter~1100-1390，1400 够，更多无益 | 保持 1400 |
| early_stopping | 300 | (经 n_est 间接) | 300 | 早停耐心；不改变 outer val | 保持 300 |

### features（数据构建）
| 参数 | baseline | 测过 | 最优 | 影响 | 结论 |
|---|---|---|---|---|---|
| candidate_k | 25 | 50,100 | 25 | val 候选数；↑recall 但 succ 崩塌更狠(net 负/中性)。100 需 tc_k 对齐否则灾难(0.329) | 保持 25 |
| train_candidate_k | 50 | 25,75,100 | 50~100 | 训练 group 大小；25 太易(-0.008)，75 局部低，100 仅+0.0007(2x慢,噪声内)。须与 cand_k 对齐 | 保持 50 |
| hard_negative_ratio | 1.0 | 0.5 | 1.0 | 难负占比；1.0(全难)最优，0.5 略差 | 保持 1.0 |
| itemknn_k | 200 | 100,400 | 200 | ItemKNN 邻居数；100 信号不足，400 略平 | 保持 200 |
| collab | auto(itemknn) | ease | itemknn | EASE 更慢(>12min/1.6GB)且 auto 已选 itemknn(2156>1500) | 保持 auto |
| score_weights | 定制 | collab8/tp6/ht50 | 定制 | 改任一权重皆劣(collab↑-0.0023, tp_on-0.0022 且 recall 降) | 保持 |
| inner_folds | 4 | 6,8 | 4 | OOF 折数；6/8 噪声波动，无系统增益 | 保持 4 |

### pipeline
| 参数 | baseline | 测过 | 最优 | 结论 |
|---|---|---|---|---|
| watch_frac | 0.1 | 0.2 | 0.1 | 早停 watch 比例；0.2 持平(+0.0001)，0.1 足够 | 保持 0.1 |

### 组合性结论
- min_child_weight + reg_lambda 组合 antagonistic（同属正则化，叠加过正则，mcw10+l3=-0.0013）。
- min_child_weight + train_candidate_k=100 组合 underfit（best_iter 605，-0.0015）。
- 无任何组合能叠加出 +0.005。

### 最终
config.py 参数空间已穷尽。**唯一改动：min_child_weight 1.0 → 10.0（+0.00076@3-fold）**。
理论上限：fold0 0.510→~0.511，3-fold 0.504→~0.505。距 0.515(+0.005)差一个数量级，
且 +0.005 需 succ+0.006，受固定特征上界限制——README 亦指出 0.51→0.515 杠杆在特征工程(item embedding)/模型融合，非 config.py 调参。

## 2026-07-01 22:12:46 — J_verify_mcw10

- baseline (config.py MODEL_CONFIG): model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 1.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 outer=0/10 watch_frac=0.1
- folds=[3, 4, 5, 6, 7]  baseline_mean_ndcg=0.50559  target=0.51500

| config | fold3 | fold4 | fold5 | fold6 | fold7 | ndcg_mean | Δvs_base | succ_mean | best_iters |
|---|---|---|---|---|---|---|---|---|---|
| baseline ((baseline)) | 0.50457 | 0.50302 | 0.49952 | 0.51149 | 0.50936 | 0.50559 | +0.00000 | 0.58653 | 1000,1370,1354,800,952 |
| mcw=10 (min_child_weight=10.0) | 0.50432 | 0.50400 | 0.49783 | 0.51125 | 0.50980 | 0.50544 | -0.00015 | 0.58635 | 1399,1335,1381,884,1327 |


## 最终验证 (官方 run.py --mode holdout, fold0)
- 改动：min_child_weight 1.0 -> 10.0（唯一正方向，其余参数确认最优不动）。
- 结果：ndcg@10=**0.51072** (baseline 0.51002, +0.0007 确定性提升), succ=0.59455, recall=3436/4000(85.9%)。
- harness 预测 0.51072 == run.py 实测 0.51072（逐位一致，流程等价性已证）。

## 8-fold 稳健性 (min_child_weight=10 vs baseline, folds 0-7)
| fold | base | mcw10 | Δ |
|---|---|---|---|
| 0 | 0.51002 | 0.51072 | +0.0007 |
| 1 | 0.50208 | 0.50352 | +0.0014 |
| 2 | 0.50004 | 0.50016 | +0.0001 |
| 3 | 0.50457 | 0.50432 | -0.0003 |
| 4 | 0.50302 | 0.50400 | +0.0010 |
| 5 | 0.49952 | 0.49783 | -0.0017 |
| 6 | 0.51149 | 0.51125 | -0.0002 |
| 7 | 0.50936 | 0.50980 | +0.0004 |
| **mean** | **0.50501** | **0.50520** | **+0.00019** |

8-fold 均值 +0.00019（噪声内，5/8 折为正）。fold0(用户度量)+0.0007 是确定性提升但量级在噪声带。
Sweep A 的 3-fold +0.00076 是 fold0-2 的幸运选择。

## 最终结论
config.py 参数空间**已穷尽**。改 min_child_weight=10 是最佳方向（fold0 +0.0007）。
**0.515(+0.005) 在 config.py 调参约束下不可达**：
1. overall=recall×succ，recall 被 candidate_k=25 最优点锁死(0.859)，succ 受固定 46 特征上界限制(~0.594)。
2. 参数调 succ 上限 +0.001 量级，距 +0.006 差 6-10 倍。
3. recall↔succ 联动(candidate_k/score_weights/hard_neg)均 net 中性(已实测)。
4. README 亦指出 0.51→0.515 杠杆为 item embedding/用户画像特征(特征工程)与 xgb+lgb 融合，均超出"只调 config.py"约束。

建议：若要冲 0.515，需放开约束做特征工程(item embedding 攻非-repeat target 稠密转移泛化)或模型融合；纯 config.py 调参已到顶 ~0.511。

## 2026-07-01 22:30:24 — K_weights_rest (FEATURE SWEEP)

- baseline model_params={'n_estimators': 1400, 'lr': 0.06, 'max_depth': 5, 'subsample': 0.8, 'colsample': 0.8, 'min_child_weight': 10.0, 'reg_lambda': 1.0, 'early_stopping': 300, 'verbose_eval': 50}
- baseline features: candidate_k=25 train_candidate_k=50 hard_neg=1.0 watch_frac=0.1
- folds=[0]  baseline_mean_ndcg=0.51072  target=0.51500

| config | fold0 | ndcg_mean | Δvs_base | succ_mean | recall_mean |
|---|---|---|---|---|---|
| baseline ((baseline)) | 0.51072 | 0.51072 | +0.00000 | 0.59455 | 0.85900 |
| pop=8 (score_weights={'pop': 8.0}) | 0.50604 | 0.50604 | -0.00468 | 0.59343 | 0.85275 |
| repeat=15 (score_weights={'repeat': 15.0}) | 0.50897 | 0.50897 | -0.00175 | 0.59252 | 0.85900 |
| markov=5 (score_weights={'markov': 5.0}) | 0.50685 | 0.50685 | -0.00388 | 0.59107 | 0.85750 |
| user_cond=5 (score_weights={'user_cond': 5.0}) | 0.50760 | 0.50760 | -0.00312 | 0.59196 | 0.85750 |


## Sweep K 收尾（剩余 4 个 score_weights，在 mcw=10 之上，fold0）
- pop=8: 0.50604(-0.0047, recall 3411↓)
- repeat=15: 0.50897(-0.0018)
- markov=5: 0.50685(-0.0039)
- user_cond=5: 0.50760(-0.0031)
全部劣。7 个 score_weights 已全覆盖，确认 README"手调权重几乎无效"。

## 穷尽声明
config.py 中**每一个**参数均已实测（model_params 全部 + features 全部含 7 个 score_weights + pipeline watch_frac）。
唯一改动 min_child_weight 1.0→10.0 已写入 config.py 并经官方 run.py 验证(fold0=0.51072)。
纯 config.py 调参上限 ~0.511(fold0)，0.515 不可达（须特征工程/模型融合，超出约束）。

## 2026-07-01 22:36:20 — RECALL SCAN S1_single (candidate_k=25, fold0)

- baseline weights: {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0}
- baseline recall@ck=0.85900 recall@10(score)=0.70900

| config | recall@ck | Δ | recall@10 | Δ10 | weights |
|---|---|---|---|---|---|
| user_cond=10 | 0.85950 | +0.00050 | 0.71225 | +0.00325 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:10} |
| baseline | 0.85900 | +0.00000 | 0.70900 | +0.00000 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| collab=4 | 0.85900 | +0.00000 | 0.71275 | +0.00375 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:1.0,htarget:30.0,user_cond:15.0} |
| markov=3 | 0.85900 | +0.00000 | 0.71300 | +0.00400 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:3,htarget:30.0,user_cond:15.0} |
| repeat=20 | 0.85900 | +0.00000 | 0.70950 | +0.00050 | {pop:2.0,target_prior:0.0,repeat:20,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| collab=8 | 0.85800 | -0.00100 | 0.71125 | +0.00225 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:8,markov:1.0,htarget:30.0,user_cond:15.0} |
| pop=4 | 0.85800 | -0.00100 | 0.70750 | -0.00150 | {pop:4,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| htarget=20 | 0.85800 | -0.00100 | 0.70800 | -0.00100 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:20,user_cond:15.0} |
| markov=5 | 0.85750 | -0.00150 | 0.71325 | +0.00425 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:5,htarget:30.0,user_cond:15.0} |
| target_prior=2 | 0.85675 | -0.00225 | 0.70650 | -0.00250 | {pop:2.0,target_prior:2,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| target_prior=4 | 0.85525 | -0.00375 | 0.70675 | -0.00225 | {pop:2.0,target_prior:4,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |


## 2026-07-01 22:37:26 — RECALL SCAN S2_joint (candidate_k=25, fold0)

- baseline weights: {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0}
- baseline recall@ck=0.85900 recall@10(score)=0.70900

| config | recall@ck | Δ | recall@10 | Δ10 | weights |
|---|---|---|---|---|---|
| baseline | 0.85900 | +0.00000 | 0.70900 | +0.00000 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| collab4+markov3 | 0.85850 | -0.00050 | 0.71350 | +0.00450 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:3,htarget:30.0,user_cond:15.0} |
| collab4+markov5 | 0.85850 | -0.00050 | 0.71425 | +0.00525 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:5,htarget:30.0,user_cond:15.0} |
| collab4+markov3+uc10 | 0.85800 | -0.00100 | 0.71275 | +0.00375 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:3,htarget:30.0,user_cond:10} |
| collab4+markov3+uc10+repeat25 | 0.85800 | -0.00100 | 0.71225 | +0.00325 | {pop:2.0,target_prior:0.0,repeat:25,collab:4,markov:3,htarget:30.0,user_cond:10} |
| collab4+markov3+uc10+htarget25 | 0.85775 | -0.00125 | 0.71300 | +0.00400 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:3,htarget:25,user_cond:10} |
| collab4+markov3+uc10+rep25+ht25 | 0.85775 | -0.00125 | 0.71275 | +0.00375 | {pop:2.0,target_prior:0.0,repeat:25,collab:4,markov:3,htarget:25,user_cond:10} |
| collab6+markov4 | 0.85750 | -0.00150 | 0.71300 | +0.00400 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:6,markov:4,htarget:30.0,user_cond:15.0} |
| collab4+markov5+uc8 | 0.85600 | -0.00300 | 0.71000 | +0.00100 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:4,markov:5,htarget:30.0,user_cond:8} |


## 2026-07-01 22:39:38 — RECALL SCAN S3_push_recall25 (candidate_k=25, fold0)

- baseline weights: {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0}
- baseline recall@ck=0.85900 recall@10(score)=0.70900

| config | recall@ck | Δ | recall@10 | Δ10 | weights |
|---|---|---|---|---|---|
| repeat=20+htarget=25 | 0.85925 | +0.00025 | 0.70825 | -0.00075 | {pop:2.0,target_prior:0.0,repeat:20,collab:2.0,markov:1.0,htarget:25,user_cond:15.0} |
| htarget=25 | 0.85925 | +0.00025 | 0.70875 | -0.00025 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:25,user_cond:15.0} |
| baseline | 0.85900 | +0.00000 | 0.70900 | +0.00000 | {pop:2.0,target_prior:0.0,repeat:30.0,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| repeat=25 | 0.85900 | +0.00000 | 0.70925 | +0.00025 | {pop:2.0,target_prior:0.0,repeat:25,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| repeat=15 | 0.85900 | +0.00000 | 0.70975 | +0.00075 | {pop:2.0,target_prior:0.0,repeat:15,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| repeat=40 | 0.85900 | +0.00000 | 0.70950 | +0.00050 | {pop:2.0,target_prior:0.0,repeat:40,collab:2.0,markov:1.0,htarget:30.0,user_cond:15.0} |
| repeat=35+htarget=35 | 0.85825 | -0.00075 | 0.71075 | +0.00175 | {pop:2.0,target_prior:0.0,repeat:35,collab:2.0,markov:1.0,htarget:35,user_cond:15.0} |
| repeat=25+htarget=25+uc20+collab4+markov3 | 0.85800 | -0.00100 | 0.71025 | +0.00125 | {pop:2.0,target_prior:0.0,repeat:25,collab:4,markov:3,htarget:25,user_cond:20} |
| uc20+repeat25 | 0.85775 | -0.00125 | 0.70850 | -0.00050 | {pop:2.0,target_prior:0.0,repeat:25,collab:2.0,markov:1.0,htarget:30.0,user_cond:20} |
| repeat=20+htarget=25+uc20 | 0.85700 | -0.00200 | 0.70650 | -0.00250 | {pop:2.0,target_prior:0.0,repeat:20,collab:2.0,markov:1.0,htarget:25,user_cond:20} |


# ============================================================
# 第二阶段：放开约束后的特征工程尝试（2026-07-01）
# ============================================================

## 诊断定位 succ=0.594 瓶颈（diagnose.py, fold0）
overall ndcg@10 = recall × succ。recall@25=0.859 锁死（候选池含 target 的用户比例），
瓶颈在 succ（ranker 把 target 排进 top10 的能力）。按用户类型拆解（已召回的 3436 用户）：

- **按 is_repeat(target是否在历史)**：
  - repeat target: n=1000, succ=0.901（接近上限，几乎完美）
  - non-repeat target: n=2436, succ=0.469（**瓶颈所在**，占 71%）
- 按 recall_rank(候选分数把target排第几)：
  - rec_rank 0: succ=0.936, top10%=100%
  - rec_rank 1-4: succ=0.622, top10%=98%
  - rec_rank 5-9: succ=0.386, top10%=91%
  - rec_rank 10-24: succ=0.124, top10%=36%（候选分数就排很后，ranker 救不动）
- 按 history_len: 冷启动(0): n=1114 succ=0.489；1-5: succ=0.641；6-15: succ=0.707；>15: succ=0.664
- recall@10(model)=0.867, top1=0.351

**结论**：succ 卡在 non-repeat target（尤其 rec_rank 5-24 池）。现有协同信号(markov/htarget/itemknn)
都是基于同一 user-item 矩阵的直接共现，缺间接共现的泛化信号。target 若与历史无直接共现，难被排高。

## 特征工程尝试：item embedding via SVD
设计（core/feature_core.py:build_item_embeddings + build_features.py:_candidate_emb_features）：
- 输入：build_user_item_matrix 的 (n_user × n_item) log1p(count) 矩阵
- 方法：randomized truncated SVD（Halko），纯 numpy。**不用 scipy svds**——ARPACK 数值不稳定
  （ArpackError -8: trid eigenvalue），randomized SVD 稳定且更快。
- 产物：item embedding (dim, n_item)，单位长度归一
- 两个特征：
  - emb_sim：用户历史 item embedding 按 decay^(len-1-pos)*sqrt(count) 加权求和(归一) vs 候选点积
    （与现有 repeat/collab 加权风格一致，user-item 级）
  - emb_sim_last：最近一个历史 item embedding vs 候选点积（强近期泛化）
- 冷启动(无历史)返回 0。无泄漏（emb 仅来自当前训练折统计）。

## Sweep N 实验结果（emb_dim, fold0, 配 mcw=10）
| emb_dim | ndcg@10 | Δvs_base | succ | 说明 |
|---|---|---|---|---|
| 0(无emb,base) | 0.51066 | 0 | 0.59448 | 基线 |
| 8 | 0.50852 | -0.0021 | 0.59199 | 最差 |
| 16 | 0.51027 | -0.0004 | 0.59403 | 最接近 |
| 32 | 0.50957 | -0.0011 | 0.59321 | emb_sim进top12 gain(18.6) |
| 64 | 0.50924 | -0.0014 | 0.59283 | 维度越大越差 |

**全部维度为负**。emb_sim 虽进 top-12 gain（模型在用），但整体降分。

## 特征工程结论
item embedding 作为 ranker 特征**再次证伪**（与 README 旧结论一致，旧 emb 代码已被删）。
可能原因：
1. emb 与已有 collab(itemknn)/htarget/markov 高度共线——都基于同一 user-item 矩阵，
   emb 的增量信息少，反而引入噪声稀释强信号（htarget gain 212 仍主导）。
2. 真瓶颈 rec_rank 10-24 池（succ=0.124）的 target 在候选分数里就排很后，
   缺把它们拉进 top10 的正交信号；emb 作为排序特征不够。

**未尝试的下一步方向（如要继续冲0.515，按优先级）**：
1. emb 用于**召回**而非排序：改 _all_item_scores 加入 emb 分数（score_weights 加 emb 项），
   直接提 recall（攻击 rec_rank>10 那 ~600 个排不进候选 top 的 target）。recall 是硬乘数，
   recall@25 每提 1pp → overall +0.006，是排序特征收益的 3-6 倍。
2. 用户画像×候选类别交叉特征（廉价补充，README 列为候选改进点）。
3. xgb+lgb 分数融合（模型族不同必涨，+0.001~0.003，但违反"只用xgboost"）。

## 本次（第二阶段）最终状态
- emb 特征**无效，已回退**。代码回到会话开始状态。
- config.py 调参结论（第一阶段）仍成立：min_child_weight=10 是唯一正方向(+0.0007 fold0)，
  但 8 折均值 +0.00019 噪声内；纯 config.py 调参不可达 0.515。
- 诊断证明：0.515 需 recall 提升（候选侧/特征工程），非排序参数可及。
