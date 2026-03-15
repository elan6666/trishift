# TriShift 下游实验计划

## 1. 目标

这份计划面向当前仓库已经具备的评估与可视化能力，目标不是重复增加相似指标，而是系统补全以下三类证据：

1. 模型是否超越强 baseline，而不只是优于弱 baseline。
2. 模型是否恢复了更高层次的生物学信号，而不只是表达均值或 DEG overlap。
3. 模型是否在更困难、更有机制意义的场景下仍然稳定。

本文档优先围绕目前仓库已经有基础设施的方向组织，包括：

- `Systema` 风格 baseline 与参考系评估
- 现有 DEG / mean / var / Wasserstein 评估
- `UMAP` 与 `condition centroid` 可视化
- `Norman` 组合扰动 subgroup 评估
- 现有 `deg20` notebook 与配套分析脚本

---

## 2. 当前已完成的下游实验基础

### 2.1 已有主评估指标

当前仓库已经有以下主评估：

- expression-level 指标：
  - `pearson`
  - `nmse`
  - `deg_mean_r2`
- `Systema` 风格参考系指标：
  - `systema_corr_all_allpert`
  - `systema_corr_20de_allpert`
  - `systema_corr_all_r2`
  - `systema_corr_deg_r2`
- `scPRAM` 风格单细胞分布指标：
  - `scpram_r2_all_mean_mean`
  - `scpram_r2_all_var_mean`
  - `scpram_r2_degs_mean_mean`
  - `scpram_r2_degs_var_mean`
  - `scpram_wasserstein_all_sum`
  - `scpram_wasserstein_degs_sum`

相关实现：

- [src/trishift/_external_metrics.py](/e:/CODE/trishift/src/trishift/_external_metrics.py)
- [scripts/trishift/_core/run_dataset_core.py](/e:/CODE/trishift/scripts/trishift/_core/run_dataset_core.py)
- [docs/eval_metrics_guide_cn.md](/e:/CODE/trishift/docs/eval_metrics_guide_cn.md)

### 2.2 已有 baseline

当前仓库已经有 `Systema` baseline 生成与对齐评估：

- `nonctl-mean`
- `matching-mean`

相关实现：

- [scripts/systema/_core/baselines_core.py](/e:/CODE/trishift/scripts/systema/_core/baselines_core.py)
- [scripts/systema/norman/run_systema_baselines_norman.py](/e:/CODE/trishift/scripts/systema/norman/run_systema_baselines_norman.py)
- [scripts/systema/adamson/run_systema_baselines_adamson.py](/e:/CODE/trishift/scripts/systema/adamson/run_systema_baselines_adamson.py)

### 2.3 已有 DEG 下游评估

当前已经有两套 DEG 分析流：

1. 现有 `deg20` notebook：
   - [notebooks/deg20_downstream_experiment.ipynb](/e:/CODE/trishift/notebooks/deg20_downstream_experiment.ipynb)
   - 支持 `truth_deg_mode` 和 `pred_deg_mode` 多口径

2. eval-derived truth / pred DEG 分析：
   - 已并入现有 [deg20_experiment.py](/e:/CODE/trishift/scripts/trishift/analysis/deg20_experiment.py) 的多口径流程

### 2.4 已有嵌入与 centroid 可视化

- UMAP / mixing / 2D Wasserstein / MMD
  - [scripts/trishift/analysis/analyze_umap_eval_metrics_compare.py](/e:/CODE/trishift/scripts/trishift/analysis/analyze_umap_eval_metrics_compare.py)
  - [docs/umap_eval_metrics_guide_cn.md](/e:/CODE/trishift/docs/umap_eval_metrics_guide_cn.md)
- condition centroid / delta centroid
  - [scripts/trishift/analysis/condition_centroid_vis.py](/e:/CODE/trishift/scripts/trishift/analysis/condition_centroid_vis.py)

### 2.5 已有 Norman 组合扰动 subgroup 评估

当前 Norman 已经不是只有单扰动评估，仓库已按组合可见性分层：

- `single`
- `seen0`
- `seen1`
- `seen2`

相关实现：

- [scripts/common/split_utils.py](/e:/CODE/trishift/scripts/common/split_utils.py)
- [scripts/trishift/_core/run_dataset_core.py](/e:/CODE/trishift/scripts/trishift/_core/run_dataset_core.py)

这意味着：

- 你们已经具备组合扰动难度分层评估的基础
- 但尚未单独分析组合的非加性 interaction 本身

---

## 3. 建议新增的 6 个下游实验

## 3.1 实验一：Systema 完整基线面板

### 3.1.1 目的

回答两个问题：

1. `TriShift` 是否超越强 baseline，而不是只超越弱 baseline。
2. 模型是否真正学到了 perturbation-specific signal，而不是只利用了 systematic variation。

### 3.1.2 当前基础

已具备：

- `nonctl-mean`
- `matching-mean`
- `systema_corr_*`

但尚未完全组织成主文级基线面板。

### 3.1.3 建议补充内容

- 把以下方法放进统一结果表：
  - `Systema nonctl-mean`
  - `Systema matching-mean`
  - `TriShift random`
  - `TriShift nearest`
  - `Scouter`
  - `GEARS`
  - `GenePert`
- 补充 `centroid accuracy` 风格指标
- 对数据集的 systematic variation 强弱做排序或分层

### 3.1.4 输入与输出

输入：

- 各模型 `metrics.csv`
- `Systema` baseline 输出目录

输出建议：

- `systema_baseline_comparison.csv`
- `systema_baseline_summary.csv`
- `systema_baseline_barplot.png`
- `systema_centroid_accuracy_plot.png`

### 3.1.5 主文价值

这项实验最适合放主文，因为它直接回答：

- 你们是不是比强 baseline 更好
- gain 是否稳健

---

## 3.2 实验二：Pathway / Mechanism 恢复

### 3.2.1 目的

把现有的 DEG overlap 提升到 pathway / mechanism 层面，回答：

- 模型恢复的是否是正确的生物过程
- 即使单基因 overlap 不完美，pathway 排名是否仍然接近真实

### 3.2.2 当前基础

已具备：

- truth DEG list
- pred DEG list
- common DEG list
- `deg20` notebook
- 现有 DEG downstream 流程

### 3.2.3 建议数据库

- `Reactome`
- `MSigDB Hallmark`
- `GO Biological Process`

### 3.2.4 建议指标

- top-N enriched pathway overlap
- truth vs pred `NES` Pearson / Spearman
- pathway `Jaccard`
- pathway `hit@k`

### 3.2.5 建议图

- truth / pred pathway top barplot
- `NES` scatter
- representative condition heatmap

### 3.2.6 输入与输出

输入：

- `deg_gene_lists_long.csv`
- truth/pred DEG top-k gene list

输出建议：

- `pathway_enrichment_truth.csv`
- `pathway_enrichment_pred.csv`
- `pathway_overlap_summary.csv`
- `pathway_nes_scatter.png`

### 3.2.7 主文价值

这项实验很适合放主文或主文扩展结果，因为它比单纯 `common_degs_at_k` 更有生物学解释力。

---

## 3.3 实验三：单细胞异质性恢复

### 3.3.1 目的

回答：

- 模型是否只是学到了均值
- 还是保留了细胞层面的响应异质性

### 3.3.2 当前基础

已具备：

- `scpram_r2_*_var_*`
- `scpram_wasserstein_*`
- UMAP
- centroid 可视化

当前缺的是：

- 把异质性从压缩指标变成更直观的生物现象分析

### 3.3.3 建议定义 response score

可选两种：

1. 到 ctrl centroid 的距离：

```text
s_i = ||x_i - mu_ctrl||_2
```

2. 在真实 delta 方向上的投影：

```text
s_i = <x_i - mu_ctrl, Delta_truth_hat>
```

### 3.3.4 建议指标

- response score Wasserstein
- response score KS statistic
- high-responder fraction error
- cluster-wise response correlation

### 3.3.5 建议图

- truth vs pred violin
- truth vs pred histogram
- cluster-specific response boxplot

### 3.3.6 输入与输出

输入：

- `Truth_full`
- `Pred_full`
- `Ctrl_full`
- 可选 cell cluster label

输出建议：

- `response_score_per_cell.csv`
- `response_distribution_summary.csv`
- `response_violin.png`
- `response_histogram.png`

### 3.3.7 主文价值

这项更适合主文补充图或 supplement，因为它能解释：

- 为什么某些模型均值指标不错，但细胞分布仍有偏差

---

## 3.4 实验四：Norman 组合扰动非加性分析

### 3.4.1 目的

回答：

- 模型是否学到了组合扰动 interaction
- 而不只是把两个单扰动简单叠加

### 3.4.2 当前基础

已具备：

- Norman `single/seen0/seen1/seen2` subgroup
- 组合扰动评估流程

尚未正式做的是：

- non-additive residual 分析

### 3.4.3 核心公式

定义相对 ctrl 的效应：

```text
Delta_A  = mu_A  - mu_ctrl
Delta_B  = mu_B  - mu_ctrl
Delta_AB = mu_AB - mu_ctrl
```

定义组合扰动的非加性 residual：

```text
Delta_nonadd_AB = Delta_AB - (Delta_A + Delta_B)
```

对 truth 和 pred 分别计算：

```text
Delta_nonadd_truth_AB
Delta_nonadd_pred_AB
```

然后比较两者。

### 3.4.4 含义

- 若 residual 接近 0：
  - 说明组合效应接近简单可加
- 若 residual 明显非 0：
  - 说明存在 synergy / suppression / epistasis

因此这个实验衡量的是：

- 模型有没有学到 interaction 本身

### 3.4.5 建议指标

- non-additive Pearson
- non-additive R2
- synergy sign accuracy
- interaction strength MAE

### 3.4.6 建议图

- truth vs pred non-additive scatter
- per-gene residual heatmap
- subgroup 柱状图：`seen0/seen1/seen2`

### 3.4.7 输入与输出

输入：

- Norman 组合条件的 `Pred_full / Truth_full / Ctrl_full`
- 对应单扰动 `A+ctrl`, `B+ctrl`

输出建议：

- `norman_nonadd_metrics.csv`
- `norman_nonadd_summary.csv`
- `norman_nonadd_scatter.png`
- `norman_nonadd_by_subgroup.png`

### 3.4.8 主文价值

这项非常值得进主文，因为 Norman 的价值本来就在组合扰动泛化。  
这项实验能把“泛化到组合条件”提升成“学到组合 interaction”。

---

## 3.5 实验五：分层鲁棒性基准实验

### 3.5.1 目的

这项实验不是新模型，也不是新单一指标。  
它的目标是：

- 把你们现有的指标按任务难度切开看
- 判断模型是“整体都稳”，还是“只在简单条件上好”

### 3.5.2 为什么要做

仅看 overall mean 有两个问题：

1. 简单条件和困难条件会被平均掉
2. 审稿人无法判断模型的泛化边界

### 3.5.3 当前基础

你们已经有一部分分层基础：

- `Norman subgroup`：
  - `single`
  - `seen0`
  - `seen1`
  - `seen2`

这已经是一种 stratified benchmark。

### 3.5.4 建议新增分层维度

#### A. 按组合可见性分层

直接使用现有 `Norman subgroup`。

目的：

- 看模型在最难的 `seen0` 上是否还能保持优势

#### B. 按扰动强度分层

用真实扰动幅度分层，例如：

- `truth_ctrl_shift_norm`
- 或 truth DEG effect size mean

分成：

- weak
- medium
- strong

目的：

- 看模型在弱信号条件上是否更容易失效

#### C. 按 train-test 距离分层

可以用：

- GenePT 相似度
- embedding 最近邻距离
- `nearest` 检索距离

分成：

- near-to-train
- medium
- far-from-train

目的：

- 判断模型是不是主要依赖近邻插值

#### D. 按 DEG 难度分层

例如：

- top-k truth DEG 的平均 effect size
- DEG 数量
- DEG 稀疏程度

目的：

- 看模型在 harder biology 上是否退化更快

### 3.5.5 推荐直接复用的指标

不需要再造新指标，直接复用现有：

- `pearson`
- `nmse`
- `deg_mean_r2`
- `systema_corr_*`
- `scpram_r2_degs_var_mean`
- `common_degs_at_k`
- `centroid_dist`

### 3.5.6 推荐图

- subgroup barplot
- difficulty vs performance scatter
- model win-rate heatmap
- strata boxplot

### 3.5.7 输入与输出

输入：

- `metrics.csv`
- subgroup label
- 难度分层元数据

输出建议：

- `stratified_metrics.csv`
- `stratified_summary.csv`
- `model_winrate_by_strata.csv`
- `stratified_boxplot.png`
- `difficulty_scatter.png`

### 3.5.8 主文价值

这项非常适合主文结果组织，因为它不要求发明新方法，只要求更清楚地说明：

- 模型好在哪些条件
- 弱在哪些条件
- 相比 baseline 的 gain 在 hard split 上是否仍然存在

---

## 3.6 实验六：网络 / 因果一致性分析

### 3.6.1 目的

回答：

- 预测表达对了，是否意味着调控逻辑也对
- 对关键 regulator 的下游 target 方向是否一致

### 3.6.2 当前基础

已具备：

- perturbation identity
- truth/pred expression
- DEG list

当前缺的是：

- network-level target consistency 评估

### 3.6.3 建议做法

选取一部分有清晰 regulator-target 先验的 perturbation，检查：

- truth 中被显著推高/压低的 target
- pred 是否保持相同方向

如果有外部 GRN 或 TF-target 数据库，可进一步做：

- target recovery
- direction agreement
- enrichment consistency

### 3.6.4 建议指标

- target sign accuracy
- TF-target enrichment recovery
- regulator-target hit@k
- edge direction agreement

### 3.6.5 建议图

- regulator-target bubble plot
- target-rank comparison
- edge-sign agreement summary

### 3.6.6 输入与输出

输入：

- perturbation gene
- truth/pred differential effect
- 外部调控网络先验

输出建议：

- `network_consistency_metrics.csv`
- `target_sign_accuracy.csv`
- `tf_target_recovery.png`

### 3.6.7 主文价值

实现成本最高，但如果做出来，最适合用来提升论文层次。  
更适合放 supplement 或第二轮增强。

---

## 4. 优先级建议

## 4.1 第一优先级

最值得先做，且最容易转化为论文主文结果：

1. `Systema` 完整基线面板
2. pathway / mechanism 恢复
3. Norman 组合扰动非加性分析

## 4.2 第二优先级

更适合补充说明模型行为：

4. 单细胞异质性恢复
5. 分层鲁棒性基准实验

## 4.3 第三优先级

更偏机制拔高：

6. 网络 / 因果一致性分析

---

## 5. 建议的执行顺序

## 5.1 第一阶段：快速产出主文增量

建议先完成：

1. `Systema` 主表整合
2. pathway enrichment
3. Norman non-additive residual

原因：

- 和现有代码最容易衔接
- 最容易形成高信息密度的主文图

## 5.2 第二阶段：增强解释力

建议再完成：

4. heterogeneity case study
5. stratified robustness panel

原因：

- 能解释模型为什么在某些数据集或某些 subgroup 上更强

## 5.3 第三阶段：机制拔高

最后视时间决定是否补：

6. network / causal consistency

---

## 6. 论文组织建议

如果后续写论文，建议把下游实验组织成下面的叙事顺序：

1. 主结果：
   - overall metrics
   - Systema baseline comparison
2. 生物学有效性：
   - DEG recovery
   - pathway recovery
3. 条件级与细胞级结构：
   - centroid / UMAP
   - heterogeneity analysis
4. 组合泛化：
   - Norman subgroup
   - non-additive residual
5. 补充材料：
   - stratified robustness
   - network / causal consistency

---

## 7. 一句话版本

当前最值得补的不是再加一个类似 Pearson 的新指标，而是：

- 用 `Systema` 强 baseline 证明你们真有增益
- 用 pathway / mechanism 证明你们恢复了正确生物学
- 用 Norman non-additive residual 证明你们学到了组合 interaction

这三项最值得优先推进。
