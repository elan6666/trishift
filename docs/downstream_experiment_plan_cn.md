# TriShift 下游实验计划（Bioinformatics 投稿版）

## 1. 目标

这份计划面向当前仓库已经具备的评估与可视化能力，目标不是重复增加相似指标，而是为 `Bioinformatics` 风格的方法论文补齐最关键的三类证据：

1. 模型是否超越强 baseline，而不只是优于弱 baseline。
2. 模型是否恢复了更高层次的生物学信号，而不只是表达均值或 DEG overlap。
3. 模型是否在更困难、更有机制意义的场景下仍然稳定。

本文档优先围绕目前仓库已经有基础设施的方向组织，包括：

- `Systema` 风格 baseline 与参考系评估
- 现有 DEG / mean / var / Wasserstein 评估
- `UMAP` 与 `condition centroid` 可视化
- `Norman` 组合扰动 subgroup 评估
- 现有 `deg20` notebook 与配套分析脚本

另外，这份计划默认采用以下收缩原则：

- 不再把已经完成的 `UMAP / centroid / Systema baseline` 重新写成新增实验
- 主文优先保留 `mean_systema_corr_20de_allpert`
- 不再主推 `mean_systema_corr_all_allpert`
- 网络相关实验优先做 `network-lite`，不直接上 full GRN
- 下游实验不追求“大而全”，而是围绕主文主线做高信息密度收束

进一步地，本文默认采用如下投稿导向：

1. 主文只保留最能证明方法 advance 的 4 类下游实验。
2. 其余分析降级到 supplement 或后续版本，避免主线发散。
3. 所有下游实验都要服务于一个问题：`TriShift` 是否在真实生物任务上相较强 baseline 提供了稳健、可解释、可复现的增益。

---

## 2. 当前已完成的下游实验基础

### 2.1 已有主评估指标

当前仓库已经有以下主评估：

- expression-level 指标：
  - `pearson`
  - `nmse`
  - `deg_mean_r2`
- `Systema` 风格参考系指标：
  - `systema_corr_20de_allpert`
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

建议主文优先指标为：

- `pearson`
- `nmse`
- `deg_mean_r2`
- `systema_corr_20de_allpert`
- `systema_corr_deg_r2`
- `scpram_r2_degs_mean_mean`
- `scpram_r2_degs_var_mean`
- `scpram_wasserstein_degs_sum`
- `centroid_dist`
- `delta_cosine`

其中：

- `systema_corr_all_allpert`
- `systema_corr_all_r2`

更适合放 supplementary 或内部分析，不建议继续作为主卖点。

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

### 2.6 已完成但不再重复扩展的模块

以下内容当前应视为“已完成基础”，不再作为新的主实验方向：

- 再新增一套 `UMAP` 可视化
- 再新增一套 `centroid` 可视化
- 重新把 `Systema baseline` 当成从零开始的新实验
- 再扩更多整体表达相关性指标

---

## 3. 投稿导向的下游实验结构

结合 `scGPT` 的任务组织方式以及 `Bioinformatics` 对方法论文的常见预期，建议不要把所有下游方向并列成“都要做完”的主实验，而是分成三层：

1. 主文核心实验
2. 补充增强实验
3. 暂缓或后续工作

`scGPT` 的经验是：

- `perturbation prediction`
- `GRN inference`
- `attention-based GRN`

是分开的任务线，而不是强行塞进一个主结果包里。  
对 `TriShift` 更稳妥的做法也是一样：主文先把 perturbation prediction 的核心 biological validation 做扎实，再把机制扩展控制在 `network-lite` 范围内。

---

## 4. 主文核心的 4 个下游实验

## 4.1 实验一：Systema 完整基线面板

### 4.1.1 目的

回答两个问题：

1. `TriShift` 是否超越强 baseline，而不是只超越弱 baseline。
2. 模型是否真正学到了 perturbation-specific signal，而不是只利用了 systematic variation。

### 4.1.2 当前基础

已具备：

- `nonctl-mean`
- `matching-mean`
- `systema_corr_*`

但尚未完全组织成主文级基线面板。

### 4.1.3 建议补充内容

- 把以下方法放进统一结果表：
  - `Systema nonctl-mean`
  - `Systema matching-mean`
  - `TriShift random`
  - `TriShift nearest`
  - `Scouter`
  - `GEARS`
  - `GenePert`
- 可补充 `centroid accuracy` 风格指标，但不作为主结果唯一支撑
- 建议按 `overall + subgroup + hard split` 三个层次组织，而不是只给 overall mean

### 4.1.4 输入与输出

输入：

- 各模型 `metrics.csv`
- `Systema` baseline 输出目录

输出建议：

- `systema_baseline_comparison.csv`
- `systema_baseline_summary.csv`
- `systema_baseline_barplot.png`
- `systema_centroid_accuracy_plot.png`

### 4.1.5 主文价值

这项实验必须进主文，因为它直接回答：

- 你们是不是比强 baseline 更好
- gain 是否稳健

这也是 `Bioinformatics` 最看重的证据之一：新方法必须和现有 state-of-the-art 在真实生物数据上做直接比较，而不是只展示单模型结果。

---

## 4.2 实验二：Pathway / Mechanism 恢复

### 4.2.1 目的

把现有的 DEG overlap 提升到 pathway / mechanism 层面，回答：

- 模型恢复的是否是正确的生物过程
- 即使单基因 overlap 不完美，pathway 排名是否仍然接近真实

### 4.2.2 当前基础

已具备：

- truth DEG list
- pred DEG list
- common DEG list
- `deg20` notebook
- 现有 DEG downstream 流程

### 4.2.3 建议数据库

- `Reactome`
- `MSigDB Hallmark`
- `GO Biological Process`

### 4.2.4 建议指标

- top-N enriched pathway overlap
- truth vs pred `NES` Pearson / Spearman
- pathway `Jaccard`
- pathway `hit@k`

### 4.2.5 建议图

- truth / pred pathway top barplot
- `NES` scatter
- representative condition heatmap

### 4.2.6 输入与输出

输入：

- `deg_gene_lists_long.csv`
- truth/pred DEG top-k gene list

输出建议：

- `pathway_enrichment_truth.csv`
- `pathway_enrichment_pred.csv`
- `pathway_overlap_summary.csv`
- `pathway_nes_scatter.png`

### 4.2.7 主文价值

这项实验强烈建议进主文，因为它比单纯 `common_degs_at_k` 更有生物学解释力，也更符合 `Bioinformatics` 对 “novel method + real biological insight” 的期待。

---

## 4.3 实验三：Norman 组合扰动非加性分析

### 4.3.1 目的

- 模型是否学到了组合扰动 interaction
- 而不只是把两个单扰动简单叠加

### 4.3.2 当前基础

已具备：

- Norman `single/seen0/seen1/seen2` subgroup
- 组合扰动评估流程

尚未正式做的是：

- non-additive residual 分析

### 4.3.3 核心公式

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

### 4.3.4 含义

- 若 residual 接近 0：
  - 说明组合效应接近简单可加
- 若 residual 明显非 0：
  - 说明存在 synergy / suppression / epistasis

因此这个实验衡量的是：

- 模型有没有学到 interaction 本身

### 4.3.5 建议指标

- non-additive Pearson
- non-additive R2
- synergy sign accuracy
- interaction strength MAE

### 4.3.6 建议图

- truth vs pred non-additive scatter
- per-gene residual heatmap
- subgroup 柱状图：`seen0/seen1/seen2`

### 4.3.7 输入与输出

输入：

- Norman 组合条件的 `Pred_full / Truth_full / Ctrl_full`
- 对应单扰动 `A+ctrl`, `B+ctrl`

输出建议：

- `norman_nonadd_metrics.csv`
- `norman_nonadd_summary.csv`
- `norman_nonadd_scatter.png`
- `norman_nonadd_by_subgroup.png`

### 4.3.8 主文价值

这项非常值得进主文，因为 Norman 的价值本来就在组合扰动泛化。  
这项实验能把“泛化到组合条件”提升成“学到组合 interaction”。

---

## 4.4 实验四：分层鲁棒性基准实验

### 4.4.1 目的

这项实验不是新模型，也不是新单一指标。  
它的目标是：

- 把你们现有的指标按任务难度切开看
- 判断模型是“整体都稳”，还是“只在简单条件上好”

### 4.4.2 为什么要做

仅看 overall mean 有两个问题：

1. 简单条件和困难条件会被平均掉
2. 审稿人无法判断模型的泛化边界

### 4.4.3 当前基础

你们已经有一部分分层基础：

- `Norman subgroup`：
  - `single`
  - `seen0`
  - `seen1`
  - `seen2`

这已经是一种 stratified benchmark。

### 4.4.4 建议新增分层维度

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

### 4.4.5 推荐直接复用的指标

不需要再造新指标，直接复用现有：

- `pearson`
- `nmse`
- `deg_mean_r2`
- `systema_corr_*`
- `scpram_r2_degs_var_mean`
- `common_degs_at_k`
- `centroid_dist`

### 4.4.6 推荐图

- subgroup barplot
- difficulty vs performance scatter
- model win-rate heatmap
- strata boxplot

### 4.4.7 输入与输出

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

### 4.4.8 主文价值

这项非常适合主文结果组织，因为它不要求发明新方法，只要求更清楚地说明：

- 模型好在哪些条件
- 弱在哪些条件
- 相比 baseline 的 gain 在 hard split 上是否仍然存在

---

## 5. 补充增强实验

## 5.1 单细胞异质性恢复

### 5.1.1 定位

这项实验有价值，但建议降级到 supplement 或主文补充图，不作为四个主结果模块之一。

### 5.1.2 保留原因

- 它能解释为什么某些模型均值指标不错，但细胞分布仍有偏差
- 你们已经有 `scpram_r2_*_var_*` 和 `Wasserstein` 指标，补一个更直观的响应分布分析成本不高

### 5.1.3 建议最小实现

- 定义 `response score`
- 画 truth / pred violin 或 histogram
- 报告：
  - response score Wasserstein
  - KS statistic
  - high-responder fraction error

不建议在这部分继续扩成新的主 benchmark。

---

## 5.2 scGPT 启发的机制扩展分析

### 5.2.1 目的

回答：

- 模型是否不仅预测对表达，还恢复了更高层的机制结构
- 模型是否能帮助定位 perturbation identity、gene program 或 target-set
- 在不直接做 full GRN reconstruction 的情况下，是否能提供有说服力的 network-lite 证据

### 5.2.2 当前基础

已具备：

- perturbation identity
- truth/pred expression
- DEG list
- `Norman` 组合扰动与 subgroup 划分

额外参考来源：

- [external/scGPT-main/scgpt正式版.pdf](/e:/CODE/trishift/external/scGPT-main/scgpt正式版.pdf)
- [Tutorial_GRN.ipynb](/e:/CODE/trishift/external/scGPT-main/tutorials/Tutorial_GRN.ipynb)
- [Tutorial_Attention_GRN.ipynb](/e:/CODE/trishift/external/scGPT-main/tutorials/Tutorial_Attention_GRN.ipynb)

当前缺的是：

- reverse perturbation prediction
- gene program / target-set consistency
- 更轻量的 GRN-like 机制分析

### 5.2.3 子实验 A：reverse perturbation prediction

目的：

- 给定真实扰动后的状态
- 在候选 perturbation 集合中检索最可能的 source perturbation
- 检查正确 perturbation 的 top-1 / top-k 排名

建议做法：

- 对每个真实 test condition，使用 `Truth` 或 `Truth centroid` 作为 query
- 对所有 candidate condition，生成 `Pred` 或使用 `Pred centroid`
- 用以下相似度排序：
  - `centroid distance`
  - `delta cosine`
  - DEG / pathway similarity
- 特别在 `Norman` 上分析：
  - 真值是否能把正确组合扰动排到前列
  - 是否能区分真实组合和近似的单扰动 / 错误组合

建议指标：

- top-1 accuracy
- top-5 / top-10 hit rate
- MRR
- correct-combo rank

### 5.2.4 子实验 B：gene program / target-set consistency

目的：

- 检查模型恢复的是否是一整组生物学程序
- 而不只是少量 top DEG

建议做法：

先做两层，优先第一层：

1. truth-derived program
   - 用 truth top-k DEG、truth pathway gene set 或 truth enrichment 结果定义 gene program
   - 观察 pred 是否恢复同一 program 的方向和强度

2. external target set
   - 对已知 regulator perturbation，引入外部 TF-target 或 pathway gene set
   - 比较 truth 和 pred 对该 gene set 的 effect / enrichment / rank

说明：

- 这里的 `target-set consistency` 更接近 `scGPT` 的 network-lite 思路
- 不等于严格 full GRN reconstruction

建议指标：

- target-set effect correlation
- target-set sign consistency
- target-set enrichment recovery
- pathway / target-set hit@k

### 5.2.5 子实验 C：GRN-lite 解释分析

目的：

- 借鉴 `scGPT` 的 gene embedding / attention probing 思路
- 但避免直接做最重的 causal GRN benchmark

建议做法：

- 如果后续模型内部表示允许解释：
  - 做 regulator-specific importance ranking
  - 做 target-set overlap
  - 做 gene program clustering / enrichment
- 如果当前模型不方便做 attention 解释：
  - 只保留 gene program / target-set consistency 即可

### 5.2.6 暂不优先：full GRN / causal network reconstruction

这部分先明确降级：

- 不作为当前主文必做实验
- 更适合 supplement 或后续版本

原因：

- 需要外部先验质量足够高
- 需要额外设计 edge-level benchmark
- 当前实现成本高，且容易把主线带偏

### 5.2.7 输入与输出

输入：

- perturbation candidate 集合
- truth/pred expression 或 centroid
- truth/pred DEG list
- 外部 TF-target / pathway / program 先验（可选）

输出建议：

- `reverse_perturbation_metrics.csv`
- `reverse_perturbation_summary.csv`
- `target_set_consistency.csv`
- `target_set_enrichment_summary.csv`
- `reverse_perturbation_barplot.png`
- `target_set_overlap_plot.png`

### 5.2.8 主文价值

推荐定位：

- `reverse perturbation prediction`
  - 更适合作为 supplement 或主文增强分析
- `gene program / target-set consistency`
  - 很适合主文扩展或 supplement
- `GRN-lite`
  - 适合 supplement
- `full GRN / causal reconstruction`
  - 暂缓

这里建议明确借鉴 `scGPT` 的组织方式，而不是照搬其任务总量：

- `scGPT` 把 `perturbation`、`GRN`、`attention-based GRN` 分成独立分析线
- `TriShift` 主文中只保留与 perturbation prediction 主线直接相关的 `network-lite` 证据
- 不建议把 full GRN 作为当前投稿版本的主线工作包

---

## 6. Bioinformatics 对齐要求

为了更符合 `Bioinformatics` 风格的方法论文，这份计划还应明确满足以下要求：

1. 必须与强 baseline 做直接比较，而不是只给单模型结果。
2. 必须使用真实生物数据，并且明确 train / validation / independent test 或 multi-split 设定。
3. 不能只展示 marginal metric gain，还要有更高层次 biological validation。
4. 需要可复现性支撑：
   - 固定 split protocol
   - 多 split 汇总
   - 误差条或显著性检验
   - 代码、配置、脚本、结果重现实验流程
5. 不宜把大量相似指标堆成主文卖点；更重要的是清楚回答科学问题。

因此，主文建议采用如下最小充分实验包：

1. 强 baseline 主表
2. pathway / mechanism recovery
3. Norman non-additive interaction
4. stratified robustness

其余分析进入 supplement。

---

## 7. 优先级建议

## 7.1 第一优先级

最值得先做，且最容易转化为论文主文结果：

1. `Systema` 主表整合与强 baseline 对比
2. pathway / mechanism 恢复
3. Norman 组合扰动非加性分析
4. 分层鲁棒性基准实验

## 7.2 第二优先级

更适合补充说明模型行为：

5. 单细胞异质性恢复
6. reverse perturbation prediction

## 7.3 第三优先级

更偏机制拔高：

7. gene program / target-set consistency
8. GRN-lite 分析
9. full GRN / causal consistency（暂缓）

---

## 8. 建议的执行顺序

## 8.1 第一阶段：快速产出主文增量

建议先完成：

1. `Systema` 主表整合
2. pathway enrichment
3. Norman non-additive residual
4. stratified robustness

原因：

- 和现有代码最容易衔接
- 最容易形成高信息密度的主文图
- 最符合 `Bioinformatics` 对 methods paper 的审稿预期

## 8.2 第二阶段：增强解释力

建议再完成：

5. heterogeneity case study
6. reverse perturbation prediction

原因：

- 能解释模型为什么在某些数据集或某些 subgroup 上更强
- 适合作为 supplement 或 rebuttal 弹药

## 8.3 第三阶段：机制拔高

最后视时间决定是否补：

7. target-set consistency
8. GRN-lite
9. full GRN / causal consistency

---

## 9. 论文组织建议

如果后续写论文，建议把下游实验组织成下面的叙事顺序：

1. 主结果：
   - overall metrics
   - Systema baseline comparison
2. 生物学有效性：
   - pathway recovery
3. 组合泛化：
   - Norman subgroup
   - non-additive residual
4. 泛化边界：
   - stratified robustness
5. 补充材料：
   - DEG recovery
   - centroid / UMAP
   - heterogeneity analysis
   - stratified robustness
   - reverse perturbation prediction
   - target-set consistency / GRN-lite

---

## 10. 一句话版本

当前最值得补的不是再加一个类似 Pearson 的新指标，也不是把所有机制实验一次性做满，而是：

- 用 `Systema` 强 baseline 证明你们真有增益
- 用 pathway / mechanism 证明你们恢复了正确生物学
- 用 Norman non-additive residual 证明你们学到了组合 interaction
- 用 stratified robustness 说明 gain 在 hard case 上是否仍然成立
- 再以 reverse perturbation 或 target-set consistency 作为补充增强项

主文先把这四项做扎实，比把六到九项下游实验都铺开，更符合 `Bioinformatics` 的方法论文要求。
