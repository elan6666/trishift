# Systema reference 对比（train vs test）说明（Adamson）

生成时间（北京时间）：2026-02-18 12:26:54

这份文档用于长期查阅，解释 `systema_ref_compare.py` 生成的各类 CSV/图是什么，以及为什么要对比 `O_pert(train)` 与 `O_pert(test)`。

## 背景
Systema 的 PearsonΔ 属于 **reference-sensitive** 指标：先把真实/预测表达都减去同一个 reference，再在基因维度上算 Pearson 相关。

在 Systema 论文里，推荐的 “perturbed centroid reference” 是用 **train 扰动条件**的 centroid 做等权平均得到的 `O_pert(train)`（避免用 test 真值构造 reference 产生 leakage）。

我们这里额外算 `O_pert(test)` 只是为了做 **oracle 诊断**：它能告诉你“如果 reference 直接对齐到 test 的平均扰动效应，PearsonΔ 会变化多少”。

## 核心定义（不依赖 LaTeX 渲染）
记某个扰动条件为 `X`，真实表达 centroid（全基因）为：
- `O(X) = mean_{cells in condition X}( x_cell )`

对某个 split：
- `P_train`：train split 里的 perturbation 条件集合（不含 `ctrl`）
- `P_test`：test split 里的 perturbation 条件集合（不含 `ctrl`）

Systema 的 perturbed centroid reference（按 perturbation 等权平均，不按细胞数加权）：
- `O_pert(train) = mean_{Y in P_train}( O(Y) )`
- `O_pert(test)  = mean_{Y in P_test}(  O(Y) )`  (oracle/诊断)

对每个 test 条件 `X`，在该条件的 DE 基因子空间（由导出 pkl 的 `DE_idx(X)` 定义）上：
- `true_de = mean_rows(Truth)`  (shape = [n_de])
- `pred_de = mean_rows(Pred)`    (shape = [n_de])
- `ref_train_de = O_pert(train)[DE_idx(X)]`
- `ref_test_de  = O_pert(test)[DE_idx(X)]`

两种 Pearson（范围 [-1, 1]）：
- `pearson_ref_train = corr(true_de - ref_train_de, pred_de - ref_train_de)`
- `pearson_ref_test  = corr(true_de - ref_test_de,  pred_de - ref_test_de)`
- `delta_test_minus_train = pearson_ref_test - pearson_ref_train`

说明：
- 这里的 `corr(a, b)` 是对基因维度计算 Pearson 相关。
- 如果向量方差为 0，Pearson 会是 NaN（表示这个条件在该子空间上不适合用 Pearson 衡量）。

## 生成的文件都是什么，怎么看

### 1) `systema_ref_compare_de20_long.csv`（最明细）
每一行 = 1 个 `run` + 1 个 `split_id` + 1 个 `condition`。

关键列：
- `pearson_ref_train`：正式口径（train reference）
- `pearson_ref_test`：oracle 诊断口径（test reference）
- `delta_test_minus_train`：reference 改变带来的分数变化

怎么用：
1. 比较方法优劣：优先看 `pearson_ref_train`。
2. 找 reference 敏感条件：按 `abs(delta_test_minus_train)` 排序。

### 2) `systema_ref_compare_de20_summary.csv`（split 级汇总）
每一行 = 1 个 `run` + 1 个 `split_id`，对该 split 下所有 conditions 做统计。

关键列：
- `mean_ref_train / std_ref_train`
- `mean_ref_test / std_ref_test`
- `mean_delta / std_delta`
- `n_nan_train / n_nan_test`

怎么用：
- 看单个 split 的整体表现与稳定性（条件间的方差）。

### 3) `systema_ref_compare_de20_over_splits.csv`（run 级汇总，推荐主表）
每一行 = 1 个 `run`，把多个 split 的结果合并成一个。

它是：先用 summary 得到每个 split 的 `mean_ref_*`，再对 splits **等权平均**：
- `mean_ref_train_over_splits`
- `mean_ref_test_over_splits`
- `mean_delta_over_splits`

同时给出 split 间波动（均值的方差）：
- `std_split_mean_ref_train`
- `std_split_mean_ref_test`
- `std_split_mean_delta`

怎么用：
- 给 15 个消融实验排总榜：按 `mean_ref_train_over_splits` 排序。
- 看是否对 reference 敏感：看 `mean_delta_over_splits` 的绝对值。
- 看是否对 split 敏感：看 `std_split_mean_*` 是否很大。

### 4) `systema_ref_compare_de20_over_splits_weighted_by_rows.csv`（对照口径）
每一行 = 1 个 `run`，直接从 `long.csv` 把所有行混在一起求均值（相当于按“行数/条件数”加权）。

怎么用：
- 当每个 split 的 condition 数相同、splits 都齐全时，它应和 over_splits 很接近。
- 当某些 split 缺失或 condition 数不一致时，它会更偏向“行数更多”的 split。

## 为什么不把 `O_pert` 用 test 来算（正式指标）
用 `O_pert(test)` 会用到 test 真值的全局统计量（所有 test 条件的 centroid），属于 oracle/peeking，会让 reference-based 指标更“容易”，不适合用来做正式对比。因此 `pearson_ref_test` 只作为诊断分析保留。

