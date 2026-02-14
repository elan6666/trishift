# 评估指标公式与含义详解（TriShift / Scouter 对齐）

本文针对当前仓库实际输出的指标（`metrics.csv`、`mean_pearson.txt`、`*_ablation*.csv`）逐项说明：
- 数学公式
- 公式中每个变量的意义
- 指标本身在生物任务中的意义（反映模型什么能力）
- 高低优劣与常见现象解读

## 1. 统一记号

- 条件（扰动类型）：$c$
- 基因总数：$G$
- 真实/预测细胞数：$N_t, N_p$
- 第 $i$ 个真实细胞表达：$\mathbf{x}^{(t)}_i \in \mathbb{R}^G$
- 第 $j$ 个预测细胞表达：$\mathbf{x}^{(p)}_j \in \mathbb{R}^G$
- 对照均值表达向量：$\boldsymbol{\mu}_0$
- 条件 $c$ 的 DE 基因集合：$D,\ |D|=K$
- top20 DE 集合：$D_{20},\ |D_{20}|=K_{20}$
- Systema 参考向量：$\mathbf{r}$

均值定义（按细胞平均）：

$$
\mu_t[g]=\frac{1}{N_t}\sum_{i=1}^{N_t}x_i^{(t)}[g],\qquad
\mu_p[g]=\frac{1}{N_p}\sum_{j=1}^{N_p}x_j^{(p)}[g]
$$

MSE 定义：

$$
\operatorname{MSE}(\mathbf{a},\mathbf{b})=\frac{1}{m}\sum_{u=1}^{m}(a_u-b_u)^2
$$

Pearson 定义：

$$
\rho(\mathbf{a},\mathbf{b})=
\frac{\sum_{u=1}^{m}(a_u-\bar a)(b_u-\bar b)}
{\sqrt{\left(\sum_{u=1}^{m}(a_u-\bar a)^2\right)\left(\sum_{u=1}^{m}(b_u-\bar b)^2\right)}}
$$

其中

$$
\bar a=\frac{1}{m}\sum_{u=1}^{m}a_u,\qquad
\bar b=\frac{1}{m}\sum_{u=1}^{m}b_u
$$

scPRAM 风格 $R^2$ 定义（注意是相关系数平方）：

$$
R^2(\mathbf{a},\mathbf{b})=\rho(\mathbf{a},\mathbf{b})^2
$$

## 2. `metrics.csv` 单条件指标

### 2.1 `nmse`

公式：

$$
\text{Num}=\frac{1}{K}\sum_{g\in D}\left(\mu_t[g]-\mu_p[g]\right)^2
$$

$$
\text{Den}=\frac{1}{K}\sum_{g\in D}\left(\mu_t[g]-\mu_0[g]\right)^2
$$

$$
\mathrm{nMSE}=
\begin{cases}
\dfrac{\text{Num}}{\text{Den}}, & \text{Den}>0 \\
\mathrm{NaN}, & \text{Den}=0
\end{cases}
$$

变量说明：
- $\mu_t[g]$：真实扰动在基因 $g$ 上的均值表达。
- $\mu_p[g]$：预测扰动在基因 $g$ 上的均值表达。
- $\mu_0[g]$：对照均值表达。
- $D$：该条件的 DE 基因集合（评估重点基因）。
- Num：预测误差；Den：该条件相对对照的“信号强度”。

指标意义（反映模型什么）：
- 反映模型对 DE 基因“幅值恢复”的能力（绝对量纲误差，经条件难度归一化）。
- 更关注“预测值离真实值有多远”。

高低解释：
- 越小越好。
- 若 $<1$，通常表示优于“直接输出对照均值”这类弱基线。

常见现象：
- `pearson` 高但 `nmse` 高：方向学对了，但幅值（强弱）没校准好。

---

### 2.2 `pearson`

先定义 DE 上的扰动增量：

$$
a_g=\mu_t[g]-\mu_0[g],\qquad b_g=\mu_p[g]-\mu_0[g],\quad g\in D
$$

$$
\mathrm{Pearson}_{\mathrm{DE}}=\rho(\mathbf{a},\mathbf{b})
$$

变量说明：
- $a_g$：真实扰动相对对照的变化量。
- $b_g$：预测扰动相对对照的变化量。
- $\rho$：比较两个变化向量的线性一致性（方向与相对排序）。

指标意义（反映模型什么）：
- 反映模型是否学到“扰动效应方向”和“基因响应排序”。
- 对整体缩放误差不敏感（即幅值偏大/偏小时仍可能较高）。

高低解释：
- 越大越好，理论范围 $[-1,1]$（通常关注接近 1）。

常见现象：
- Pearson 高 + nMSE 差：说明方向一致，但幅值不准。
- Pearson 低 + nMSE 还行：可能均值接近，但基因排序/方向错位。

---

### 2.3 `systema_corr_all_allpert`

先定义相对参考向量的增量：

$$
\Delta_t[g]=\mu_t[g]-r[g],\qquad \Delta_p[g]=\mu_p[g]-r[g]
$$

$$
\mathrm{Sys}_{\mathrm{all}}=\rho\!\left(\Delta_t[1{:}G],\;\Delta_p[1{:}G]\right)
$$

对应列名：`systema_corr_all_allpert`

变量说明：
- $\mathbf{r}$：Systema 风格参考向量（代码中由 perturbation centroid 方案构建）。
- $\Delta_t,\Delta_p$：真实/预测相对参考的全基因变化。

指标意义（反映模型什么）：
- 反映模型在“全基因空间”上重建扰动方向结构的能力。
- 比只看 DE 更全面，能观察是否过度聚焦少数基因而忽略整体转录组模式。

高低解释：
- 越大越好。

---

### 2.4 `systema_corr_20de_allpert`

$$
\mathrm{Sys}_{20\mathrm{DE}}=\rho\!\left(\Delta_t[D_{20}],\;\Delta_p[D_{20}]\right)
$$

对应列名：`systema_corr_20de_allpert`

变量说明：
- $D_{20}$：top-20 DE 基因索引集合。
- 其他同上。

指标意义（反映模型什么）：
- 反映模型在“最显著扰动基因”上的方向一致性能力。
- 相比全基因相关，更强调主要生物信号是否被抓住。

高低解释：
- 越大越好。

---

### 2.5 `scpram_r2_all_mean_mean` 与 `scpram_r2_all_mean_std`

重复采样 $T=100$ 次（采样比例 0.8）。第 $s$ 次：

$$
m_t^{(s)}[g]=\frac{1}{n_t^{(s)}}\sum_{i\in I_t^{(s)}}x_i^{(t)}[g],\qquad
m_p^{(s)}[g]=\frac{1}{n_p^{(s)}}\sum_{j\in I_p^{(s)}}x_j^{(p)}[g]
$$

$$
r_{\mu,\mathrm{all}}^{(s)}=R^2\!\left(m_t^{(s)}[1{:}G],\,m_p^{(s)}[1{:}G]\right)
$$

$$
\overline{r}_{\mu,\mathrm{all}}=\frac{1}{T}\sum_{s=1}^{T}r_{\mu,\mathrm{all}}^{(s)}
$$

$$
\sigma_{r_{\mu,\mathrm{all}}}=
\sqrt{\frac{1}{T-1}\sum_{s=1}^{T}\left(r_{\mu,\mathrm{all}}^{(s)}-\overline{r}_{\mu,\mathrm{all}}\right)^2}
$$

对应列名：
- `scpram_r2_all_mean_mean` = $\overline{r}_{\mu,\mathrm{all}}$
- `scpram_r2_all_mean_std` = $\sigma_{r_{\mu,\mathrm{all}}}$

变量说明：
- $I_t^{(s)}, I_p^{(s)}$：第 $s$ 次重采样选到的真实/预测细胞索引集合。
- $m_t^{(s)},m_p^{(s)}$：该次采样下的全基因均值向量。
- $R^2$：相关平方，衡量均值结构一致性。

指标意义（反映模型什么）：
- `mean_mean`：模型是否重建了“全基因平均表达结构”。
- `mean_std`：该能力在随机采样下是否稳定。

高低解释：
- `*_mean` 越大越好，`*_std` 越小越好。

---

### 2.6 `scpram_r2_all_var_mean` 与 `scpram_r2_all_var_std`

第 $s$ 次采样下，按基因计算方差（样本方差）：

$$
v_t^{(s)}[g]=\frac{1}{n_t^{(s)}-1}\sum_{i\in I_t^{(s)}}\left(x_i^{(t)}[g]-m_t^{(s)}[g]\right)^2
$$

$$
v_p^{(s)}[g]=\frac{1}{n_p^{(s)}-1}\sum_{j\in I_p^{(s)}}\left(x_j^{(p)}[g]-m_p^{(s)}[g]\right)^2
$$

$$
r_{v,\mathrm{all}}^{(s)}=R^2\!\left(v_t^{(s)}[1{:}G],\,v_p^{(s)}[1{:}G]\right)
$$

对 $\{r_{v,\mathrm{all}}^{(s)}\}_{s=1}^{T}$ 取均值与标准差，对应：
- `scpram_r2_all_var_mean`
- `scpram_r2_all_var_std`

变量说明：
- $v_t^{(s)}[g], v_p^{(s)}[g]$：真实/预测在基因 $g$ 上的细胞间离散程度。

指标意义（反映模型什么）：
- 反映模型对“细胞异质性结构（方差模式）”的重建能力，而不仅是均值。

高低解释：
- `*_mean` 越大越好，`*_std` 越小越好。

---

### 2.7 `scpram_r2_degs_mean_mean` 与 `scpram_r2_degs_mean_std`

$$
r_{\mu,\mathrm{de}}^{(s)}=R^2\!\left(m_t^{(s)}[D],\,m_p^{(s)}[D]\right)
$$

对 $\{r_{\mu,\mathrm{de}}^{(s)}\}$ 取均值和标准差，对应：
- `scpram_r2_degs_mean_mean`
- `scpram_r2_degs_mean_std`

变量说明：
- 在 2.5 基础上，只在 DE 子空间上计算。

指标意义（反映模型什么）：
- 反映模型在“关键响应基因”上的均值重建能力。

高低解释：
- `*_mean` 越大越好，`*_std` 越小越好。

---

### 2.8 `scpram_r2_degs_var_mean` 与 `scpram_r2_degs_var_std`

$$
r_{v,\mathrm{de}}^{(s)}=R^2\!\left(v_t^{(s)}[D],\,v_p^{(s)}[D]\right)
$$

对 $\{r_{v,\mathrm{de}}^{(s)}\}$ 取均值和标准差，对应：
- `scpram_r2_degs_var_mean`
- `scpram_r2_degs_var_std`

变量说明：
- 在 2.6 基础上，只在 DE 子空间上计算方差结构相关。

指标意义（反映模型什么）：
- 反映模型对“关键响应基因异质性”的重建能力。

高低解释：
- `*_mean` 越大越好，`*_std` 越小越好。

---

### 2.9 `scpram_wasserstein_all_sum`

每个基因 $g$ 上，真实/预测经验分布分别记为 $Q_g,P_g$，其 Wasserstein-1 距离：

$$
W_g=\int_{-\infty}^{+\infty}\left|F_{P_g}(z)-F_{Q_g}(z)\right|\,dz
$$

全基因求和：

$$
W_{\mathrm{all}}=\sum_{g=1}^{G}W_g
$$

对应列名：`scpram_wasserstein_all_sum`

变量说明：
- $F_{P_g},F_{Q_g}$：预测/真实在基因 $g$ 上的经验 CDF。
- $W_g$：该基因分布形状差异（不是只看均值方差）。

指标意义（反映模型什么）：
- 反映模型在“分布层面”的重建能力（多峰、尾部、偏态等）。

高低解释：
- 越小越好。

---

### 2.10 `scpram_wasserstein_degs_sum`

$$
W_{\mathrm{de}}=\sum_{g\in D}W_g
$$

对应列名：`scpram_wasserstein_degs_sum`

变量说明：
- 仅在 DE 基因上累积 Wasserstein 距离。

指标意义（反映模型什么）：
- 反映模型在“关键扰动基因”上的分布拟合能力。

高低解释：
- 越小越好。

---

### 2.11 `split_id` / `n_ensemble`

- `split_id`：当前数据划分编号（元信息，不是性能好坏指标）。
- `n_ensemble`：评估时每个条件采样次数（元信息，会影响方差稳定性）。

## 3. 聚合指标（多条件）

### 3.1 `mean_*`

对某指标 $m_c$，在非 NaN 条件集合 $\mathcal{C}_v$ 上：

$$
\operatorname{MeanMetric}=\frac{1}{|\mathcal{C}_v|}\sum_{c\in\mathcal{C}_v}m_c
$$

变量说明：
- $m_c$：条件 $c$ 的单条件指标值。
- $\mathcal{C}_v$：该指标可计算的条件集合。

指标意义（反映模型什么）：
- 反映模型“跨条件平均性能”。
- 不直接反映稳定性；稳定性需配合分布/方差看。

---

### 3.2 `*_delta_vs_scouter`

$$
\Delta_{\text{vs-scouter}}=m_{\text{trishift}}-m_{\text{scouter}}
$$

变量说明：
- $m_{\text{trishift}}$：TriShift 在同指标上的聚合值。
- $m_{\text{scouter}}$：Scouter 在同指标上的聚合值。

指标意义（反映模型什么）：
- 反映相对基线（Scouter）的净提升或退化幅度。

高低解释：
- 对“大为优”指标（Pearson/Systema/$R^2$）：$\Delta>0$ 更好。
- 对“小为优”指标（nMSE/Wasserstein）：$\Delta<0$ 更好。

## 4. 实战解读模板（建议）

- 幅值能力：看 `nmse`（主） + `scpram_wasserstein_*`（分布补充）。
- 方向能力：看 `pearson` + `systema_corr_*`。
- 异质性能力：看 `scpram_r2_*_var_*`。
- 稳定性：看所有 `*_std`。

一句话模板：
- “模型在方向上更好，但幅值校准不足”：`pearson/systema` 上升，`nmse/wasserstein` 未同步下降。
- “模型均值对齐但异质性不足”：`r2_*_mean` 高、`r2_*_var` 低。

