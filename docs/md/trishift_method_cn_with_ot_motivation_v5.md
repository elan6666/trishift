# 三元参考条件化状态转移模型（Tripartite Reference-Conditioned Shift Model, TriShift）

## 3 方法

### 3.1 问题表述与模型概览

设基因表达空间为 $\mathcal{X}\subset\mathbb{R}^{G}$，其中 $G$ 为基因维度。对于每个扰动条件 $c\in\mathcal{C}$，观测控制细胞集合 $\{x_j^{(0)}\}_{j=1}^{N_0}$ 与扰动细胞集合 $\{x_i^{(p)}\}_{i=1}^{N_p}$。TriShift 不将任务写成从扰动标签到扰动后表达的直接映射，而将其写成参考条件化状态转移问题：给定一个合适的控制参考状态，预测扰动如何推动该状态发生迁移，并由此生成扰动后的表达。

本文主文默认实例采用当前 Adamson/Norman 主配置的共同设置：OT 匹配、sum pooling 的 GenePT 条件表示、transformer-based shift 模块、`full` generator 输入模式以及 residual output head。模型预测写为

$$
\hat{x}^{(p)} = x^{(0)} + F\bigl(z_{\mathrm{state}}^{(0)}, c, h\bigr),
$$

其中 $z_{\mathrm{state}}^{(0)}$ 表示控制状态表示，$c$ 表示扰动条件的 GenePT 语义向量，$h$ 表示条件化位移表征，$F(\cdot)$ 为表达生成器。这个写法对应一条明确的信息流：先确定参考状态，再建模位移，最后恢复表达。

### 3.2 控制状态表示与 GenePT 条件表示

TriShift 首先使用去噪变分自编码器学习单细胞表达的潜空间。给定输入表达向量 $x$，模型先注入高斯噪声

$$
\tilde{x}=x+\epsilon,\qquad \epsilon\sim\mathcal{N}(0,\sigma_n^2 I),
$$

随后由编码器参数化近似后验分布

$$
q_{\phi}(z\mid \tilde{x})=\mathcal{N}\bigl(\mu_{\phi}(\tilde{x}),\mathrm{diag}(\exp(\log \sigma_\phi^2(\tilde{x})))\bigr).
$$

Stage 1 的基本目标为

$$
\mathcal{L}_{\mathrm{vae}}
=
\frac{1}{B}\sum_{b=1}^{B}
\left[
\frac{1}{2}\lVert x_b-\hat{x}_b\rVert_2^2
+
\frac{1}{2}\beta\lambda_{\mathrm{kl}}
\mathrm{KL}\bigl(q_{\phi}(z\mid\tilde{x}_b)\Vert \mathcal{N}(0,I)\bigr)
\right].
$$

按照主文默认配置，Stage 1 还对负预测表达加入惩罚

$$
\mathcal{L}_{\mathrm{neg}}^{\mathrm{stage1}}
=
\operatorname{mean}\bigl(\operatorname{ReLU}(-\hat{x})\bigr),
$$

从而得到

$$
\mathcal{L}_{\mathrm{stage1}}
=
\mathcal{L}_{\mathrm{vae}}
+
\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}^{\mathrm{stage1}}.
$$

这一阶段输出两类后续表示。其一是潜均值

$$
z_{\mu}=\mu_{\phi}(x)\in\mathbb{R}^{d_z},
$$

主文默认配置中 $d_z=32$。这一表示主要服务于潜空间匹配与位移建模。其二是控制状态表示

$$
z_{\mathrm{state}} = f_{\mathrm{enc}}(x^{(0)})\in\mathbb{R}^{d_s},
$$

其中 $d_s=64$。这一表示直接进入表达生成器，用于保留控制细胞的基础状态背景。

扰动条件则由 GenePT 表示。与把扰动写成离散标签不同，TriShift 使用离线加载且训练中固定不更新的基因语义嵌入矩阵，并为 `ctrl` 额外添加一个零向量行。对于扰动条件 $c$，记其包含的基因集合为 $\mathcal{G}(c)$，每个基因 $g$ 对应一个固定的 GenePT 向量 $e_g\in\mathbb{R}^{d_c}$。默认配置下，多基因扰动采用 sum pooling：

$$
c=\sum_{g\in\mathcal{G}(c)} e_g.
$$

因此，单基因扰动直接使用对应基因的 GenePT 向量，多基因扰动则使用各基因向量之和作为条件表示。与 Scouter 相比，TriShift 同样把 GenePT 作为固定条件嵌入，但并不让该嵌入直接与控制表达拼接后立刻预测表达，而是先用于参考条件化位移建模。

### 3.3 在潜空间中用最优传输构造参考控制原型

单细胞测序的破坏性决定了控制细胞与扰动细胞通常不是逐细胞配对观测；我们因此不假设真实配对已知，而是假设相应细胞在潜空间中保持局部邻近关系，并据此用最优传输恢复控制分布与扰动分布之间的软对应结构。

给定控制潜向量集合 $\{z_i^{(0)}\}_{i=1}^{n_0}$ 与扰动潜向量集合 $\{z_j^{(p)}\}_{j=1}^{n_p}$，定义代价矩阵

$$
C_{ij}=\lVert z_i^{(0)}-z_j^{(p)}\rVert_2.
$$

默认配置采用带熵正则项的最优传输

$$
P^{\star}
=
\arg\min_{P\in\Pi(a,b)}
\left[
\langle P,C\rangle-\varepsilon H(P)
\right],
$$

其中 $\Pi(a,b)=\{P\ge 0\mid P\mathbf{1}=a,\;P^\top\mathbf{1}=b\}$，$a$ 和 $b$ 分别表示控制池与扰动池的边缘分布。实践中，这一问题通过 Sinkhorn 迭代求解。OT 在这里的作用不是直接输出预测表达，而是在潜空间中为每个扰动样本恢复更合理的参考控制结构。

在主文默认实例中，TriShift 根据 OT 耦合矩阵为每个扰动样本构造一个 top-$k$ 控制参考池，其中 $k$ 约为控制池规模的 $1\%$；在当前 Adamson 配置下对应 `k=300`。进一步地，模型用耦合权重构造控制原型

$$
\tilde{z}^{(0)}=\sum_{j\in\mathcal{M}} w_j z_j^{(0)}.
$$

这一控制原型不是最终预测，而是后续位移建模的参考状态。

### 3.4 条件化位移建模与表达生成

在获得参考控制原型和 GenePT 条件表示之后，TriShift 学习扰动如何推动参考状态发生迁移。按照主文默认配置，Stage 2 以控制侧潜均值和扰动语义为输入，通过单层 Transformer block 输出一个紧凑的位移表征

$$
h=f_{\mathrm{shift}}(z_{\mu}^{(0)}, c),\qquad h\in\mathbb{R}^{8}.
$$

这里的 $h$ 不是显式的潜空间终点，而是对“参考状态在当前扰动条件下应当朝哪个方向变化”的紧凑概括。由于默认配置中 `predict_delta=false`，主文默认实例不显式预测 $\Delta z$，而是将 $h$ 视为供生成器使用的 shift representation。

表达生成阶段采用 `full` 输入模式，将 GenePT 条件表示、控制状态表示和位移表征拼接为

$$
g_{\mathrm{in}} = [c;\, z_{\mathrm{state}};\, h].
$$

生成器输出残差项

$$
\Delta \hat{x}^{(p)} = f_{\mathrm{gen}}(g_{\mathrm{in}}),
$$

最终预测写为

$$
\hat{x}^{(p)} = x^{(0)} + \Delta \hat{x}^{(p)}.
$$

残差式写法与任务本身是一致的。模型需要恢复的不是一个脱离控制状态的全新表达谱，而是扰动相对于参考控制状态造成的变化量。

### 3.5 训练目标与推理流程

TriShift 的主训练目标聚焦于表达恢复。Stage 1 完成潜空间学习后，Stage 2 和 Stage 3 以 joint 模式联合训练。由于当前默认实例不显式预测潜空间终点位移，联合训练阶段不再额外引入 latent supervision，而是直接用表达恢复目标优化整个参考条件化链路。

对于表达预测，模型在每个批次中对有效评估基因集合 $S_r$ 计算加权损失。记 $\omega_{r,g}$ 为基因权重，$\gamma$ 为 autofocus 指数，$\lambda_{\mathrm{dir}}^x$ 为方向损失权重，则默认表达损失为

$$
\mathcal{L}_{\mathrm{expr}}
=
\frac{1}{|R|}
\sum_{r\in R}
\frac{1}{|B_r||S_r|}
\sum_{i\in B_r}
\sum_{g\in S_r}
\omega_{r,g}
\left(
|x_{ig}^{(p)}-\hat{x}_{ig}|^{2+\gamma}
+
\lambda_{\mathrm{dir}}^x
\bigl[
\mathrm{sign}(x_{ig}^{(p)}-x_{ig}^{(0)})
-
\mathrm{sign}(\hat{x}_{ig}-x_{ig}^{(0)})
\bigr]^2
\right).
$$

默认配置下，生成器输出同样带有负值惩罚。若记

$$
\mathcal{L}_{\mathrm{neg}}^{\mathrm{expr}}
=
\operatorname{mean}\bigl(\operatorname{ReLU}(-\hat{x}^{(p)})\bigr),
$$

则联合训练阶段实际优化目标为

$$
\mathcal{L}_{\mathrm{joint}}
=
\mathcal{L}_{\mathrm{expr}}
+
\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}^{\mathrm{expr}}.
$$

其中主文默认配置中 $\gamma=0$，$\lambda_{\mathrm{dir}}^x=0.05$，$\lambda_{\mathrm{neg}}=0.5$。推理阶段的流程与训练期保持一致：模型先为目标扰动构造 GenePT 条件表示，再在训练控制池中检索参考控制集合，并据此得到控制原型；随后 Stage 2 输出条件化位移表征，Stage 3 生成相对于控制表达的残差项，最终得到扰动后的表达预测。

### 3.6 Metrics of prediction performance

我们采用一组互补指标来衡量预测性能，并在正文中按 figure 顺序使用它们。

#### 3.6.1 Pearson and nMSE

设真实扰动均值、预测扰动均值和控制均值分别为 $\mu_t$、$\mu_p$ 与 $\mu_0$，差异表达基因集合记为 $D$。在 DE 基因上定义真实与预测扰动增量

$$
a_g=\mu_t[g]-\mu_0[g],\qquad b_g=\mu_p[g]-\mu_0[g],\qquad g\in D.
$$

Pearson 相关定义为

$$
\mathrm{Pearson}_{\mathrm{DE}}=\rho(\mathbf{a},\mathbf{b}),
$$

用于衡量模型是否恢复了 DE 基因上的扰动方向和相对排序。nMSE 定义为

$$
\mathrm{nMSE}
=
\frac{\frac{1}{|D|}\sum_{g\in D}(\mu_t[g]-\mu_p[g])^2}
{\frac{1}{|D|}\sum_{g\in D}(\mu_t[g]-\mu_0[g])^2},
$$

用于衡量模型相对于“直接输出控制均值”这一弱基线的幅值恢复能力。Pearson 越高越好，nMSE 越低越好。Fig. 2 主要使用这两个指标。

#### 3.6.2 Systema Pearson and residualized Systema Pearson

为检验模型是否真正恢复了 perturbation-specific signal，而不仅仅拟合了公共系统性偏移，我们进一步采用 Systema 风格指标。与普通控制差分不同，Systema 指标比较的是相对于 perturbed reference 的位移。当前主文主要使用 top-20 DE 版本，即仓库输出中的 `systema_corr_20de_allpert`，本文简称 `Systema Pearson`。

在机制分析中，我们进一步定义 `Residualized Systema Pearson`。记数据集级 generic shift 轴为

$$
g = \mu_{\mathrm{pert,train}}-\mu_{\mathrm{ctrl,train}},
$$

并对真实与预测 shift 去除沿着 $g$ 的投影：

$$
\Delta^\perp = \Delta - \mathrm{Proj}_g(\Delta).
$$

随后在 $\Delta_t^\perp$ 与 $\Delta_p^\perp$ 上计算 Pearson 相关。该指标越高，说明模型在扣除 generic shift 后仍能恢复更多 perturbation-specific residual signal。Fig. 3c 和 Fig. 3e 使用这一指标。

#### 3.6.3 Centroid accuracy

Centroid accuracy 衡量预测后的扰动 centroid 是否更接近正确 perturbation 的真实 centroid，而不是仅接近某个平均扰动态。对每个测试条件，我们计算预测 centroid 到所有真实 perturbation centroids 的距离，并据此得到正确 perturbation 的相对排名。分数越高，说明模型越能恢复正确的 perturbation identity。Fig. 3d 使用这一指标。

#### 3.6.4 Distribution-level metrics

为了评估模型是否恢复了条件级分布结构，我们保留 scPRAM 兼容指标：

- `scPRAM mean R^2`：衡量预测分布在均值层面与真实分布的一致性。
- `scPRAM var R^2`：衡量预测分布在方差层面的恢复情况。
- `scPRAM Wasserstein`：衡量真实分布与预测分布之间的距离，越低越好。

这些指标强调的是群体结构与异质性恢复，而不是单一均值向量的接近程度。主文第 6.3 节与 Fig. S3 主要围绕这一组指标展开。

#### 3.6.5 Pathway-level metric

在 pathway-level biological recovery 分析中，我们使用 `Top pathway overlap` 作为主文 summary 指标。对于每个条件，分别对真实 DEG 列表和预测 DEG 列表做通路富集，然后统计两者 top-10 pathways 的重叠数。该指标越高，说明模型恢复出的 pathway hierarchy 与真实 biology 越一致。Fig. 5a 使用该指标，Fig. 5b 则给出代表性 pathway case。
