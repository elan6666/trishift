# 三元参考条件化状态转移模型（Tripartite Reference-Conditioned Shift Model, TriShift）

## 6 Results

### 6.1 TriShift establishes a stronger and more stable benchmark across single-perturbation datasets

我们首先回答一个最基础的问题：TriShift 是否在标准单扰动 benchmark 上稳定优于现有方法。按照主文图顺序，这一结论由 Fig. 2 支撑。Fig. 2a 和 Fig. 2b 分别比较 task-specific baseline 组与 semantic/foundation 组上的 Pearson 相关。整体上，TriShift 在 Adamson 和 Dixit 上取得最高的 Pearson，在 Norman 上则与最强 Pearson 基线保持接近，而不是明显落后。这一结果表明，TriShift 的收益不是依赖某个单一数据集或单一模型族成立，而是在三套主文数据中都保持了稳定竞争力。

但 Pearson 只说明了方向和排序上的一致性，并不能完全反映幅值恢复的质量。因此，我们进一步在 Fig. 2c 和 Fig. 2d 中比较 nMSE。与 Pearson 相比，nMSE 更直接地衡量模型相对于控制均值基线的误差缩减能力。在这一指标上，TriShift 在 Adamson、Norman 和 Dixit 三个数据集上都取得最低值，说明它不仅能恢复扰动方向，也能更稳健地校准扰动幅值。换言之，TriShift 在主 benchmark 上的优势不是单纯的相关性优势，而是更全面的预测优势。

Fig. 2e 则提供了一个代表性单扰动案例，用来把跨数据集的统计结论落到具体的基因表达变化上。这里的重点不是再引入一个新主张，而是让读者直观看到：TriShift 的预测在关键基因上更接近真实扰动相对于控制的表达变化。按照 GEARS 和 Scouter 的结果写法，这样的 figure-level case 不承担独立结论，而是作为前面 benchmark 结论的直观收束。由此，Fig. 2 支撑的第一条结论可以明确表述为：TriShift 在标准单扰动预测任务上建立了更强且更稳定的整体基准。

### 6.2 TriShift gains are not explained by systematic variation alone

标准 benchmark 的提升并不自动意味着模型恢复了真正的 perturbation-specific response。正如 Systema 所指出的，模型可能仅仅学到了扰动样本与控制样本之间稳定存在的系统性偏移，而在常规相关性指标上获得偏乐观的得分。因此，我们将 Fig. 3 专门用于回答第二个问题：TriShift 的优势是否真正超越了 systematic variation。

Fig. 3a 首先比较 nearest retrieval 与 random retrieval。这个对比的目的不是展示一个实现细节，而是说明 reference retrieval 本身是否是有效信号。结果表明，在 Adamson 和 Norman 上，nearest retrieval 都系统性优于 random retrieval；也就是说，TriShift 的收益并不是来自随意的控制参考，而是来自更合理的 reference-conditioned matching。这一结果自然引出 Fig. 3b：如果 reference 真正重要，那么 TriShift 应该优于那些只利用平均扰动态的 reference baselines。事实也确实如此。无论是 `Perturbed mean` 还是 `Matching mean`，它们都可以捕捉一部分公共偏移，但一旦比较 reference-centered signal，它们就明显落后于 TriShift。

这一点在 Fig. 3c 中变得更明确。我们进一步构造了 `Residualized Systema Pearson`，即在 Systema 的参考差分基础上，先移除数据集级 generic shift 轴，再重新计算相关性。这个实验的目的很直接：如果一个模型的高分主要来自公共偏移，那么在 residualization 后它的优势应该迅速消失。结果表明，TriShift 在 Adamson 和 Norman 上仍保持最高或并列最高的 residualized score，而 `Perturbed mean` 和 `Matching mean` 的得分下降更明显。这说明 TriShift 的优势并不是建立在“更像一个平均被扰动细胞”之上，而是建立在更真实的 perturbation-specific residual recovery 之上。

Fig. 3d 从另一个角度强化了这一结论。Centroid accuracy 不再问“预测是否像一个扰动”，而是问“预测是否更像正确的那个扰动”。如果模型只学到 generic shift，它可能整体看起来像 perturbed cells，却未必最接近正确 perturbation 的真实 centroid。TriShift 在 Adamson 和 Norman 上都取得了最高的 centroid accuracy，说明它恢复的不是一个模糊的“被扰动状态”，而是更接近正确 perturbation identity 的条件表示。

最后，Fig. 3e 检验这种优势在更困难条件下是否仍然成立。我们按 train-distance 将条件分为 near、medium 和 far 三档，同时报告普通 Systema Pearson、Residualized Systema Pearson 和 generic projection ratio。结果显示，随着条件更远离训练分布，`Perturbed mean` 与 `Matching mean` 更容易退化为沿 generic shift 方向的预测，而 TriShift 保持了更低的 generic projection ratio 和更高的 residualized correlation。Fig. 3f 给出的 reference-sensitive case 进一步将这一机制结论具体化。整体来看，Fig. 3 并不是简单补充一些指标，而是建立了一条清楚的因果链：TriShift 的优势来自 reference-conditioned perturbation-specific recovery，而不是对系统性偏移的被动拟合。

### 6.3 TriShift more faithfully preserves condition-level distribution structure

如果 TriShift 真的学到了更好的 perturbation-specific response，那么这种优势不应只出现在均值相关性上，还应体现在条件级分布结构的恢复上。因此，我们在第三节结果中不再讨论“是否超越 systematic variation”，而转向另一个问题：TriShift 是否更好地恢复了真实条件分布的几何和统计结构。

为此，我们采用 scPRAM 兼容指标来评估预测分布在均值、方差和 Wasserstein 距离上的一致性。与 Pearson 或 nMSE 主要针对均值向量不同，这些指标更接近“预测出的整组细胞是否像真实条件分布”。整体上，TriShift 在 `scPRAM mean R^2`、`scPRAM var R^2` 和 Wasserstein 距离上都表现出稳定优势，尤其在方差恢复和 Wasserstein 距离上更为明显。这一点很重要，因为它说明 TriShift 的收益不只是把每个条件的均值预测得更准，而是更好地保留了真实扰动景观中的分布结构。

补充材料中的 Fig. S3 为这一结论提供了几何层面的支持。Fig. S3a 将 5 个 split 上的 `centroid_dist_mean` 汇总为单一比较图，显示 TriShift 的条件 centroid 与真实 centroid 的平均偏差最小；Fig. S3b 进一步在 representative split 上展示不同模型对应的条件级几何结构。按照 Scouter 的写法，这类 supplementary figure 最适合作为“主结论的结构性支持”，而不是在正文里重复列举新的指标。因此，这一节的核心结论应写成：TriShift 不仅在标量性能上更强，也更忠实地保留了条件级分布结构。

### 6.4 TriShift remains effective under hard generalization on Norman

在建立了标准 benchmark 和机制层面的证据之后，我们进一步考察一个更困难、也更接近真实外推边界的场景：Norman 组合扰动。与前两节不同，这一节不再追求跨数据集概括，而是聚焦于一个 stress test：当测试条件从 single 扰动扩展到不同难度的 unseen combinations 时，TriShift 是否仍然保持优势。

Fig. 4a 首先定义了 `single / seen2 / seen1 / seen0` 四类 subgroup。随后，Fig. 4b–e 从 Pearson、nMSE、DEG mean R2 和 Systema Pearson 四个角度比较不同模型在这些 subgroup 上的表现。这里最重要的趋势不是某个单独 subgroup 上的孤立高分，而是随着泛化难度升高，TriShift 仍然在关键指标上保持领先，尤其在 seen0、seen1 和 seen2 三个组合 subgroup 上表现最稳定。换言之，TriShift 的 reference-conditioned 设计并不是只在容易条件上有效；在 hardest generalization settings 中，它反而更能体现出优势。

Fig. 4f 提供了一个 representative combo case，用于把 subgroup-level 的统计结论落到具体组合扰动上。与 Fig. 2e 的角色类似，这里案例图的意义在于说明：TriShift 在 hard-generalization 场景中恢复的并不只是一个总体趋势，而是更接近真实组合扰动下关键基因的响应模式。因此，Fig. 4 所支撑的主张可以概括为：TriShift 的收益不仅存在于标准单扰动 benchmark，也延伸到了更困难的 Norman 组合泛化场景。

### 6.5 TriShift recovers biologically meaningful pathway programs

最后，我们检验前述数值和结构优势是否会进一步传导到更粗粒度的生物学解释层面。GEARS 和 Systema 的结果写法都表明，真正有说服力的论文收束往往不是“最后再加一个指标”，而是证明模型改进可以延伸到一个更接近 downstream use case 的层面。对本文而言，这个下游层面就是 pathway-level biological recovery。

Fig. 5a 汇总了 Norman 上不同模型的 `Top pathway overlap`。这一指标直接衡量预测 top pathways 与真实 top pathways 的重叠程度，因此比单纯的 DEG 相似度更接近 biological program recovery。结果显示，TriShift 取得了最高的平均 top-10 pathway overlap，并与 GenePert 形成领先组，整体高于 scGPT、biolord 和 GEARS。这一结果说明 TriShift 的优势并没有止步于表达层面，而是能够延伸到 pathway hierarchy 的恢复。

Fig. 5b 则给出一个 representative pathway case。与只比较 DEG 列表不同，这里直接对照真实和预测的 top pathways 及其相对强度。可以看到，TriShift 不仅命中了真实 pathway hierarchy 中的关键通路，而且在这些通路上的相对强度也与真实结果保持一致。换句话说，TriShift 的收益最终体现为更可信的 biological programs，而不只是更好看的数值指标。

综合 Fig. 2 到 Fig. 5，本文的结果链可以自然收束为一条清楚的叙事：TriShift 先在标准 benchmark 上建立稳定优势，随后证明这种优势并非来自系统性偏移，而是来自更真实的 perturbation-specific recovery；在此基础上，我们进一步看到它更好地保留了条件级分布结构，并在 Norman hard-generalization 场景和 pathway-level biology 上继续维持优势。这一顺序与 GEARS、Scouter 和 Systema 的结果写法一致，即先建立 benchmark，再解释机制，最后落到更困难或更接近生物学的问题上。
