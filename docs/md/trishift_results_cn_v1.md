# 三元参考条件化状态转移模型（Tripartite Reference-Conditioned Shift Model, TriShift）

## 6 Results

### 6.1 TriShift improves standard benchmark performance across datasets

我们首先在主文默认 benchmark 上评估 TriShift 的整体预测性能。Fig. 2a 和 Fig. 2b 分别比较 task-specific baseline 组与 semantic/foundation 组上的 Pearson 相关。TriShift 在 Adamson 和 Dixit 上取得最高的 Pearson，分别达到 0.945 和 0.911；在 Norman 上，GenePert 取得略高的 Pearson（0.839 对 0.832），但 TriShift 与其差距很小，同时明显优于 GEARS、biolord 和 scGPT。这说明 TriShift 在标准单扰动设置下并非只在某个特定数据集上有效，而是在三套主文数据中都保持了稳定竞争力。

Fig. 2c 和 Fig. 2d 进一步从误差尺度检验这一趋势。与 Pearson 不同，nMSE 直接衡量模型相对于“控制均值”弱基线的幅值恢复能力。在这一指标上，TriShift 在三个数据集上都取得最低值，分别为 Adamson 0.168、Norman 0.271 和 Dixit 0.250；其余方法在 Adamson 与 Dixit 上均与 TriShift 存在明显差距，在 Norman 上也未能超过 TriShift。这一结果表明，TriShift 的优势不仅体现在方向一致性上，也体现在对扰动幅值的稳健校准上。

Fig. 2e 给出了一个共享基线覆盖的代表性单扰动案例。该图并不承担新的统计主张，而是将跨数据集 benchmark 的结论落到具体基因表达变化上：与其他基线相比，TriShift 更稳定地恢复了显示基因集合上相对于控制的表达变化方向和幅度。结合 Fig. 2a–e，可以将主文的第一条结论概括为：TriShift 在标准单扰动 benchmark 上取得了最稳健的整体表现，尤其是在误差尺度上表现最强。

### 6.2 TriShift better recovers perturbation-specific responses beyond systematic variation

主 benchmark 的提升并不自动意味着模型恢复了真正的 perturbation-specific response。为了区分“恢复扰动特异信号”和“拟合公共系统性偏移”，我们将 Fig. 3 专门用于机制分析。Fig. 3a 先比较 nearest retrieval 与 random retrieval。无论在 Adamson 还是 Norman，nearest retrieval 都提高了常规 Pearson 和 Systema Pearson：例如在 Adamson 上，TriShift nearest 的 Pearson 为 0.945，而 random 版本为 0.911；在 reference-centered 的 Systema Pearson 上，两者分别为 0.526 和 0.342。Norman 上也呈现相同趋势，说明参考检索本身就是 TriShift 收益的一部分，而不是后处理细节。

Fig. 3b 将 TriShift 与两类 reference baselines 直接对照。`Perturbed mean` 和 `Matching mean` 分别代表“公共扰动态均值”与“匹配后均值”的极简 reference 策略。它们在某些常规指标上可以取得不差的表现，但一旦评估转向 perturbation-specific recovery，这些方法的局限就变得明显。尤其在 Adamson 上，两类 mean baseline 的 Systema Pearson 都明显低于 TriShift；在 Norman 上，Matching mean 虽然优于 Perturbed mean，但仍落后于 TriShift。

这一点在 Fig. 3c 中得到更直接的量化。我们对 Systema 指标进一步做 residualization，先移除数据集级 generic shift 轴，再计算 `Residualized Systema Pearson`。如果模型的高分主要来自公共偏移，那么该指标应显著下降。结果表明，TriShift 在 Adamson 和 Norman 上分别达到 0.423 与 0.783，而 Perturbed mean 仅为 0.213 与 0.211。Matching mean 在 Adamson 上同样只有 0.213，在 Norman 上虽可达到 0.740，但仍低于 TriShift。这说明 TriShift 的优势并非简单来自对 generic drift 的拟合，而是在去除公共方向后仍保留了更强的 perturbation-specific residual signal。

Fig. 3d 从另一角度验证了这一点。Centroid accuracy 衡量预测后的条件 centroid 是否更接近正确 perturbation 的真实 centroid。若模型只学到一个平均扰动态，它可能整体像“被扰动细胞”，却未必最接近正确的 perturbation identity。TriShift 在 Adamson 与 Norman 上分别取得 0.602 和 0.935 的 centroid accuracy，高于 random retrieval 的 0.557 和 0.911，也高于两类 mean baseline。这表明 TriShift 不仅恢复了“被扰动”的总体状态，而且更好地恢复了“正确的那个扰动”。

Fig. 3e 则进一步检验这种优势是否会随着测试条件更远离训练分布而消失。我们按 train-distance 将条件分为 near、medium 和 far 三档，同时报告普通 Systema Pearson、Residualized Systema Pearson 和 generic projection ratio。对 Adamson 而言，Perturbed mean 与 Matching mean 的 generic projection ratio 几乎等于 1，说明其预测几乎完全沿着公共系统性偏移轴展开；TriShift 则保持在 0.79–0.83 之间。Norman 上这种差异更明显：Perturbed mean 的 projection ratio 约为 0.99，而 TriShift 约为 0.36–0.43。换言之，TriShift 不仅在 reference-centered 相关性上得分更高，而且显著更少塌缩到 generic shift 方向。Fig. 3f 的 reference-sensitive case 进一步给出了这一机制结论的直观例子。整体来看，Fig. 3 支撑了本文的第二条核心结论：TriShift 的优势主要来自更好的 perturbation-specific recovery，而不是对系统性偏移的被动拟合。

### 6.3 TriShift better preserves condition-level distribution structure

除了均值层面的 benchmark 和 Systema 机制分析，我们还考察 TriShift 是否更好地恢复了条件级分布结构。这里我们采用 scPRAM 兼容指标，包括 `mean R^2`、`var R^2` 和 Wasserstein 距离。与传统 Pearson 或 nMSE 不同，这组指标更强调预测分布是否保留了真实条件下的均值、方差和异质性结构，而不只是恢复单个条件均值。

在 `scPRAM mean R^2` 上，TriShift 在三个主文数据集上均取得最佳结果：Adamson 为 0.946，Norman 为 0.961，Dixit 为 0.973。相比之下，GEARS 和 GenePert 虽然在个别数据集上也能取得较高分数，但总体仍低于 TriShift。在更严格的 `var R^2` 指标上，TriShift 的优势更明显：Adamson、Norman 与 Dixit 分别达到 0.826、0.512 和 0.666，而当前可比的 scGPT 分别只有 0.122、0.298 和 0.061。最后，在 Wasserstein 距离上，TriShift 也在三个数据集上都取得最低值，分别为 3.82、5.54 和 2.56，明显优于 GEARS、GenePert、biolord 和 scGPT。这说明 TriShift 的优势不仅体现在标量指标上，也体现在对条件级表达分布的恢复上。

补充材料中的 Fig. S3 进一步为这一结论提供支持。Fig. S3a 对 5 个 split 的 `centroid_dist_mean` 做汇总比较，显示 TriShift 在 Norman 上的平均 centroid 偏差最小。Fig. S3b 给出一个代表性 split 的条件 centroid 可视化，进一步说明 TriShift 在条件几何结构上更接近真实扰动分布。与 Fig. 3 的机制证据不同，这里强调的是结构恢复本身：TriShift 预测出的条件级景观不仅更“像正确扰动”，也更“像正确分布”。

### 6.4 TriShift remains effective under hard generalization on Norman

接下来我们考察更困难的组合泛化场景。Fig. 4 将 Norman 组合扰动按照 `single / seen2 / seen1 / seen0` 划分为不同难度的 subgroup，其中 `seen0` 代表最困难的完全未见组合。Fig. 4b 显示，TriShift 在所有三个组合 subgroup 上都取得最高的 Pearson：seen0、seen1 和 seen2 分别为 0.956、0.954 和 0.978。对于 single subgroup，TriShift 的 Pearson 为 0.670，虽低于 GenePert 的 0.766，但仍与 GEARS、scGPT 和 biolord 相当。

Fig. 4c 从 nMSE 角度给出更清晰的结论。TriShift 在 seen0、seen1、seen2 和 single 四个 subgroup 上都取得最低 nMSE，分别为 0.148、0.109、0.081 和 0.482。这说明即使在最困难的 unseen-combination 场景下，TriShift 仍能稳定控制误差幅值，而不只是偶然在相关性上占优。

Fig. 4d 和 Fig. 4e 则表明，这种优势同样延伸到扰动特异指标上。TriShift 在 DEG mean R2 和 Systema Pearson 上均保持领先，尤其在 seen0/seen1/seen2 三个组合 subgroup 上优势最为明显。换言之，随着泛化难度升高，TriShift 的 reference-conditioned 设计并没有失去作用，反而在 hardest cases 上表现出更稳定的收益边界。Fig. 4f 的 representative combo case 为这一趋势提供了具体例子：在 unseen combo 条件下，TriShift 对关键基因方向和幅值的恢复更接近真实扰动响应。

### 6.5 TriShift recovers biologically meaningful pathway programs

最后，我们检验 TriShift 的数值增益是否延伸到 pathway-level biological recovery。Fig. 5a 汇总了 Norman 上五个模型的 `Top pathway overlap`。TriShift 取得最高的平均 top-10 pathway overlap，达到 4.23，与 GenePert 基本持平，但明显高于 scGPT 的 3.45、biolord 的 3.40 和 GEARS 的 2.82。该结果说明，TriShift 不仅更接近真实 DEG 方向，也更能恢复由这些 DEG 所支撑的高层次通路结构。

Fig. 5b 给出一个代表性 pathway case。与只比较 DEG 列表不同，该图直接对照真实 top pathways 与 TriShift 预测 top pathways 的重叠和相对强度。可以看到，TriShift 恢复出的主导 pathway 与真实 pathway hierarchy 大体一致，且在关键通路上的 combined score 保持接近。这说明 TriShift 的收益并没有停留在单基因或条件均值层面，而是进一步传导到了 pathway program 的恢复上。

综上，TriShift 的结果链可以概括为四步。首先，TriShift 在标准 benchmark 上取得最稳健的整体表现；其次，这种优势在去除系统性偏移后仍然存在，并体现为更好的 perturbation identity recovery；再次，TriShift 更好地保留了条件级分布结构；最后，这些数值和结构优势进一步转化为更强的 hard-generalization 表现和更具生物学意义的 pathway-level recovery。
