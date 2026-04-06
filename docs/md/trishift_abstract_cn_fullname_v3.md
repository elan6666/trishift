# 三元参考条件化状态转移模型（Tripartite Reference-Conditioned Shift Model, TriShift）

## 摘要

单细胞基因扰动预测的目标，是在有限实验覆盖下外推未测条件的转录组响应。现有方法通常将该任务表述为从扰动条件到表达终点的直接映射，但这种写法容易弱化控制细胞状态的作用，也难以区分模型究竟恢复了扰动特异信号，还是主要拟合了扰动样本与控制样本之间稳定存在的系统性偏移。本文提出三元参考条件化状态转移模型（Tripartite Reference-Conditioned Shift Model, TriShift），将单细胞扰动预测重新表述为参考条件化状态转移问题。TriShift 首先使用去噪变分自编码器学习可比较的潜空间；随后在潜空间中通过最优传输构造与目标扰动更一致的参考控制原型；再结合控制状态表示、GenePT 扰动语义表示和条件化位移表征，生成扰动后的基因表达。

在主文默认 benchmark 中，我们在 Adamson、Norman 和 Dixit 三个单扰动数据集上系统评估 TriShift。结果显示，TriShift 在三个数据集上均取得最低的 nMSE，并在 Adamson 与 Dixit 上取得最高的 Pearson 相关；在 Norman 上，TriShift 与最强 Pearson 基线保持接近，同时在误差尺度上更稳健。进一步的机制分析表明，TriShift 在去除数据集级 generic shift 后仍保持更高的 residualized Systema Pearson，并在 centroid accuracy 上持续优于 reference baselines，说明其优势并非仅来自对系统性偏移的拟合。围绕条件级分布结构的分析进一步显示，TriShift 在 scPRAM 兼容的均值、方差和 Wasserstein 指标上同样保持领先。最后，在 Norman 的 hard-generalization 组合扰动场景与 pathway-level biological recovery 分析中，TriShift 仍表现出稳定优势。

整体而言，TriShift 将参考匹配、状态位移建模和表达生成统一到同一链路中，为未见扰动和困难泛化场景下的单细胞转录组响应预测提供了一种更贴近生物学过程、也更便于解释的建模框架。
