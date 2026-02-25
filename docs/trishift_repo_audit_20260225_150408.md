# TriShift 仓库体检与重构报告（修正版：按模型 + 按数据集目录）

## 范围

本轮覆盖本地三模型脚本重构（不改 `external/`）：

- `TriShift`
- `Scouter`
- `Systema`

目标结构是：

- 第一层按模型
- 第二层按数据集目录
- 每个数据集目录内放 `run_xxx.py + yaml config`
- 共享实现放 `_core/`
- 旧平铺 `scripts/*.py` 保留兼容壳

## 本轮已处理

### 1) 结构重组（按模型 + 按数据集）

新增/落地的主结构：

- `scripts/trishift/_core/`
- `scripts/trishift/<dataset>/run_<dataset>.py + config.yaml`
- `scripts/scouter/_core/`
- `scripts/scouter/<dataset>/run_scouter_*.py + config.yaml`
- `scripts/systema/_core/`
- `scripts/systema/<dataset>/run_systema_baselines_*.py + systema_baselines.yaml`
- `scripts/systema/adamson/run_systema_ref_compare_adamson.py + systema_ref_compare.yaml`

说明：

- `TriShift` Adamson 的消融/组合消融脚本已迁至 `scripts/trishift/adamson/ablation/`
- 原 `scripts/trishift/ablation/...` 路径保留为兼容 wrapper（转发到 Adamson 数据集目录）

### 2) `_core` 内部实现收敛

过渡版任务目录中的核心实现已收敛到 `_core`：

- `scripts/trishift/_core/run_dataset_core.py`
- `scripts/trishift/_core/train_main_core.py`
- `scripts/scouter/_core/scouter_eval_core.py`
- `scripts/systema/_core/baselines_core.py`
- `scripts/systema/_core/ref_compare_core.py`

同时保留任务目录 wrapper：

- `scripts/trishift/train/main.py`
- `scripts/trishift/train/run_dataset.py`
- `scripts/scouter/eval/main.py`
- `scripts/systema/baselines/main.py`
- `scripts/systema/ref_compare/main.py`

### 3) 共享 split 逻辑复用（TriShift + Scouter）

已使用共享模块（Norman split 对齐）：

- `scripts/common/split_utils.py`

接入：

- `scripts/trishift/_core/run_dataset_core.py`
- `scripts/scouter/_core/scouter_eval_core.py`

这项修复了 TriShift 与 Scouter split 逻辑漂移风险（此前为重复实现）。

### 4) 兼容壳保留（旧命令不破坏）

旧平铺入口仍可用，包括：

- `scripts/run_norman.py`
- `scripts/run_ablation.py`
- `scripts/run_combo_ablation.py`
- `scripts/scouter_eval.py`
- `scripts/run_scouter_eval_norman.py`
- `scripts/run_systema_baselines.py`
- `scripts/systema_ref_compare.py`

关键旧 import API 仍可用：

- `from scripts.run_dataset import run_dataset, run_dataset_with_paths`
- `from scripts.scouter_eval import run_scouter_eval`

## 性能问题清单（Deferred）

### P0：TriShift 评估重复推理

问题：

- `TriShift.evaluate()` 与 `TriShift.export_predictions()` 对同一 split 做了两遍推理

影响：

- 评估阶段推理开销重复，`n_eval_ensemble` 大时更明显

状态：

- 本轮未改（避免结构重构叠加核心评估逻辑修改）
- 已记录为 Phase 2 项

## 逻辑风险清单

### P0：TriShift / Scouter split 逻辑漂移（本轮已处理）

问题：

- `run_dataset` 与 `scouter_eval` 各自维护 Norman split 逻辑，可能导致对比实验出现“伪差异”

处理：

- 抽取 `scripts/common/split_utils.py`
- 两端改为共享实现（通过兼容 wrapper 接入，降低改动面）

## 结构问题清单（本轮已处理/缓解）

1. `scripts/` 平铺混杂职责
- 已按模型重组，并进一步提供按数据集目录入口

2. 配置未按模型+数据集组织
- 已在各数据集目录放本地 yaml config（TriShift / Scouter / Systema）

3. 数据集入口分散且命名不统一
- 新推荐入口统一为模型目录下数据集脚本

## 验证（本轮执行）

静态检查：

- `python -m compileall -q scripts`

新入口 smoke（帮助）：

- `python scripts/trishift/norman/run_norman.py --help`
- `python scripts/scouter/norman/run_scouter_norman.py --help`
- `python scripts/systema/adamson/run_systema_baselines_adamson.py --help`
- `python scripts/systema/adamson/run_systema_ref_compare_adamson.py --help`

兼容入口 smoke（帮助）：

- `python scripts/run_norman.py --help`
- `python scripts/run_ablation.py --help`
- `python scripts/run_combo_ablation.py --help`
- `python scripts/run_systema_baselines.py --help`

兼容 import API：

- `from scripts.run_dataset import run_dataset, run_dataset_with_paths`
- `from scripts.scouter_eval import run_scouter_eval`

## Phase 2（后续建议）

1. 合并 `TriShift.evaluate()` 与 `export_predictions()` 的推理循环（性能优化）
2. 继续下沉脚本重复工具逻辑到 `scripts/common/`
3. 在文档中逐步把示例命令切换到新“模型 + 数据集”入口

## 说明

- 本轮不修改 `external/scouter/...` 与 `external/systema-main/...`
- 过渡版任务目录（如 `scripts/trishift/train/`, `scripts/scouter/eval/`）保留为兼容层
