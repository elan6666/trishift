# Codex App 官方文档中文整理（意译版）

> 目标：基于 OpenAI Developers 的 `codex/app` 文档页，做一份可直接上手的中文说明。
> 
> 说明：以下为**结构化意译 + 实操讲解 + 示例**，不是逐句逐字直译；重点保留官方功能和操作路径。
> 
> 信息来源时间：检索于 2026-02-27（美国时区）。

## 0. 文档范围

本整理覆盖 `https://developers.openai.com/codex/app` 及其子页：

- 概览（app）
- Features
- Settings
- Review
- Automations
- Worktrees
- Local Environments
- Commands
- Troubleshooting

---

## 1. 概览（Codex App 是什么）

Codex App 是一个桌面端工程工作台，用来把“对话式代码协作”与“真实工程流程”合并在一起：

- 多线程并行处理任务
- 在本地或隔离工作树里执行改动
- 直接查看和审阅 Git diff
- 将周期性任务交给自动化

### 官方给出的上手顺序（中文化）

1. 安装应用并登录。
2. 选择你的代码目录（workspace）。
3. 打开一个线程并发送第一条任务消息。
4. 默认线程通常是 `Local` 执行模式。

### 运行模式（官方语义）

- `Local`：直接在当前本地目录执行。
- `Worktree`：在隔离 Git worktree 中执行，避免污染当前工作目录。
- `Cloud`：在云端环境执行（适合你不想占用本机算力的任务）。

### 示例

你要并行做两件事：

- 线程 A（Local）：修一个小 bug 并本地跑单测。
- 线程 B（Worktree）：尝试重构，不影响当前工作目录。

---

## 2. Features（核心能力）

### 2.1 多线程并行（Multitask）

你可以在多个线程中同时推进任务，每个线程保留自己的上下文与执行历史。

**例子**：

- 线程 1：排查训练脚本参数错误。
- 线程 2：生成实验汇总报告。
- 线程 3：整理 README。

### 2.2 Skills 支持

应用支持通过 skills 注入专门工作流（例如部署、游戏调试、Notebook 模板化等）。

**例子**：

- 调用 `jupyter-notebook` skill 快速创建实验 notebook。

### 2.3 内建 Git 工具

可在 UI 中完成常见 Git 操作与审阅流程：

- Stage / Unstage
- Revert（按文件或按片段）
- Commit / Push
- 发起 PR（在支持的集成下）

### 2.4 Worktree 支持

Codex 可在隔离 worktree 中执行任务，降低“实验性修改”影响主分支的风险。

### 2.5 集成终端

每个线程可配套一个终端上下文，便于在同一任务链路里执行命令和校验结果。

### 2.6 IDE 扩展同步

支持与 IDE 扩展联动，便于在编辑器和 App 之间同步上下文/操作。

### 2.7 语音与图片输入

- 语音：可用语音方式发起任务。
- 图片：可上传界面截图/图表帮助说明问题。

### 2.8 本地通知与收件箱

- 长任务结束可通过系统通知提醒。
- 自动化或后台任务结果进入 Inbox，便于统一处理。

### 2.9 保持唤醒（Keep awake）

可配置运行时保持系统唤醒，避免任务中断。

---

## 3. Settings（设置页）

官方设置页主要分成“体验、行为、集成、工程上下文”几类。

### 3.1 常规 / 外观 / 通知

- 常规：基础行为。
- 外观：主题和视觉偏好。
- 通知：任务完成和提醒策略。

### 3.2 个性化（Personalization）

定义默认回复风格、协作偏好，减少重复沟通成本。

### 3.3 Agent Configuration（`config.toml`）

用于更细粒度控制代理行为（例如模型偏好、执行策略等，依版本而异）。

### 3.4 MCP Servers

配置外部能力接入（第三方系统、内部工具、数据源），让 Codex 能“直接调用工具”。

### 3.5 Git / Environment / Worktrees / Archived threads

- Git：仓库与提交流程相关设置。
- Environment：本地执行环境与动作。
- Worktrees：隔离工作树的使用和清理。
- Archived threads：历史线程归档管理。

**例子**：

你做训练实验时，`Environment` 里预设好激活环境与依赖安装动作；`Worktrees` 开启后每次 ablation 都在隔离分支执行。

---

## 4. Review（代码审阅）

Review 面板用于集中看差异并给出修复意见。

### 4.1 审阅入口与范围

可选择不同审阅范围：

- 当前未提交改动
- 整个分支改动
- 上一轮对话相关改动

### 4.2 行内评论（Inline comments）

可在 diff 行上直接给评论，让 Codex 按评论点修复。

### 4.3 实践建议

1. 先选小范围（例如本轮改动）快速收敛。
2. 再切到全分支范围做回归检查。
3. 用行内评论标记“必须修”的点，避免口头遗漏。

---

## 5. Automations（自动化）

自动化用于定时或周期运行任务，结果投递到 Inbox。

### 5.1 自动化能做什么

- 定时执行检查（例如日志、构建、质量扫描）
- 周期性汇总（日报/周报）
- 触发后生成可审阅结果

### 5.2 调度语义（官方 UI 支持范围）

常见是按小时/按周调度；执行完成后会在收件箱生成结果项。

### 5.3 与 Git 项目的关系

在 Git 项目中，自动化倾向于在后台隔离环境（worktree）运行，避免污染主工作区。

### 5.4 示例：每周实验汇总

- 任务：汇总 `artifacts/ablation` 本周新增结果。
- 调度：每周五 18:00。
- 输出：一份 Markdown 摘要，写入 inbox 结果。

---

## 6. Worktrees（隔离工作树）

Worktree 是并行开发和风险隔离的核心机制。

### 6.1 为什么用

- 同仓库并行做多任务，互不覆盖。
- 重构/实验分支不干扰当前稳定分支。

### 6.2 创建方式（官方文档语义）

- 方式 A：新建 `Worktree` 线程时自动创建隔离分支与目录。
- 方式 B：把已有本地 worktree 关联到一个线程。

### 6.3 清理策略

任务结束后及时清理不再使用的 worktree，保持仓库整洁。

### 6.4 示例

- `main` 保持可发布状态。
- `codex/ablation-k1` 在独立 worktree 跑实验改动。
- 完成后合并，删除该 worktree。

---

## 7. Local Environments（本地环境）

本地环境机制用于把“项目准备动作”和“常用命令”标准化。

### 7.1 Setup scripts

用于初始化环境（例如安装依赖、准备缓存）。

### 7.2 Actions

把常用操作做成一键动作（例如启动服务、跑测试、导出报告）。

### 7.3 在项目内落地（文档语义）

可在项目目录下维护专用配置（文档示例提到 `.codex` 目录约定），让团队成员获得一致执行体验。

### 7.4 示例

- Setup：`pip install -e .`。
- Action：`pytest -q`。
- Action：`python scripts/run_ablation.py --config configs/defaults.yaml`。

---

## 8. Commands（快捷命令）

官方文档列出快捷键和 slash commands 用于加速操作。

### 8.1 常见快捷键（示例）

- 新建聊天
- 打开命令面板
- 打开设置
- 在线程间切换
- 展开/收起 diff 视图

### 8.2 常见 slash 命令（示例）

- `/review`：进入审阅流程
- `/status`：查看当前线程状态
- `/mcp`：查看可用 MCP 工具
- `/plan-mode`：切换为计划协作模式

---

## 9. Troubleshooting（故障排查）

### 9.1 登录卡住

官方建议可从终端启动并重新认证（例如 `codex` + 登录命令）。

### 9.2 Cloud 连接异常

在 Cloud 设置中尝试重连；必要时重新认证或重启会话。

### 9.3 收不到通知

检查应用内通知开关和系统层面的通知权限。

### 9.4 SSH key / 密钥问题

检查本地 SSH 配置与仓库访问权限是否匹配。

### 9.5 版本相关问题

确认是否为最新版本，再复现问题并提交反馈。

---

## 10. 给你当前项目（`e:\CODE\trishift`）的实操模板

### 10.1 推荐最小配置

1. `工作树`：固定到 `e:\CODE\trishift`。
2. `环境`：预置 Python 环境激活 + 常用命令动作。
3. `Git`：默认使用 `codex/*` 分支做实验性改动。
4. `Review`：先审“上一轮改动”，再审“全分支”。

### 10.2 你可以直接这么用

- 线程 A（Local）：对 `configs/defaults.yaml` 调参数并做快速验证。
- 线程 B（Worktree）：在 `codex/ablation-scpram` 分支跑完整 ablation。
- 自动化：每周生成一次 `artifacts/ablation` 结果摘要。

---

## 11. 参考链接（官方）

- https://developers.openai.com/codex/app
- https://developers.openai.com/codex/app/features
- https://developers.openai.com/codex/app/settings
- https://developers.openai.com/codex/app/review
- https://developers.openai.com/codex/app/automations
- https://developers.openai.com/codex/app/worktrees
- https://developers.openai.com/codex/app/local-environments
- https://developers.openai.com/codex/app/commands
- https://developers.openai.com/codex/app/troubleshooting

---

## 12. 差异说明（重要）

- 官方页面某些平台可用性描述可能与你当前实装版本存在时间差。
- 若你看到 UI 文案与本文不同，以你当前 App 版本中的文案为准。
