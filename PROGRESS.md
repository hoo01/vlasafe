# VLA-SafeBench Progress

> 最后更新：2026-09-14。README 负责解释项目；本文只记录完成状态、证据和下一步。

## 当前判断

项目已完成 **task-4 Outcome v0.1 阶段快照**，尚未完成完整四周计划。

当前证据支持：冻结视觉和状态/动作历史能够在 held-out initial states 上预测 task 4 的最终 outcome；模型主要读取执行进度。

当前证据不支持：独立失效前兆、Impending Failure Detection、safe stop、RQ1 覆盖率或任务级泛化。

## 已完成

### 系统与数据

- SmolVLA + LIBERO task 4 闭环运行。
- 从 step 0 同步记录双相机 RGB、proprioception、预测/执行动作、timing、provenance 和 label-only diagnostics。
- Artifact validator 检查完成标记、字段、有限数值、连续 ID、时间戳和视频对齐。
- 正式 cohort：50 episodes，18 success / 32 failure。
- 冻结 split：30/10/10，按 `initial_state_id` 分组。
- Predictor deploy-input allowlist 与 privileged-field rejection 已测试。
- 25/25 单元测试通过。

### 路线选择

- 自然失败中的可用 self-collision/joint-violation 覆盖率：`1/32 = 3.125%`。
- Validation/test 均无 event-positive episode。
- 按预注册门槛选择 Outcome Prediction；停止 Impending lead-time 和 safe-stop 主张。

### Outcome 模型

| Method / checkpoint | Test AUPRC | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Initial-proprio | 0.799 | 0.452 | 0.238 | 0.173 |
| Temporal MLP, step 80 | 0.909 | 0.762 | 0.229 | 0.252 |
| Temporal MLP, step 120 | 0.982 | 0.952 | 0.096 | 0.095 |
| Dual-camera vision, step 80 | 0.982 | 0.952 | 0.100 | 0.096 |

Vision-step80 相对 initial-proprio：AUPRC `+0.183 [0.031, 0.450]`，AUROC `+0.500 [0.125, 0.857]`。Temporal-step120 的对应区间也支持排序改善。

### 机制审计

- 成功轨迹在 step 122–160 结束；所有失败轨迹运行到 step 280。
- RGB/error 审计显示风险主要随目标接近和成功构型进度变化。
- Vision-step80 与 temporal-step120 错排同一 failure/success pair。
- 两者概率 Pearson `0.9998`、Spearman `0.9515`。
- 关键 false negative 在 step 80 看似顺利，step 120–160 后才与成功轨迹可靠分叉，最终超时。

结论：保留 outcome prediction 结果；撤回“视觉更早发现独立失效前兆”的解释。

### 相机消融

| Input at step 80 | AUPRC | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Main | 0.844 | 0.667 | 0.306 | 0.330 |
| Wrist | 1.000 | 1.000 | 0.001 | 0.017 |
| Dual | 0.982 | 0.952 | 0.100 | 0.096 |

Wrist/dual 相对 main 的 Brier/ECE 改善得到配对区间支持；wrist 与 dual 没有可靠差异。正式 pipeline 仍使用预注册 dual-camera 配置。

### Offline utility 与展示

- Validation 阈值：`0.9998072982`，选择时要求 0 sacrificed validation successes。
- Test：检出 4/7 failures，观察到 0/3 sacrificed successes，节省 756 steps。
- 估算节省 52.95 s，按 `70.042 ms/step` 换算。
- 已生成 10 条 test risk 曲线。
- 已生成 360×360、20 FPS、280 帧 outcome-risk overlay video。
- 18 个 v0.1 artifact 的 SHA-256 检查为 18/18 通过。

这些是 offline faster-confirmation 结果，不是安全干预结果。

## 当前缺口

| 缺口 | 为什么重要 |
| --- | --- |
| 没有独立 confirmatory cohort | 当前主要结论依赖 10 条 test episode |
| 只有一个正式任务 | 无法判断信号是否只属于 task 4 |
| Failure 全为 280-step timeout | Outcome 很可能被完成进度和时长结构驱动 |
| 没有进度匹配评测 | 尚未严格分离进度与进度之外的失败信息 |
| RQ1 未测试 | 没有 A1/A2 注入、command-effect consistency 或 learned A2 coverage |
| Impending 数据不足 | 不能评测事件预警、lead time 或 safe stop |

## 下一步执行顺序

### 1. 独立 confirmatory cohort

- 保持 v0.1 cohort、test、checkpoint 和阈值不变。
- 使用新 seed 和 initial states 收集独立轨迹。
- 直接复验 vision-step80、temporal-step120 和固定 utility policy。
- 不用 confirmatory 数据重新选择模型或阈值。

### 2. 第二任务

- 对候选任务各跑 10–20 条筛选 mixed-success regime。
- 冻结任务后再收集正式 cohort。
- 单独报告结果，不把两个任务混成一个指标，也不声称 task-level generalization。

### 3. 进度控制实验

- 定义部署可得或仅用于分层评测的进度 proxy。
- 比较进度相近但 outcome 不同的 episode。
- 检查控制进度后 predictor 是否仍有可靠排序信号。

### 4. RQ1 supporting study

- A1：shape、dtype、NaN/Inf、timestamp 和显式 range 注入。
- A2：优先动作维度置换与主/腕相机映射错位。
- 实现最小 command-effect consistency。
- 报告前 N 步检出率、正常对照误报率和首次报警步数。

## 冻结边界

- v0.1 manifest、split、test predictions 和 SHA-256 清单不得为改善结果而修改。
- Outcome 与 Impending 标签、表格和结论保持分离。
- Privileged simulator state 不进入 predictor 或 normalization。
- 不从 test 选择模型、阈值或相机配置。
- 不把全零动作当作通用 safe stop。
- 不升级 Transformer 或扩展模型矩阵来追逐正结果。

## 产物位置

- 冻结 split：`docs/manifests/week1_task4_split.json`
- Artifact 索引与复现命令：`docs/release-artifacts.md`
- v0.1 checksum：`artifacts/release-sha256.txt`（实验主机）
- 主结果：`artifacts/results/week1_task4_*.json`（实验主机）
- 可视化：`artifacts/visualizations/week1_task4/`（实验主机）
