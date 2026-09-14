# VLA-SafeBench Progress

> 最后更新：2026-09-14。README 负责解释项目；本文只记录完成状态、证据和下一步。

## 当前判断

项目已完成 **task-4 Outcome v0.1 阶段快照**，尚未完成完整四周计划。

当前证据支持：冻结视觉和状态/动作历史能够在 held-out initial states 上预测 task 4 的最终 outcome；执行进度可以解释大部分排序能力。现有 10 条 test episode 不足以判断 vision-step80 是否还包含进度之外的信息。

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

### 进度控制实验

进度 proxy 定义为 frozen dual-camera features 在 train 数据中从平均 step 0 指向平均 step 120 的投影方向，不使用 outcome label 或 test 数据定义。Risk 对 progress 的线性关系只在 train 上拟合，冻结 test 只评一次。

| Model | Raw AUROC | Progress-only | Residualized risk | Partial r |
| --- | ---: | ---: | ---: | ---: |
| Vision step 80 | 0.952 | 0.905 | 0.714 | 0.552 |
| Temporal step 120 | 0.952 | 1.000 | 0.381 | -0.202 |

Vision residual AUROC CI `[0.111, 1.000]`、partial-r CI `[-0.232, 1.000]`，均无法排除无额外信号。Vision 只有两对 success/failure 在 0.5 train SD 内，虽均排序正确但 `n=2`；temporal 的最近匹配仍相差 1.61–2.08 SD。

结论：进度解释得到定量支持；数据缺少足够 outcome overlap，无法可靠检验 vision 的进度外信号。

## 当前缺口

| 缺口 | 为什么重要 |
| --- | --- |
| Failure 全为 280-step timeout | Outcome 很可能被完成进度和时长结构驱动 |
| Test 只有 10 episodes | 进度控制只有两对较近的 vision success/failure 匹配 |
| 只有一个正式任务 | 当前研究必须定位为单任务方法论研究 |
| RQ1 未测试 | 没有 A1/A2 注入、command-effect consistency 或 learned A2 coverage |
| Impending 数据不足 | 不能评测事件预警、lead time 或 safe stop |

## 下一步执行顺序

### 1. 文档与最终收口

- 对照原始四周目标逐项标记完成、部分完成和未完成。
- 根据进度控制结果更新机制结论和最终主表。
- 保留 v0.1 artifact，不用新分析反向修改冻结预测。

### 明确延期

- 独立 confirmatory cohort。
- 第二个任务。
- RQ1 A1/A2 fault-injection supporting study。
- Transformer、π0.5、RoboTwin、recovery 和 adaptive chunking。

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
