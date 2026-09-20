# VLA-SafeBench Progress

> 最后更新：2026-09-20。README 负责解释项目；本文只记录完成状态、证据和下一步。

## 当前判断

项目已完成并冻结四周计划的 v0.1 有界 release。现已启动 **v0.2 完整性审计**；新分析和新数据不得覆盖 v0.1 manifest、预测或 checksum。

当前证据支持：冻结视觉和状态/动作历史能够在 held-out initial states 上预测 task 4 的最终 outcome；执行进度可以解释大部分排序能力。现有 10 条 test episode 不足以判断 vision-step80 是否还包含进度之外的信息。

当前证据不支持：独立失效前兆、Impending Failure Detection、safe stop、未测试 A2 类型的覆盖率或任务级泛化。

### v0.2 审计触发原因

1. `policy_record.py` 将 `episode_index` 写为 `initial_state_id`，但环境通过 `env.reset(seed=seed)` 重置；必须确认该编号是否对应 LIBERO 实际使用的固定初始状态。核验前，“held-out initial state”属于待审计主张。
2. Temporal MLP 显式输入 `checkpoint_step / 280`。该值在同一 checkpoint 内为常数，不直接产生该 checkpoint 内的排序，但可能影响跨 checkpoint 联合训练、概率尺度和校准，需要固定配置消融。
3. Validation/test 各只有 10 条，不能通过继续调参解决校准与置信区间问题；若需要增强证据，必须使用冻结 pipeline 和真正独立的新 cohort。

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
- 最终 21 个 release artifact 的 SHA-256 检查为 21/21 通过。

这些是 offline faster-confirmation 结果，不是安全干预结果。

### 进度控制实验

进度 proxy 定义为 frozen dual-camera features 在 train 数据中从平均 step 0 指向平均 step 120 的投影方向，不使用 outcome label 或 test 数据定义。Risk 对 progress 的线性关系只在 train 上拟合，冻结 test 只评一次。

| Model | Raw AUROC | Progress-only | Residualized risk | Partial r |
| --- | ---: | ---: | ---: | ---: |
| Vision step 80 | 0.952 | 0.905 | 0.714 | 0.552 |
| Temporal step 120 | 0.952 | 1.000 | 0.381 | -0.202 |

Vision residual AUROC CI `[0.111, 1.000]`、partial-r CI `[-0.232, 1.000]`，均无法排除无额外信号。Vision 只有两对 success/failure 在 0.5 train SD 内，虽均排序正确但 `n=2`；temporal 的最近匹配仍相差 1.61–2.08 SD。

结论：进度解释得到定量支持；数据缺少足够 outcome overlap，无法可靠检验 vision 的进度外信号。

### RQ1 supporting study

- A1：11/11 预定义协议案例得到预期处理。
- A2：20 组同 seed/initial-state 三路配对 rollout；前 10 组 normal calibration，后 10 组 evaluation；统一统计前 40 步。
- Normal：command-effect 1/10，95% Wilson CI `[0.018, 0.404]`，success 6/10。
- `action_swap_xy`：command-effect 10/10，CI `[0.722, 1.000]`，首次报警中位 step 3，success 0/10。
- `camera_swap`：command-effect 7/10，CI `[0.397, 0.892]`，首次报警中位 step 29，success 0/10；只解释为闭环异常的间接响应。
- 协议规则对两类 A2 均为 0/10。Range warning 在 normal 为 10/10，不能用作 A2 检测。

## 当前缺口

| 缺口 | 为什么重要 |
| --- | --- |
| Failure 全为 280-step timeout | Outcome 很可能被完成进度和时长结构驱动 |
| Test 只有 10 episodes | 进度控制只有两对较近的 vision success/failure 匹配 |
| 只有一个正式任务 | 当前研究必须定位为单任务方法论研究 |
| RQ1 范围有限 | 只测试一个任务、两类 A2、10 组 evaluation pairs；无 learned A2 monitor |
| Impending 数据不足 | 不能评测事件预警、lead time 或 safe stop |

## 下一步执行顺序

### 1. Initial-state identity audit（进行中）

- 检查固定 LeRobot/LIBERO revision 的 reset 与 init-state 选择逻辑。
- 对重复 seed 和多个 seed 记录环境实际 init-state index；若接口不暴露编号，则对 reset 后完整 MuJoCo state 生成仅用于审计的 fingerprint。
- 检查现有 metadata 中相同 `initial_state_id` 是否真的对应相同初始状态。
- 若不成立：v0.1 只保留 held-out episode 结论；v0.2 使用真实 ID/fingerprint 重建 manifest，不覆盖 v0.1。

### 2. Temporal checkpoint-progress ablation（进行中）

- 保持数据、split、训练 seeds、MLP、优化器和 early stopping 不变。
- 比较包含与移除 `checkpoint_step / 280` 的两个版本。
- 按 episode、checkpoint 做 paired bootstrap，分别报告 AUPRC、AUROC、Brier 和 ECE 差值。
- 该实验用于解释特征依赖和概率尺度，不根据 test 选择新主模型。

### 3. 新数据决策（等待前两项）

- 只有确认存在真正未使用的 init states 后，才采独立 calibration/confirmation cohort。
- 新 cohort 采集前冻结模型、progress proxy、阈值、checkpoint 和指标。
- 若 task 4 没有未使用状态，则筛选第二个同任务内具有混合 outcome 的任务，单独复现方法；不得把不同 task 的成功/失败直接混合。

### v0.1 已完成的收口

- 两个 RQ1 result JSON 已加入 21-file release checksum；A2 report 保留全部 paired episode ID、seed 和逐条结果。
- 保留 v0.1 artifact，没有用新分析反向修改冻结预测。

### 仍不进入 v0.2 的工作

- 专门训练的 learned A2 monitor。
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
