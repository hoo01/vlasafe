# VLA-SafeBench Progress

> 最后更新：2026-09-20。README 负责解释项目；本文只记录完成状态、证据和下一步。

## 当前判断

项目已完成并冻结四周计划的 v0.1 有界 release，并完成 **v0.2 完整性审计与独立确认集**；v0.1 manifest、预测和 checksum 保持不变。

当前证据支持：冻结视觉和状态/动作历史能够在 held-out initial states 上预测 task 4 的最终 outcome；该关联在 20 条全新 preset-state confirmation episodes 上复现。执行进度可以解释大部分排序能力，现有数据仍不能判断 vision-step80 是否还包含进度之外的信息。

当前证据不支持：独立失效前兆、Impending Failure Detection、safe stop、未覆盖 A2 类型的检测率或任务级泛化。

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
- 38/38 单元测试通过。

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

### v0.2 完整性审计与独立确认

- Initial-state identity：50/50 条 claimed ID 等于 reset 前真实 LIBERO `init_state_id`；重复 ID 的 preset fingerprint 一致；原 group-disjoint split 成立。
- Temporal checkpoint-feature ablation：移除 `checkpoint_step / 280` 后，test step 80/120 的 AUPRC 和 AUROC 均不变。Step 80 Brier `+0.018 [0.004, 0.035]`、ECE `+0.013 [0.004, 0.026]`，表明该特征影响概率尺度而非同 checkpoint 排序。
- Confirmation cohort：preset states 30–49，共 20 条，11 success / 9 failure；采集前冻结 v0.1 predictor、normalization、checkpoint 和指标。
- Frozen vision step 80：AUPRC `0.939 [0.786, 1.000]`，AUROC `0.919 [0.747, 1.000]`，Brier `0.102`，ECE `0.114`。
- Frozen temporal step 120：AUPRC `0.882 [0.671, 1.000]`，AUROC `0.828 [0.571, 1.000]`，Brier `0.140`，ECE `0.140`。
- Step 0 两种模型均接近无排序信息；确认集支持 outcome association 在未见 preset states 上复现，但不证明独立失效前兆或任务级泛化。

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

### 1. Initial-state identity audit（完成）

- 检查固定 LeRobot/LIBERO revision 的 reset 与 init-state 选择逻辑。
- 对重复 seed 和多个 seed 记录环境实际 init-state index；若接口不暴露编号，则对 reset 后完整 MuJoCo state 生成仅用于审计的 fingerprint。
- 检查现有 metadata 中相同 `initial_state_id` 是否真的对应相同初始状态。
- 结果：50/50 claimed ID 与真实 preset index 一致，重复 ID 的 preset fingerprint 一致；完整 simulator fingerprint 会随 seed 改变，不作为分组 ID。

### 2. Temporal checkpoint-progress ablation（完成）

- 保持数据、split、训练 seeds、MLP、优化器和 early stopping 不变。
- 比较包含与移除 `checkpoint_step / 280` 的两个版本。
- 按 episode、checkpoint 做 paired bootstrap，分别报告 AUPRC、AUROC、Brier 和 ECE 差值。
- 结果：排序指标不依赖该特征；它会影响概率尺度。不根据 test 选择新主模型。

### 3. 独立 confirmation cohort（完成）

- 已确认 task 4 共有 50 个 LIBERO preset states；v0.1 仅使用 0–29。v0.2 使用 30–49 采 20 条独立 confirmation episodes。
- 新 cohort 采集前冻结模型、progress proxy、阈值、checkpoint 和指标。
- confirmation cohort 只评测冻结的 v0.1 predictor，不重新训练、调阈值或选择 checkpoint；结果单独报告，不并入 v0.1 test。

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
- v0.2 confirmation manifest：`docs/manifests/v02_task4_confirmation.json`
- v0.2 小型结果 JSON：`artifacts/results/v02_*.json`
- v0.2 大型 artifact checksum：`artifacts/v02/release-sha256.txt`
- v0.1 checksum：`artifacts/release-sha256.txt`（实验主机）
- 主结果：`artifacts/results/week1_task4_*.json`（实验主机）
- 可视化：`artifacts/visualizations/week1_task4/`（实验主机）
