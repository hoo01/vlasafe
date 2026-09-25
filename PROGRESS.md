# VLA-SafeBench Progress

> 最后更新：2026-09-25。README 解释项目与结论；本文只记录冻结状态、关键证据和 Phase 2 入口。

## 当前判断

Phase 1 已完成。冻结视觉和状态/动作历史能够稳定预测 LIBERO-Spatial task 4 的最终 outcome，但预先冻结的 v0.3 阶段控制实验确认：**已观察到的排序能力主要由任务执行进度解释；控制真实物体阶段后，没有发现可靠的阶段外预测信号。**

当前证据不支持独立失效前兆、Impending Failure Detection、safe stop、held-out task 泛化或未测试 A2 类型的检测率。

## 冻结版本

### v0.1：四周 MVP

- SmolVLA + LIBERO task 4 闭环运行；日志从 step 0 进入 rollout 路径。
- 50 条自然轨迹：18 success / 32 failure。
- 按 `initial_state_id` 冻结 30/10/10 train/validation/test split。
- 自然失败中的 self-collision/joint-violation 覆盖率为 `1/32 = 3.125%`，因此按门槛选择 Outcome Prediction，放弃 Impending lead-time 与 safe-stop 主张。
- Initial-proprio、16-step temporal MLP、冻结双相机 ResNet-50 predictor 完成。
- v0.1 test：vision-step80 与 temporal-step120 均为 AUPRC `0.982`、AUROC `0.952`。
- RGB/error、LOEO 和 false-negative 审计首先发现 execution-progress 混淆。
- 相机消融、offline early-termination analysis、risk overlay 和 21-file checksum 完成。
- RQ1 supporting study：A1 预定义案例 11/11；动作 `x/y` 置换 command-effect 检出 10/10，normal 误报 1/10；相机置换 7/10 只作间接异常响应。

### v0.2：完整性与独立确认

- Initial-state identity：50/50 recorded IDs 与 LIBERO preset index 一致；重复 ID 的 preset fingerprint 稳定。
- 移除 temporal 的 `checkpoint_step / 280` 后，同 checkpoint 的 AUPRC/AUROC 不变；该特征只影响概率尺度。
- 冻结 v0.1 模型后，在未用于原训练/选模的 preset states 30–49 采集 20 条：11 success / 9 failure。
- Frozen vision step 80：AUPRC `0.939 [0.786, 1.000]`，AUROC `0.919 [0.747, 1.000]`。
- Frozen temporal step 120：AUPRC `0.882 [0.671, 1.000]`，AUROC `0.828 [0.571, 1.000]`。
- 结论：同任务 outcome association 可复现；该实验本身不识别信号机制。

### v0.3：真实任务阶段控制

采集前冻结协议：preset states 30–49，每个状态使用 5 个新 seed，共 100 条；不因 outcome 停止、扩充或排除样本。100/100 episode 通过 sidecar/video validator，最终为 40 success / 60 failure。

`label_only` 记录两个黑碗、盘子、柜体/抽屉和任务区域位姿。Predictor 仍只读取部署可用 RGB、proprioception、action 与 timing；特权场景状态只用于事后阶段控制。

| Model | Raw AUPRC | Raw AUROC [95% CI] | Stage-only AUROC [95% CI] | Residual AUROC [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Vision step 80（100 条） | 0.926 | 0.863 [0.745, 0.957] | **0.908 [0.841, 0.959]** | **0.426 [0.260, 0.597]** |
| Temporal step 120（99 条） | 0.937 | 0.876 [0.759, 0.968] | 0.741 [0.562, 0.903] | **0.514 [0.334, 0.705]** |

控制变量包括黑碗 XYZ/总位移、黑碗到末端和盘子的距离、上层抽屉位移与夹爪位置。控制模型采用 leave-one-initial-state-out；置信区间以 20 个 `initial_state_id` cluster bootstrap。

关键阶段覆盖：

| Stage event | Success | Failure |
| --- | ---: | ---: |
| 黑碗位移超过 4 cm | 40/40 | 28/60 |
| 黑碗抬升超过 4 cm | 40/40 | 19/60 |
| 黑碗进入盘子 XY 10 cm | 40/40 | 16/60 |

同 initial-state 阶段最近配对：

| Model | Raw risk | Stage-only | Stage-residualized |
| --- | ---: | ---: | ---: |
| Vision step 80 | 13/16 | 15/16 | 7/16 |
| Temporal step 120 | 15/16 | 13/16 | 9/16 |

配对阶段距离中位数仍为 2.89/4.41 SD；原始配对准确率不能解释为严格同阶段 precursor。残差排序与随机水平一致。

## Phase 1 最终结论

1. 冻结 predictor 的 outcome association 真实且可以在新数据上复现。
2. 高 AUROC/AUPRC 主要来自“轨迹执行到了哪里”，不是经验证的未来失效信号。
3. 控制真实任务阶段后，vision 与 temporal 都没有可靠的阶段外排序能力。
4. Outcome risk 不能报告 lead time，不能驱动 safe stop，也不能称为 runtime failure detector。
5. 这是一个单任务机制结果，不支持 task/object/platform generalization。

## Phase 2 入口

Phase 2 不再扩大普通 outcome prediction，也不先升级模型。目标是：

> 在相同执行阶段下，预测未来 `K` 步内是否发生具有明确 `t_event` 的失败事件。

执行顺序：

1. 在现有 v0.3 中构造 lift-aligned 与 plate-approach-aligned cohort。
2. 审计 19 条 lifted-but-failed 和 16 条 plate-approach-but-failed 轨迹，冻结可自动标注的 grasp-loss/object-drop 事件。
3. 分开报告 progress-controlled outcome 与真正的 future-event precursor 标签。
4. 比较 progress-only、progress + state/action、progress + vision；只有增量的分组置信区间稳定高于 0，才声称存在 precursor。
5. 简单模型无残差信号时，不升级 Transformer。

## 冻结边界

- v0.1 split、test predictions、模型和阈值不得为改善结果而修改。
- v0.2/v0.3 是独立确认和机制审计，不并入 v0.1 训练。
- Outcome 与 Impending 的标签、指标和结论保持分离。
- Privileged simulator state 不进入 predictor 或 normalization。
- 不从 test/confirmation 选择模型、相机或阈值。
- 不把全零动作当作通用 safe stop。

## 产物位置

- v0.1 split：`docs/manifests/week1_task4_split.json`
- v0.2 confirmation：`docs/manifests/v02_task4_confirmation.json`
- v0.3 预声明协议：`docs/manifests/v03_task4_stage_protocol.json`
- v0.3 cohort manifest：`docs/manifests/v03_task4_stage_cohort.json`
- v0.3 冻结评测：`artifacts/results/v03_task4_frozen_stage_cohort.json`
- v0.3 阶段控制：`artifacts/results/v03_task4_stage_aligned_signal.json`
- Artifact 索引：`docs/release-artifacts.md`
- 大型 rollout、视频、NPZ、PT 与 checksum 保留在实验主机，不提交 Git。
