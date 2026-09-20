# VLA-SafeBench

VLA-SafeBench 是一个基于 SmolVLA 和 LIBERO 的闭环评测项目，用来研究机器人策略失败时，系统能够观察到什么、预测什么，以及哪些结论不能从现有数据推出。

当前核心发现是：冻结视觉特征可以较早预测 task 4 的最终成功或失败，并在 20 个从未用于训练、选模或阈值选择的 LIBERO preset states 上复现；vision-step80 的确认集 AUPRC 为 0.939、AUROC 为 0.919。RGB/error、LOEO 和定量进度控制同时表明，**任务执行进度可以解释大部分已观察到的预测能力**。独立确认集验证了 outcome association 的可重复性，但没有回答视觉是否读取了进度之外的失效信号。风险分数只用于 Outcome Prediction 和离线效率分析，不能描述为即将发生危险或安全停止依据。

## 四周执行状态

| 阶段 | 状态 | 已完成 | 尚未完成 |
| --- | --- | --- | --- |
| **Day 0–3 / Week 1** | **完成** | SmolVLA + LIBERO 闭环；从首步启用同步日志和双相机视频；20-episode 性能/成本验收；50 条 task-4 自然 rollout；事件覆盖统计；冻结 30/10/10 group-disjoint split；确定 Outcome 主线 | 无 Week-1 阻塞项 |
| **Week 2** | **完成** | Initial-proprio difficulty baseline；16-step state/action temporal MLP；冻结 ResNet-50 双相机 predictor；paired episode bootstrap；输入泄漏测试；RGB/error、LOEO 和关键 false-negative 审计；形成 execution-progress 假设 | 不再增加 Transformer 或继续调参 |
| **Week 3** | **完成** | 主/腕/双相机消融；offline utility；risk 曲线与视频；定量进度控制实验 | 无 Week-3 阻塞项 |
| **Week 4** | **完成** | 完成度和 claim audit；A1 case table；60 条 A2 配对 rollout；command-effect evaluation；21-file release checksum | 无 Week-4 阻塞项 |
| **v0.2 post-release audit** | **完成** | Initial-state identity audit；temporal checkpoint-feature ablation；20 条冻结模型独立确认集 | 不改写 v0.1 test 或选模结果 |

四周主计划已在 v0.1 收口。v0.2 是发布后的完整性审计与确认性补充：它验证原 split 和模型信号的可重复性，但不倒写成原计划内的预注册实验，也不修改 v0.1 test 结果。

## 1. 研究问题

| Regime | 含义 | 示例 | 合适的处理方式 |
| --- | --- | --- | --- |
| **A1：语法/协议故障** | 数据格式或明确约束被破坏 | shape、dtype、NaN/Inf、时间戳倒退、显式越界 | 确定性规则监控器 |
| **A2：语义配置故障** | 数值合法，但部署含义错误 | 动作维度置换、符号/单位错误、相机映射错位、chunk 语义错误 | command-effect consistency 与专门训练的故障检测器 |
| **B：自然/涌现失败** | 配置正常，策略仍未完成任务 | 偏移、停滞、误差累积、遮挡或场景歧义 | 学习式 Outcome 或 Impending predictor |

三类问题必须分开报告。容易检测的 A1 不能与困难的 A2/B 合并成一个总准确率。

项目研究四个问题：

1. **RQ1（A1/A2）：** 规则监控器能拦截多少协议错误？面对数值合法的语义配置错误，command-effect consistency 和学习监控分别能覆盖多少？
2. **RQ2a（Outcome Prediction）：** 根据当前历史，能否预测 episode 最终成功或失败？
3. **RQ2b（Impending Failure Detection）：** 对具有明确事件时刻的危险事件，能否预测未来 `K` 步内是否发生？
4. **RQ3（Utility / Control）：** 预测结果能否节省无效 rollout 计算，或在有可靠 Impending 标签时支持安全干预？

当前完成的是 **RQ2a 的单任务阶段结果和离线 efficiency 分析，以及 RQ1 的有界 supporting study**。RQ1 尚未包含 learned A2 monitor；RQ2b 和真实闭环干预尚未完成。

## 2. Outcome 与 Impending 的区别

| 任务 | 回答的问题 | 标签 | 可以报告什么 |
| --- | --- | --- | --- |
| **Outcome Prediction** | 这一局最终会成功还是失败？ | episode 最终 success/failure | AUROC、AUPRC、Brier、ECE、离线计算节省 |
| **Impending Failure Detection** | 未来 `K` 步内是否会发生明确危险事件？ | 事件时刻 `t_event` | event recall、误报率、lead time、安全干预效果 |

```text
y_t^unsafe  = 1  if 0 < t_event - t <= K  else 0
y_t^outcome = 1  if episode eventually fails else 0
```

Step 80 的 outcome risk 为 0.8，只表示模型认为整条 episode 最终失败的概率较高，不表示随后几步会碰撞。只有存在真实 `t_event` 的任务才有资格报告 lead time。

A1/A2 故障通常从 `t=0` 就存在，因此也不报告 lead time。它们应报告前 `N` 步检出率、正常对照误报率和首次报警步数。

## 3. 系统与数据边界
当前平台为 **SmolVLA + LIBERO `libero_spatial` task 4**。这里的 LIBERO 提供语言条件操作任务、视觉观测、连续控制接口、可重复的初始状态和自动成功判定；robosuite/MuJoCo 提供闭环仿真。项目使用 LeRobot 的环境适配与 SmolVLA 预处理链路，但训练监控器的数据来自本项目实际执行策略后采集的 rollout，并非直接使用 LIBERO 的离线示范作为失败预测数据。

### 为什么使用 LIBERO-Spatial

LIBERO 官方原始 benchmark 包含四组 task suite：Spatial 侧重空间关系变化，Object 侧重操作对象变化，Goal 侧重任务目标变化，LIBERO-100 则包含知识因素相互交织的 100 个任务，并划分为 LIBERO-90 和 LIBERO-10。后续 VLA 评测中常把包含长程任务的 LIBERO-10 称为 LIBERO-Long；它与前三组控制单一变化来源的设计目的不同。
本项目没有进行跨 suite 性能比较，选择 Spatial 是四周 MVP 的预先范围控制，不能据此声称它优于其他 suite。

Spatial 适合当前研究的原因是：先固定在一个以空间关系为主要变化来源的 suite 中，可以减少跨物体类别、跨目标语义和长程子任务结构同时变化造成的混杂。环境控制频率为 20 Hz；离线仿真无需等待真实时间，在 RTX 4090 上实测采集吞吐约为 30–33 个仿真步/秒。需要注意，**suite、task 和 initial state 是三个不同层级**：

- `libero_spatial` 是 suite，规定这一组任务主要考察空间关系；
- task 4 是默认 task order 下的一条固定语言任务：从木柜顶层抽屉中取出黑碗并放到盘子上；
- initial state 是执行同一 task 4 时的一种起始场景配置。

因此，held-out initial-state 评测指：训练、validation 和 test 使用同一 task 4，但按 `initial_state_id` 隔离不同起始配置，使模型不能在训练中见到 test 的初始状态。v0.2 审计直接读取 LIBERO wrapper 的 preset-state 游标，50/50 条记录的 ID 均与 reset 前真实 `init_state_id` 一致，同一 ID 在两次 collector session 中也始终对应同一个 preset fingerprint。它只检验同一任务内对未见初始配置的适应能力；其他 LIBERO suite 同样可以这样切分，这不是 Spatial 独有的性质，也不构成 held-out task 或跨 suite 泛化。

### 为什么最终使用 task 4

先对 task 1–9 各运行一条初筛，再对有代表性的候选扩测：task 5 连同初筛共 6/6 失败，过难；task 7 连同初筛共 6/6 成功，过易；task 4 的五条扩测为 1/5 成功、4/5 失败，首次确认了同一任务内的混合 outcome。因此正式采集选择 task 4，而没有把一个全成功任务和另一个全失败任务混合，否则 predictor 可能仅凭 task identity 完成分类。最终 task-4 自然 cohort 为 50 条，其中 18 条成功、32 条失败。

Rollout 路径从第一步开始同步记录：

- 主相机和腕部相机 RGB；
- proprioception；
- VLA 预测 action chunk 与实际执行动作；
- observation、inference 和 control 时间戳及延迟；
- task、seed、initial state、Git revision 和 checkpoint digest；
- success、终止原因及 label-only simulator diagnostics。

每个 episode 使用稳定的 episode/step/frame ID。Recorder 采用 `.incomplete` 到 `COMPLETE` 的原子完成协议；validator 检查必填字段、有限数值、连续 ID、时间戳和 result/JSONL/video 对齐。

Predictor 只允许读取真实部署可得的 RGB、proprioception、历史动作和 timing metadata。物体真值位姿、完整 MuJoCo 状态、接触力、joint margin 和碰撞标签只能用于生成标签或事后分析，不能进入 predictor 输入或归一化统计。

## 4. 冻结数据协议

当前 task 4 cohort 包含 50 条自然 rollout：

| Split | Episodes | Success | Failure |
| --- | ---: | ---: | ---: |
| Train | 30 | 11 | 19 |
| Validation | 10 | 4 | 6 |
| Test | 10 | 3 | 7 |
| **Total** | **50** | **18** | **32** |

切分按 `initial_state_id` 分组，确保同一 initial state 不跨 split。Manifest 在生成 step 0/40/80/120 的样本之前冻结，test 不参与模型、超参数或阈值选择。

自然失败中，已实现的 self-collision 或 joint-violation 事件只覆盖 `1/32 = 3.125%`，validation/test 均无 event-positive episode。按照预设的 20% 数据门槛，项目选择 **Outcome Prediction**，停止 Impending lead-time 和 safe-stop 主张。Workspace violation 和 impact 尚未形成冻结协议，不能记作已验证的零事件。

## 5. 方法与基线

五个模型不是互相竞争的方案，而是一条递进的排除链：每一层都用来排除一种"看起来有效、实际另有解释"的可能。只有逐层通过，才能说预测能力来自真正的执行过程信号。

### 第 1 层：Train-prevalence constant

对所有 episode 输出同一个数——训练集失败率 `19/30 = 0.633`。

**排除什么**：类别不平衡本身就能得到的分数。test 是 7 失败 / 3 成功，这个常数模型能拿到 AUPRC 0.700，但 AUROC 只有 0.500（完全没有排序能力）。这提醒读者：任何模型的 AUPRC 都要减掉 0.700 这个起点来看。

它与 always-failure classifier（恒输出 1.0）的排序指标相同，但输出 0.633 更接近真实基率，校准更合理。

### 第 2 层：Initial difficulty baselines

两个变体：initial-proprio 只读第 0 步的机器人状态；frozen vision 的 step-0 结果只读初始双相机画面。

**排除什么**："成败在开始那一刻就已注定"。如果初始场景就能高精度区分，说明执行过程没有产生新信息，"过程中的失败预测"这个研究问题在这批数据上不成立——该换任务，而不是换模型。

### 第 3 层：Visual progress proxy

用 train-only、不含 outcome label 的冻结视觉特征时间方向，估计"画面看起来执行到了哪里"。

**排除什么**："模型只是在读执行进度"。这是整条链里最关键的一层——如果 predictor 的排序能力能被单纯的进度估计复现，那它就没有学到进度之外的任何东西。

需要说明的是，这个 proxy 不是 `step / 280`。在同一个 checkpoint 上所有轨迹的 step 编号完全相同，无法产生任何排序；它估计的是视觉上的完成程度，同一时刻的不同轨迹可以有不同的进度。

### 第 4 层：State/action temporal MLP

将最近 16 步的 proprioception、动作和 timing 历史展平，输入一个小型 MLP，不使用画面。

具体输入为每步 25 维 proprioception、7 维实际执行动作和 2 维 timing，共 16 步；padding mask 与 `checkpoint_step / 280` 也作为显式特征。网络为 `input → 128 → 64 → 1`，隐藏层使用 ReLU 和 0.1 dropout，最终概率取 5 个固定训练 seed 模型的平均。v0.2 单独消融了 checkpoint 特征，确认它不产生同 checkpoint 内的排序能力。

**回答什么**：抛开视觉，光凭机器人自身的运动学轨迹能否判断成败。如果能，说明失败信号已经体现在本体状态层面。

### 第 5 层：Frozen dual-camera vision

用冻结的 `torchvision/resnet50 IMAGENET1K_V2` 提取主／腕相机特征，再训练 checkpoint-specific linear probe。

每个相机产生 2048 维特征，双相机拼接为 4096 维；step 0、40、80、120 各自训练一个线性概率头。ResNet 权重全程冻结，L2 候选只按 validation AUPRC 选择，并以 validation Brier 处理并列。

**回答什么**：视觉是否提供了本体状态之外的信息，尤其是更早的信息。

冻结 encoder、只训线性层是小样本下的必要约束：训练集仅 30 个 episode，若让 ResNet-50 的两千多万参数一起训练，模型会直接记住这些 episode。这个设计测的是"预训练视觉特征中是否已包含 outcome 信息"。

### 关于 progress proxy 的引入时机

第 1、2、4、5 层属于原始训练与评测流程；progress proxy 是在机制审计阶段才引入的。本文将它提升到正式 baseline 位置以补全证据链，**但不将其倒写为预注册设计**。

### 评测约束

模型选择和 L2 正则只使用 validation，test 只评一次。所有差值按完整 episode 做 paired bootstrap，不把同一 episode 内相关的滑动窗口当作独立样本——否则有效样本量会被严重高估。

AUROC 衡量总体排序，AUPRC 在 failure/success 不平衡时强调失败类的 precision-recall；两者都不能说明输出的 `0.8` 是否真能解释为 80% failure probability。因此同时报告 Brier score 衡量概率平方误差，并以 ECE 辅助检查置信度与经验失败率是否一致。Brier/ECE 越低越好；ECE 依赖分箱且当前样本很少，只作为辅助证据，不单独用于模型选择或结论升级。

### v0.2 发布后审计方法

v0.2 不重新打开模型选择，而是针对 v0.1 的三个关键疑点做固定协议验证：

1. Initial-state identity audit：确认数据切分真的按初始状态隔离。
按原 collector session 顺序重放 50 条 v0.1 episode，在每次 reset 前直接读取 LIBERO wrapper 的真实 init_state_id 和 preset-state fingerprint，并与原视频首帧核对。
**目的：**确认 train / validation / test 中记录的 initial state 身份没有错，held-out initial-state split 是真实成立的。完整 MuJoCo state 只用于审计，不进入 predictor。

2. Temporal checkpoint-feature ablation：确认 temporal MLP 不是靠“当前做到第几步”作弊。
原 temporal MLP 输入中包含 checkpoint_step / 280。现在保持数据、split、16-step history、模型结构、优化器、训练 seed 和 early stopping 全部不变，只删除这一项，再与原模型在同一批 test episodes 上比较。
**目的：**判断原来的排序能力到底来自 state/action history，还是主要由显式 checkpoint 信息制造出来。这个实验只用于解释模型依赖，不用 test 结果重新选模型。

3. Independent initial-state confirmation：确认原来的 outcome signal 不是 10 条 test episode 偶然得到的。
冻结 v0.1 已训练好的 temporal / vision predictor、normalization、相机配置、checkpoint 和评测指标，然后在 task 4 从未使用过的 preset states 30–49 上重新采集 20 条 rollout，并直接评测。
**目的：**检查原来观察到的 outcome prediction 能力，能否在新的 held-out initial states 上再次出现。确认集不参与训练、选模、阈值选择或校准，也不并入原 test。

这三项依次回答：原 split 的 initial-state 身份是否真实、temporal 排序是否由显式 checkpoint 特征制造、原 outcome association 能否在独立 initial states 上复现。它们不回答 held-out task 泛化，也不把 Outcome Prediction 升级为 Impending Failure Detection。

## 6. 主结果与审计结论

### 6.1 v0.1 主结果

下表只回答一个问题：在冻结的 10 条 held-out initial-state test episodes 上，predictor 能否把最终失败排在成功之前？每格依次为 **AUPRC / AUROC**。

| Method | Step 0 | Step 40 | Step 80 | Step 120 |
| --- | --- | --- | --- | --- |
| Train-prevalence constant | 0.700 / 0.500 | 0.700 / 0.500 | 0.700 / 0.500 | 0.700 / 0.500 |
| Initial-proprio difficulty | 0.799 / 0.452 | 0.799 / 0.452 | 0.799 / 0.452 | 0.799 / 0.452 |
| State/action temporal MLP | 0.609 / 0.286 | 0.856 / 0.571 | 0.909 / 0.762 | **0.982 / 0.952** |
| Frozen dual-camera vision | 0.652 / 0.190 | 0.856 / 0.571 | **0.982 / 0.952** | **0.982 / 0.952** |

AUPRC 的无排序信息起点等于 test failure prevalence，即 `7/10 = 0.700`；常数分数的 AUROC 为 `0.500`。这里的 constant probability 使用训练集失败率 `0.633`，不是始终输出 `1.0` 的 always-failure classifier。

Paired episode bootstrap 支持 vision-step80 相对 initial-proprio 的排序改善：AUPRC `+0.183 [0.031, 0.450]`，AUROC `+0.500 [0.125, 0.857]`。Temporal-step120 的对应差值为 AUPRC `+0.183 [0.033, 0.450]`、AUROC `+0.500 [0.143, 0.860]`。

完整的 Brier/ECE、逐 checkpoint bootstrap 和预测概率保留在冻结 result JSON 中。它们用于检查概率质量，不改变这里的主排序结论。

### 6.2 v0.2 独立确认集

完成 v0.1 后，项目冻结原 predictor、normalization、checkpoint 和指标，在 task 4 尚未使用的 LIBERO preset states 30–49 上重新采集 20 条 rollout（11 success、9 failure）。这批数据不参与训练、选模或阈值选择，也不并入原 test。

| Frozen predictor | Checkpoint | AUPRC [95% CI] | AUROC [95% CI] |
| --- | ---: | ---: | ---: |
| Dual-camera vision | 80 | **0.939 [0.786, 1.000]** | **0.919 [0.747, 1.000]** |
| State/action temporal MLP | 120 | **0.882 [0.671, 1.000]** | **0.828 [0.571, 1.000]** |

确认集失败率为 0.45，因此无排序信息的 AUPRC 起点为 0.45、AUROC 为 0.50。结果支持“同一 task 内、未见 preset initial states 上存在可复现的 outcome signal”，降低了原 10 条 test 偶然产生高分的可能性。

确认集只复评了两个冻结 learned predictor，没有重新拟合 initial-proprio baseline 或 progress proxy。它回答的是“原 outcome 排序能否在新 initial states 上复现”，而不是重复机制识别实验；关于模型主要读取执行进度的判断仍来自 v0.1 train/test 上的 RGB、LOEO 和 progress-control 审计。

### 6.3 审计结论

| 审计问题 | 关键结果 | 结论 |
| --- | --- | --- |
| 模型是否主要读取视觉执行进度？ | Progress-only AUROC：vision-step80 `0.905`；temporal-step120 `1.000` | 执行进度可以解释大部分排序能力；是否存在进度外视觉信号仍无法确定。 |
| Temporal 排序是否由 `checkpoint_step / 280` 制造？ | 移除后 step 80/120 的 AUPRC、AUROC 均不变 | 显式 checkpoint 特征不产生同 checkpoint 内的排序，只影响概率尺度。 |

因此保留的主张是：**冻结 predictor 能在同一 task 的未见 initial states 上复现 outcome 排序，但现有证据不能把它解释为独立失效前兆。** 完整进度控制、匹配对和置信区间见下一节；完整校准与消融数值见冻结 artifacts。

## 7. 模型实际读取了什么

最初的表面结果是：vision 在 step 80 达到了 temporal MLP 在 step 120 才达到的排序指标。RGB/error、leave-one-episode-out 和逐 episode 审计限制了这一解释：

- 成功轨迹通常在 step 122–160 结束，中位数约为 133；所有失败轨迹都运行到 step 280 超时。
- Step 80 的低风险画面通常更接近成功构型，高风险画面通常进度较慢。
- Vision-step80 与 temporal-step120 错排同一对 failure/success，概率高度相关（Pearson `0.9998`，Spearman `0.9515`）。
- 关键 false negative 在 step 80 看起来仍顺利推进，之后进入近完成构型，却持续微调至 step 280 超时；可靠分叉约在 step 120–160。

定量进度控制使用 train-only、无 outcome label 的 frozen-feature 时间方向作为 progress proxy。结果如下：

| Model | Raw AUROC | Progress-only AUROC | Progress-residualized AUROC | Partial correlation |
| --- | ---: | ---: | ---: | ---: |
| Vision, step 80 | 0.952 | 0.905 | 0.714 | 0.552 |
| Temporal, step 120 | 0.952 | 1.000 | 0.381 | -0.202 |

Temporal-step120 的排序几乎完全可由进度代理解释。Vision-step80 的残差点估计仍为正，但 residual AUROC 的 95% CI 为 `[0.111, 1.000]`，partial correlation CI 为 `[-0.232, 1.000]`，无法排除无额外信号。

进度匹配进一步暴露数据重叠不足：vision-step80 只有 2 对 success/failure 的距离小于 0.5 个 train SD（两对均排对），第三对相差 2.0 SD；temporal-step120 的三对距离为 1.61–2.08 SD，没有真正接近的匹配对。`2/2` 只是描述性结果，不能据此声称视觉学到进度之外的信息。

因此最终机制结论是：**执行进度可以解释大部分已观察到的 outcome 排序能力；现有 10 条 test episode 不足以判断 vision-step80 是否还包含进度之外的信息。**

## 8. Supporting results

### 相机消融

| Input at step 80 | AUPRC | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Main only | 0.844 | 0.667 | 0.306 | 0.330 |
| Wrist only | 1.000 | 1.000 | 0.001 | 0.017 |
| Dual camera | 0.982 | 0.952 | 0.100 | 0.096 |

Wrist 和 dual 相对 main 的 Brier/ECE 配对区间支持改善；wrist 与 dual 的关键区间触及或跨过 0。正式结果继续使用预注册的 dual-camera pipeline，不根据 test 改选模型。

### Offline decision utility

Validation 在零 sacrificed-success 约束下选择阈值 `0.9998072982`，随后固定到 test：

| Split | Detected failures | Sacrificed successes | Saved steps | Estimated wall time |
| --- | ---: | ---: | ---: | ---: |
| Validation | 5/6 | 0/4 | 1,115 | 78.10 s |
| Test | 4/7 | 0/3 | 756 | 52.95 s |

Test 中 3 条失败在 step 80、1 条在 step 120 首次触发。时间按实测 `70.042 ms/step` 换算。该结果只是已记录轨迹上的反事实效率上界，不能当作通用阈值、闭环 safe stop 或安全改进证据。

### A1/A2 部署故障诊断

A1 使用 11 个预定义 deterministic cases，覆盖合法动作、shape、dtype、NaN/Inf、上下界和时间戳类型/负值/倒退，结果为 **11/11**。这是对所列案例的完整覆盖，不是对所有可能协议错误的统计泛化率。

A2 使用 20 组相同 seed 和 initial state 的三路配对 rollout：normal、`action_swap_xy` 和 `camera_swap`。前 10 组 normal 只用于冻结三步 command-effect 窗口阈值 `0.259284`，后 10 组用于一次性评测；三种 regime 都只统计前 40 步。

| Evaluation regime | Task success | Protocol rule | Command-effect | 95% Wilson CI | Median first alarm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 6/10 | 0/10 | 1/10 | [0.018, 0.404] | 36 |
| `action_swap_xy` | 0/10 | 0/10 | **10/10** | [0.722, 1.000] | **3** |
| `camera_swap` | 0/10 | 0/10 | 7/10 | [0.397, 0.892] | 29 |

协议规则对两类 A2 都是 0/10，说明合法数值的语义错误会通过 A1 schema/range checks。Command-effect 对动作维度置换有直接诊断意义；`camera_swap` 的 7/10 只表示错误视觉使闭环进入异常 command/effect 分布，不能解释为相机映射分类能力。Range clipping warning 在 normal 为 10/10、动作置换为 9/10、相机置换为 1/10，因此不能作为语义故障报警器。

## 9. 当前状态与下一阶段

2026-09-14 最终 release checksum 包含 21 个文件并通过 21/21 校验：原 18-file Outcome v0.1 快照、进度控制报告和两个 RQ1 报告。大体积配对 rollout 留在实验主机，并由 A2 report 中的 episode ID、seed 和逐条结果索引。

2026-09-20 的 v0.2 checksum 另列 10 个文件，覆盖 confirmation manifest、完整性审计、temporal 消融、确认集 dataset/feature metadata 与冻结评测结果；大型 NPZ/PT 仍只通过 SHA-256 追踪，不并入 Git。

进度控制实验已经完成，结果属于“当前样本不足以区分”：进度解释得到定量支持，vision 的额外信息没有得到可靠统计证据。该结果已纳入最终文档和 release。

Week 4 的 RQ1 supporting study 已完成：A1 case table 为 11/11；A2 的协议规则无法识别两类数值合法故障，command-effect 在 10 条 evaluation pairs 中对动作 `x/y` 置换检出 10/10，正常误报 1/10。相机置换结果只作为间接异常响应报告。本阶段不训练专门的 learned A2 monitor。

v0.2 已完成 20 条独立 initial-state confirmation cohort，并以冻结 v0.1 predictor 一次性评测。第二任务、Transformer、π0.5、RoboTwin、recovery 和 adaptive chunking 不进入当前阶段。

### 原始目标完成度核对

| 原始目标 | 状态 | 证据或缺口 |
| --- | --- | --- |
| SmolVLA + LIBERO 闭环 | **完成** | task 4 正式 rollout 与同步 artifact |
| 日志从第一条 rollout 进入执行路径 | **完成** | 稳定 episode/step/frame ID、原子完成协议 |
| 50 episodes / 20 failures gate | **完成** | 50 episodes，32 failures |
| Outcome/Impending 数据分流 | **完成** | event coverage 1/32，选择 Outcome |
| 冻结无泄漏 split | **完成** | 30/10/10，按 initial state 分组 |
| 强 baseline 与 learned predictor | **完成** | initial-proprio、temporal、frozen vision |
| Held-out 指标、校准与 bootstrap | **完成** | 10 条原 test + 20 条独立 confirmation；机制控制仍受小样本限制 |
| 机制审计与进度控制 | **完成** | RGB/error、LOEO、残差化和进度匹配 |
| 风险叠加视频 | **完成** | outcome-risk offline overlay |
| 可复现 release | **完成** | 21-file checksum 21/21 |
| v0.2 审计与独立确认 | **完成** | 20 条未见 preset states；10-file checksum 清单 |
| 2–3 个正式任务 | **未完成** | 当前只有 task 4，不声称 task generalization |
| A1 运行时规则覆盖 | **完成（有界案例集）** | 11/11 deterministic cases；不外推到未测试协议错误 |
| A2 / command-effect consistency | **完成（supporting study）** | 20 组配对 cohort；动作置换 10/10、normal 误报 1/10；相机结果仅为间接响应 |
| Impending detector / safe stop | **不适用当前主线** | event-positive 数据不足，按门槛主动放弃 |

## 10. 复现

完整 artifact 索引与命令见 [`docs/release-artifacts.md`](docs/release-artifacts.md)。

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

大规模 rollout、视频、模型权重和 feature cache 不提交 Git，通过 manifest、artifact index 和 SHA-256 清单追踪。

## 11. 已知限制

- 当前正式数据只覆盖一个 LIBERO task。
- 原 test 只有 10 条 episode，7 条失败全部是 280-step timeout；20 条 confirmation 增强了复现证据，但没有扩大 task 范围。
- Outcome predictor 没有明确 failure timestamp，不能报告 unsafe lead time。
- Self-collision/joint-violation 标签覆盖不足，workspace/impact 尚未冻结。
- RQ1 只测试两类 A2 故障、一个任务和 10 组 evaluation pairs，且不包含 learned A2 monitor。
- Offline utility 没有执行真实闭环干预。
- 当前结果不能支持 held-out task、物体或平台泛化主张。

## 12. 项目定位

这是一个关于 VLA 失败预测的**单任务方法论研究**。核心贡献是泄漏受控的闭环数据与评测流程、冻结模型的独立 initial-state 复现，以及两次基于证据的结论收缩：先因 unsafe-event 覆盖不足从 Impending 转向 Outcome，再因机制审计撤回“独立失效前兆”的解释。项目不把当前结果包装成可泛化的失败检测方法。
