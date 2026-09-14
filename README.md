# VLA-SafeBench

*Learning-Based Failure Prediction and Risk-Aware Closed-Loop Control for VLA Policies*

**项目实施方案（修订版）**

> **项目一句话：** 在 LIBERO 等仿真环境中，围绕 VLA 闭环执行，将故障区分为「语法/协议故障」「语义配置故障」与「自然/涌现失败」：前者用规则监控器拦截，后两者结合 command-effect consistency 与学习式 predictor 分析。第 1 周根据自然失败中的明确 unsafe-event 覆盖率，在 Outcome Prediction 与 Impending Failure Detection 中选择唯一主线；只有后者的数据条件成立时，才用近期风险驱动 safe stop。

**修订说明（相对初稿的关键改动）**

- 把失败分成三类：语法/协议故障由规则处理；数值合法但语义错误的配置故障不能假设规则必胜；自然涌现失败是学习模型的主战场。
- 重新定义失败标签与任务：将 Outcome Prediction 和 Impending Failure Detection 分开评测；前者预测最终结果，后者只预测具有明确事件时刻的近期风险，不再人为构造 point-of-no-return。
- 新增评测协议：核心泛化只做 held-out episode / seed / initial state，并按 A1/A2/B 三种 regime 分别报告；held-out task 与 held-out 故障类型仅作探索性结果。
- adaptive chunking 降为 stretch goal；澄清高风险下唯一可靠动作是 stop，"replan" 需真正的 recovery。
- 明确禁止 predictor 使用仿真特权状态；特权信息只能生成标签与分析失败。
- 明确 safe stop 取决于控制语义，不能把全零动作普遍当成停止。
- 简历表述改为按 regime 诚实陈述，不预设"全面超越规则"。

## 0. 项目概览

**当前状态（2026-09-14）：** Outcome Prediction MVP 的核心实验已完成。项目冻结了 task 4 的 50-episode cohort、group-disjoint split 与 predictor 输入边界；完成 initial-proprio、state/action temporal MLP 和 frozen dual-camera vision 对照、相机消融、错误案例/LOEO 审计、offline decision utility 与风险叠加视频。Vision 在 step 80 显著超过初始难度基线，但现有证据表明它主要读取 outcome-associated execution progress，不能解释为独立的早期失效前兆。当前只做主表、文档与 release 收口；A1/A2 RQ1 和 Impending Failure Detection 尚未评测，不进入本次主张。

**范围说明：** 本项目保留完整研究愿景，但把工作拆成可独立交付的层级。MVP 缩小不等于删除最终方向：π0.5、多故障类型、Transformer、双臂 RoboTwin、recovery 与 adaptive chunking 均保留在后续阶段，只有在前置证据成立后才启动。

### 4 周主交付（唯一验收口径）

四周内不追求填满 A1/A2/B × rule/consistency/learned 的整张矩阵，只交付：

1. **一条由第 1 周数据决定的主线；**
2. **一张带 held-out episode/seed/initial-state 与校准指标的主结果表；**
3. **一个叠加风险曲线、事件时刻与干预时刻的失败预警视频；**
4. **一个可复现 release。**

A1 单元测试、A2 stress test 和另一预测任务只能作为 supporting result；如果主线数据不支持，不得为了保留原计划而继续铺矩阵。

### 总体架构
```text
                     ┌──────────────────────────────┐
RGB / proprio / task │                              │ action chunk
────────────────────▶│        Frozen VLA           ├──────────────┐
                     └──────────────────────────────┘              │
                                                                   ▼
┌──────────────────┐   synchronized rollout   ┌────────────────────────┐
│ Simulator        │◀────────────────────────▶│ Recorder / Fault Injector│
│ LIBERO / RoboTwin│                          └────────────┬───────────┘
└────────┬─────────┘                                       │ history
         │ privileged labels only                          ▼
         │                                  ┌─────────────────────────┐
         └─────────────────────────────────▶│ Rule + Learned Monitor  │
                                            └────────────┬────────────┘
                                                         │ calibrated risk
                                                         ▼
                                            execute / safe stop / recovery
```

仿真特权状态只沿“labels only”路径流动，不能进入部署侧 predictor。

---

## 1. 项目定位

**目标岗位：** VLA 算法工程师 / VLA 系统算法工程师。

**核心差异：** 不是在冻结策略外套阈值规则，而是把「闭环失败诊断」拆成三种机制不同的问题，并把其中没有简单规则可解的部分——语义配置错误与自然/涌现失败预测——做成可训练、可泛化、可用于执行决策的 ML 问题。

能力覆盖：

- **系统：** VLA serving、仿真 rollout、action chunk、控制接口、日志与视频、故障注入与 unsafe-event 插桩。
- **算法：** 监督数据构造、时序/多模态 failure predictor、消融、泛化协议设计、风险分数建模。
- **控制：** 按控制语义执行 safe stop；recovery / 动态 chunk 作为后续扩展。

## 2. 失败分类（本项目的核心概念）

VLA（视觉-语言-动作模型）在仿真里执行任务的时候，会失败。这个项目想搞清楚，能不能提前发现"这次要失败了"，甚至提前让机器人停下来。

但"失败"这个词太笼统了，项目一开始就把失败拆成了三种，因为这三种失败的应对方式完全不同：

**A1.语法/协议故障：**比如动作的维度对不上、数值出现了 NaN、时间戳乱了。这种问题用简单的规则就能抓住，不需要训练模型。
**A2.语义配置故障：**数值本身合法，但意思配错了。比如相机接反了、单位弄混了（角度当成了弧度）、动作是"绝对位置"还是"相对位移"搞反了。这种问题规则不一定能抓住，因为从数值上看一切正常。
**B.自然/涌现失败：**没有人为制造任何错误，policy 自己把任务做砸了。比如抓偏了、没放稳、遮挡导致判断错了。这才是训练模型去预测的主战场。

**关键结论：**
- A1 类上规则便宜、可靠，学习模型不该争功；A2 类不能预设规则接近满分，因为错误可能满足所有数值约束；学习模型的主要价值仍限定在 B 类——“提前预测一个原本正常运行的策略正在走向失败”。
- 三类结果分别报告，绝不混成一个 F1。如果把这三种混在一起算一个总的准确率，简单的语法故障会因为规则很容易抓到而拉高整体分数，掩盖模型在真正困难的"自然失败"上表现平庸的事实。

## 3. 研究问题

- **RQ1（A1/A2 类）：** 最基础的监控手段——写死的规则——能不能用？具体来说，规则能不能完整拦截语法/协议故障？面对那些数值合法但语义配错了的故障，规则加上一种叫"command-effect consistency"（命令-效果一致性检查）的方法，再加上训练出来的模型，各自能抓住多少？

command-effect consistency 说白了就是"说到做到"检查：机器人收到一个"往右移"的指令，就应该真的往右移动。如果动作和效果对不上（比如根本没动、或者动错方向了），就报警。它的死穴是：如果 VLA 因为某个 bug 看错了相机画面，但机械臂依然精确执行了"根据这张错误画面算出来的动作"——命令和效果是一致的，这套方法完全发现不了问题，因为它只检查"做没做到"，不检查"做的这件事本身是不是基于正确信息"。

- **RQ2a（Outcome Predictor）：** 预测最终是成功还是失败

给定当前历史，模型能否预测 episode 最终成功/失败？它提供的是 outcome risk，不自动等价于“即将失败”。

- **RQ2b（Impending Failure Detection，核心）：** 对于那些有明确发生时刻的具体危险事件（比如自身碰撞），模型能不能比手工设定的阈值更早、更准地预测"接下来几步之内会不会发生"？(这叫Impending Failure Detection)

- **RQ3（控制）：** 由 impending-risk 驱动的 safe stop 能否降低 unsafe event，同时把误停与被牺牲的成功 episode 控制在可接受范围？Outcome score 在 MVP 中只做诊断，不直接触发停止。

如果真的能提前预测出"接下来要出危险了"，那拿这个预测去主动让机器人停下来，能不能既降低危险，又不要误停太多、不要把本该成功的任务也搅黄？

- **RQ4（核心泛化）：** predictor 能否迁移到 held-out episode、seed 与 initial state，而不是记住 rollout？Held-out task、物体与 A2 故障类型仅作探索性结果，不在 2–3 个任务上声称 task-level generalization。

## 4. 数据与标签构造

数据全部来自 VLA 在仿真中的 rollout，不依赖额外人工采集。

数据来源：

- **B 类自然数据（主）：** 未改动策略，在多任务、多 seed 下大量 rollout，自然收集成功与失败 episode。为拿到足够多的自然失败，每个任务跑足够 seed，并可优先选策略本就不稳的任务/初始化。
目前只完成了task4这一个正式cohort

- **A1/A2 类故障数据（辅,还没有做）：** 注入 §5 的部署错误，用于 stress-test 规则监控器、一致性检测与探索性泛化评测。

每步记录：RGB、proprioception、action、action chunk、时间戳、推理延迟、仿真状态、接触/碰撞信号、成功标记、视频。

**信息边界：**
关于哪些信息能给模型用、哪些不能，这里有一条硬性边界：模型只能读取真实部署时真的能拿到的东西——画面、本体状态、过去的动作历史、时间信息。仿真器才知道的"作弊信息"——比如物体的精确三维坐标、仿真引擎的完整内部状态、精确的接触力——只能用来生成标签、拿来做失败分析，绝对不能进入模型的输入，也不能用来做数据归一化。一旦这条线被打破，后面所有关于"这个模型能不能泛化"的结论都作废。

**标签定义（对应前面第二个问题的两个版本，千万不能混用）：**

1. **Impending Failure Detection / unsafe-event 标签（有时刻）：**
 - 一种标签是"接下来 K 步内会不会发生一次具体的、有明确时刻的危险事件"。这类事件只用容易自动、且和具体任务无关的判定标准——自身碰撞、关节超出限位、明显的撞击——因为这些有一个确切的"发生时刻"，才能问"提前了多久预测到"这种有意义的问题。
 - MVP 不做“unexpected contact”及其任务相关 whitelist。

2. **Outcome Prediction / episode-outcome 标签（episode 级）：**
另一种标签是"这一局最终是成功还是失败"，这是整个 episode 只有一个固定答案的标签，不随时间变化。

**重要约束：** Outcome Prediction 很可能学到“这个任务/初始场景有多难”，即使指标很好，也不能据此声称实现了 runtime failure detection。只有带明确事件锚点、预测窗口 K 和 lead time 的 Impending Failure Detection 才回答“是否即将发生故障”。两者可以共享 encoder，但必须使用独立 label、输出、结果表和结论。

A1/A2 类注入故障多数从 t=0 起坏，不对它报 lead time——那只会得到虚高的提前量；这两类只报“是否在前 N 步内检出”。lead time 只对有明确 `t_event` 的 unsafe event 报告。Outcome Prediction 报告随执行进度变化的 AUROC/AUPRC 与校准度，不声称存在真实 failure timestamp。

形式化定义如下；MVP 可用同一时序 encoder 加两个独立 head，但训练损失和评测必须分开：

$y_t^{\text{unsafe}} =
\begin{cases}
1 & \text{if } 0 < t_{\text{event}} - t \le K \\
0 & \text{otherwise}
\end{cases}
\qquad
y_t^{\text{outcome}} =
\begin{cases}
1 & \text{if episode eventually fails} \\
0 & \text{otherwise}
\end{cases}$

### 4.1 第 1 周必须测出的分流统计

在任何 predictor 训练前，先对自然 rollout 计算：
`p_event_given_failure` = `自然失败且至少触发一次已定义 unsafe-event 的 episode 数`
    / `自然失败 episode 总数`

**为什么要先算这个再决定方向：**如果这个比例够高，说明"失败"和"危险事件"高度相关，值得往更有野心的方向（提前预测具体危险）去做；如果比例很低，说明大部分失败根本不是什么惊险的事故，做不了那个更难的目标，应该退而求其次，只做"预测最终成功失败"这个更朴素的版本。

同时报告分子、分母、各事件类型计数和任务分布，不能只报百分比。这个统计决定项目主线：

- **Pilot 最低样本：** 至少 50 个自然 episode，且至少观察到 20 个自然失败；达不到 20 个失败说明所选任务/checkpoint 不适合该研究，先换任务而不是训练。
- **Safety / Impending 主线：** 仅当 `p_event_given_failure ≥ 20%`，并且按实测速率预计能在 48 GPU-hours 内获得至少 50/15/20 个 event-positive train/val/test episode 时进入。主表为 `Impending Detector vs Strong Rule Baseline`，Outcome 仅作辅助分析。窗口指标必须按 episode bootstrap，不把相关窗口当独立样本。
- **Outcome 主线：** 若 `p_event_given_failure < 20%`，或预计无法在预算内达到上述 event-positive episode 数，则立即停止 unsafe lead-time 主张。主表改为 `Outcome Predictor vs Difficulty/Temporal Baselines`，不再声称 runtime safety warning，也不以 Outcome score 驱动 safe stop。
- **禁止做法：** 不为了挽救 Safety 叙事而临时加入模糊、任务相关的“unexpected contact”标签；unsafe 定义变更必须形成新版本协议并重新采集/评测。

第 1 周报告必须明确写出最终选择及被放弃的主张。两条主线只能选一条作为四周项目的核心结论。

**Week-1 实际分流（2026-09-11）：**
正式收集了 50 个 episode（18 个成功、32 个失败）。已经能检测的两类危险事件（自身碰撞、关节超限）里，32 个失败里只有 1 个伴随着这样的事件，比例是 3.125%，远低于 20% 的门槛。而且唯一那一个"带危险事件"的样本恰好落在训练集里，验证集和测试集里一个都没有——这意味着就算侥幸凑够训练数据，也完全没法在没见过的数据上验证这个方向到底行不行。所以项目正式选择了"预测最终成功失败"这条更朴素的路线，放弃了"提前预警具体危险"和"安全停止"这两个原本更有野心的目标。

需要额外说明的是：还有两类危险事件（工作空间越界、明显撞击）到现在都还没有开始检测，不是"测了发现没有"，而是压根没测，不能把这两类也算作已经验证过的"零事件"。

## 5. Fault Injection：部署故障分类（A1/A2 类，stress-test 轴）

| 故障类型 | 示例 | 本质 |
| --- | --- | --- |
| Action schema | 动作维度置换 / 某一维错位 | schema mismatch |
| Unit | 角度与弧度混用 | unit mismatch |
| Control semantics | absolute 与 delta 混用 | controller semantics mismatch |
| Observation mapping | 错误相机 / 相机顺序错位 | observation mismatch |
| Chunk protocol | 训练与部署执行长度不一致 | execution protocol mismatch |
| Latency | 推理延迟、网络抖动 | closed-loop timing mismatch |
| Dropped / stale obs | 丢帧、使用旧 observation | asynchronous feedback |

**说明：** 以上均为单臂 LIBERO 上可注入的故障，已去掉对单臂环境不适用的双臂“左右臂交换”。其中 shape/dtype/NaN 等 A1 故障由规则单测覆盖；表中多数属于 A2 stress test。若日后要覆盖双臂故障，切到 harness 支持的 RoboTwin，别在 LIBERO 上硬凑。

## 6. Baseline：规则式 Runtime Monitor（做成强 baseline，不是稻草人）

规则监控器是 A1 类故障的正解，也是 A2/B 类的强对照。必须认真实现，但只要求它在可形式化验证的 A1 类上接近满分；不能通过为每一种已知 A2 注入写专用规则，把测试集答案泄漏进 baseline。

- **Schema / Protocol：** shape、dtype、NaN/Inf、range、timestamp、camera ID。
- **Kinematic safety：** 关节范围、末端工作空间、单步位移/旋转上限。
- **Temporal anomaly：** velocity / acceleration / jerk / oscillation / action variance。
- **Command-effect consistency：** 命令持续变化但状态长期无响应，或反馈方向与命令明显相反。

**诚实定位：** 在 A1 类故障上，好规则又快又准；A2 类需要验证，不能预设谁胜；学习模型的核心目标是在 B 类自然失败上提供规则难以获得的预警能力。

## 7. 核心算法：两个预测问题，两个结论

给定最近 h 步的视觉、状态、动作历史，可以共享特征抽取器，但必须区分两个预测目标：

```
(r_t^unsafe, r_t^outcome) = f_φ(o_{t−h:t}, a_{t−h:t}, m_{t−h:t}) ∈ [0,1]^2
```

### 7.1 Outcome Prediction（辅助诊断任务）

`r_t^outcome` 预测 episode 最终失败概率。它可以用于失败案例排序、数据筛选和分析“模型何时开始显露失败倾向”，但不能仅凭该分数宣称检测到了 imminent failure。必须按 episode 进度分桶报告性能，并加入只看 task ID / 初始帧的 difficulty baseline；如果完整历史模型没有明显超过该 baseline，说明它主要在判断任务难度。

### 7.2 Impending Failure Detection（runtime 核心任务）

`r_t^unsafe` 预测未来 K 步内是否触发有明确时刻的 self-collision、joint/workspace violation 或明显 impact。只有该输出用于报告 event recall、false alarms per episode/time、lead time，并在 MVP 中驱动 safe stop。

**模型阶梯：**

- **Baseline A：** MLP，仅 state + action history。
- **Baseline B：** Temporal MLP / 1D temporal encoder。
- **主模型：** Transformer encoder，输入 state/action token + 冻结视觉特征 + timing metadata。

视觉用冻结 encoder，避免单 A100 上把项目做成视觉预训练。

**关键消融：** state only / +action / +image；rule vs learned；初始帧 difficulty baseline vs 完整历史；MLP vs Transformer；window h 与 horizon K 的影响。

## 8. 评测协议（决定结果可信度，必须先定）

不做这个切分，"learned 比 rule 好"就是过拟合自证。

- **Held-out 故障类型：** 仅作为 A2 探索性实验；不能与 B 类主结果合并，也不作为核心成功标准。
- **核心切分：** 在同一组 2–3 个任务内，按完整 episode、seed 与 initial state 分 train/val/test。所有滑动窗口必须跟随所属 episode，杜绝相邻窗口跨集合泄漏。
- **Held-out task / object：** 仅作 exploratory evaluation。任务数扩充前，不在标题、摘要、简历或核心结论中声称 task-level generalization。
- **分 regime 报告：** A1、A2 与 B 类分别出指标，绝不合并成一个 F1。
- **阈值选择：** 所有报警阈值仅在 validation set 上确定，test set 只评一次。
- **lead time 范围：** 仅对具有明确时刻的 unsafe event 上报。
- **任务分开：** Outcome Prediction 与 Impending Failure Detection 使用独立结果表；outcome AUROC 不能作为 runtime detector 的成绩。

## 9. 风险感知闭环控制

Impending Failure Detection 不只报警，还参与执行决策。Outcome Prediction 在 MVP 中只用于诊断与离线分析，避免因“任务看起来很难”而直接终止一个本来可能成功的 episode。

### 9.1 Safe Stop（仅 Safety / Impending 主线）

```
r_t = r_t^unsafe
if r_t <  τ :  execute
if r_t ≥  τ :  safe_stop
```

**Safe stop 不是通用的全零动作。** Delta 控制下零动作可能表示保持，但 absolute joint 控制下全零可能让机械臂猛然回零。MVP 在仿真中采用“终止 episode”或由环境明确支持的 hold-current-state；未来真机部署必须调用机器人安全控制器或发送当前关节保持目标。

**Recovery（选做）** 必须是真正的恢复行为：换 prompt、触发脚本化撤退或切到 fallback 策略。把当前 chunk 丢掉后用同一策略、同一 observation 重新 query，通常仍会得到相似坏动作，不算 replan。

### 9.2 Adaptive Action Chunking（stretch goal，非必做）

想法：风险低时保持长 chunk 省调用，风险高时缩短闭环。

```
K_t = g(r_t)   low → 长 chunk ; high → 短 chunk / stop
```

**风险提示：** π0 / ACT 系是按固定 chunk 长度训练的，依赖 action chunking / temporal ensembling 的 open-loop 先验；随意砍到每步重查会打破训练时的动作分布，可能更差更抖，且 7B 模型每步重查的延迟未必扛得住。因此这条只作为选做验证，不写进核心结论。

## 10. 实验与指标

| 指标 | 定义 / 含义 | 为什么重要 |
| --- | --- | --- |
| Success Rate | 任务成功率 | 安全模块不能过度破坏原有能力 |
| Collision / Unsafe-event Rate | 碰撞或 unsafe-event 比例 | 直接衡量风险降低 |
| Precision / Recall / F1 | 异常检测质量 | 防止只靠激进 stop 刷"安全" |
| Outcome AUPRC / AUROC | episode 最终结果预测，按执行进度分桶 | 衡量 outcome prediction，不代表 impending detection |
| Outcome Decision Utility | validation 选定阈值下的 saved steps/time、false terminations、sacrificed successes | 衡量离线提前终止是否可能具有实际收益；不等同于 safe-stop 验证 |
| Impending AUPRC / Event Recall | 未来 K 步事件检测 | 衡量 runtime detection |
| Brier Score / ECE | 所选主线输出的风险概率校准 | 决定风险分数是否可信 |
| Risk–Coverage | 在不同干预覆盖率下的剩余风险 | 衡量安全与可用性的权衡 |
| False Stop Rate | 正常 episode 被提前终止比例 | 衡量误报成本 |
| Sacrificed Successes | 原本可成功但被 stop 的 episode 数 | 防止用终止一切换取低风险 |
| Early-Warning Lead Time | unsafe event 之前提前多少步预警；仅用于 impending | 核心 runtime 能力 |
| Action Smoothness | 动作变化 / 加速度 / jerk | 描述异常控制行为 |
| Queries per Episode | 每 episode 的 VLA 调用次数 | 衡量 adaptive chunking 开销 |
| Inference / Control Latency | 端到端时延 | 衡量闭环效率 |

### 冻结主结果：Outcome Prediction

下表是本次 release 的唯一主结果表。所有数字来自同一组 10 个 held-out initial-state test episodes（7 failure / 3 success）；checkpoint、模型选择和 split 均在查看 test 前冻结。Initial-proprio 是 episode 初始状态难度对照，因此各 checkpoint 使用同一预测。数值格式为 AUPRC / AUROC / Brier / ECE；Brier 与 ECE 越低越好。

| Method | Step 0 | Step 40 | Step 80 | Step 120 |
| --- | --- | --- | --- | --- |
| Initial-proprio difficulty | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 |
| State/action temporal MLP | 0.609 / 0.286 / 0.246 / 0.036 | 0.856 / 0.571 / 0.230 / 0.167 | 0.909 / 0.762 / 0.229 / 0.252 | **0.982 / 0.952 / 0.096 / 0.095** |
| Frozen dual-camera vision | 0.652 / 0.190 / 0.563 / 0.636 | 0.856 / 0.571 / 0.377 / 0.399 | **0.982 / 0.952 / 0.100 / 0.096** | **0.982 / 0.952 / 0.100 / 0.101** |

配对 episode bootstrap 支持 vision-step80 相对 initial-proprio 的排序改善：AUPRC +0.183（95% CI [0.031, 0.450]），AUROC +0.500（[0.125, 0.857]）。Temporal-step120 相对 initial-proprio 的对应差值为 +0.183（[0.033, 0.450]）和 +0.500（[0.143, 0.860]）。这证明冻结 test 上存在 outcome/progress signal；它不证明 impending failure detection。Test 只有 10 episodes，所有精度结论均受小样本限制。

**Supporting utility result：** validation 在零 sacrificed-success 约束下选择阈值 `0.9998072982`。固定策略在 test 检出 4/7 failures、牺牲 0/3 successes，共节省 756 recorded steps（按 `70.042 ms/step` 估算 52.95 秒）；3 条在 step 80、1 条在 step 120 首次触发。该结果是 offline counterfactual efficiency bound，不是闭环 safe stop 或安全改进证明。

**Supporting camera ablation：** step 80 的 main / wrist / dual AUPRC 分别为 0.844 / 1.000 / 0.982，AUROC 为 0.667 / 1.000 / 0.952。Wrist 与 dual 相对 main 的 Brier/ECE 配对区间支持改善；wrist 与 dual 之间的所有关键区间触及或跨过 0，不能声称 wrist 显著更好，也不据此在 test 上更换正式 dual-camera pipeline。

若选择 Safety 线，safe-stop 干预作为一张 supporting table，另报 Vanilla／Rule Stop／Learned Stop 的 success、unsafe rate、prevented events、false stops 与 sacrificed successes。A1/A2 stress test 同样只作为 supporting table。Fixed short chunk 与 adaptive chunk 仅在 MVP release 后实验。

## 11. MVP：最小可交付

- **环境：** LIBERO。
- **策略：** 先跑通 SmolVLA；π0.5 只在 MVP 完成后加入。
- **任务：** 当前 release 冻结 `libero_spatial` task 4；task-level generalization 留待后续多任务数据。
- **第一里程碑：** 确定性复现至少一个成功和一个失败 episode，并保存同步 RGB、proprioception、action、timing、标签与 MP4。
- **故障：** 当前完成 artifact/schema 验证与 simulator diagnostics；A1 主动注入、A2 stress test 和 command-effect consistency 明确记录为未测试，不进入本次主表。
- **检测：** 强规则 baseline + state/action temporal MLP + frozen-vision predictor；Transformer 延后。
- **控制：** Outcome 分数只做离线诊断；未实施 safe stop。Recovery / adaptive chunk 延后。
- **评测：** 核心只做 held-out episode / seed / initial state；held-out task 仅作 exploratory。Outcome 与 Impending 分表，后者才报告 unsafe-event lead time。
- **展示：** 结构化日志、唯一 Outcome 主表、test risk 曲线和 outcome-risk 叠加视频。

### 11.1 Day 0–3：Plumbing 与成本验收

Logger 必须从第一条正式 rollout 起就是执行路径的一部分，禁止先裸跑一周、之后再补日志。第 3 天前必须满足：

- 稳定获得至少一个成功和一个失败 episode；
- 每步 RGB、proprioception、VLA action/chunk、实际执行 action、timestamp、inference/control latency、event label 使用同一 step ID 对齐；
- episode metadata 包含 task、seed、initial state、checkpoint revision、resolved config 与终止原因；
- MP4 能与结构化日志按 step 对齐，抽查首帧、首次动作和终止帧一致；
- 中断或异常退出时使用原子写入/完成标记，半截 episode 不混入训练集。

第 3 天前还必须在计划使用的同一 GPU、分辨率、录像设置与模型配置下，连续测至少 20 个 episode，并记录：

| 成本项 | 必须实测的值 |
| --- | --- |
| Rollout throughput | episodes/hour、environment steps/second |
| Model latency | inference p50 / p95 / p99 |
| Recording overhead | 开/关录像的吞吐差与编码 CPU 占用 |
| Storage | GB/100 episodes；RGB、结构化日志、缓存特征分别统计 |
| Projected dataset cost | 达到 pilot 与 train/val/test 目标所需 GPU-hours、墙钟时间和费用 |

如果 20-episode benchmark 推算“仅采集目标数据就超过 1 周或超过既定算力预算”，第 3 天立即缩任务、分辨率、录像策略或模型；不能带着未经验证的成本假设进入第 2 周。

#### Day 0–3 实际验收（2026-09-11）

**结论：GO。** SmolVLA + LIBERO 闭环、同步 sidecar logger、双相机视频和原子 finalize 已跑通；`libero_spatial` task 4 的 intended-config benchmark 共 20 episodes，20/20 artifact 验证通过，得到 8 success / 12 failure。

| 指标 | 实测结果 |
| --- | ---: |
| Environment steps | 4,435 |
| Throughput | 246.37 episodes/hour |
| Query latency p50 / p95 / p99 | 344.59 / 363.18 / 400.22 ms |
| Control latency p50 / p95 / p99 | 38.94 / 46.49 / 50.33 ms |
| 录像总采集开销（按 step 归一化） | +13.8% |
| MP4 / structured / total，投影每 100 episodes | 56.49 / 207.26 / 263.75 MiB |
| 100 episodes 成本投影 | 约 0.41 GPU-hour / ¥0.84（按 ¥2.08/hour） |

Day 0–3 的 20-episode benchmark 中，仅 1/12 个失败 episode 触发可靠 self-collision 或 joint violation（8.33%）。该结果随后由正式 Week-1 50-episode cohort 取代作为分流依据；正式统计为 `1/32 = 3.125% < 20%`。因此按 §4.1 的预注册规则，四周主线冻结为 **Outcome Prediction**；停止核心 unsafe lead-time / safe-stop 主张。Workspace/impact 尚未插桩，不能记作已验证的零事件。

相对初始范围的剩余缺口是：当前只冻结一个主任务，workspace/impact 尚未形成正式协议。这些不推翻 Day 0–3 的 GO。完整数字、兼容性说明与后续动作见 `PROGRESS.md`。

#### Week 1 实际收口（2026-09-11）

**结论：Week 1 完成，Outcome 主线和数据协议冻结。** 正式 cohort 为 50 episodes（18 success / 32 failure）；按 `initial_state_id` 分组冻结为 train/validation/test = 30/10/10，之后才生成 steps 0/40/80/120 的因果窗口。Predictor 输入仅含部署可得 proprio、历史 action 与 timing；privileged labels 被输入边界拒绝。

| 项目 | 结果 |
| --- | --- |
| Initial-proprio difficulty baseline（test） | AUPRC 0.799；AUROC 0.452；Brier 0.238；ECE 0.173，未显示可靠泛化 |
| State/action temporal MLP（test step 80） | AUPRC 0.909；AUROC 0.762；相对 difficulty 的配对 CI 跨 0 |
| State/action temporal MLP（test step 120） | AUPRC 0.982；AUROC 0.952 |
| Step 120 vs initial-proprio | AUPRC 差值 +0.183，95% CI [0.033, 0.450]；AUROC +0.500，[0.143, 0.860] |
| 校准结论 | step 120 Brier/ECE 点估计改善，但配对 CI 均跨 0，不能声称显著改善 |

State/action temporal MLP 在冻结 test 上呈现**较晚出现的 outcome 排序信号**，但截至 step 80 尚无可靠的相对 difficulty 优势；不能将其描述为提前故障预警、unsafe-event 检测或 safe-stop 依据。Test 只有 10 episodes，结论必须保留小样本限制。随后按预注册计划完成 frozen-image-feature baseline；不根据 test 结果修改 split、checkpoint steps 或超参数。

#### Week 2 learned-signal gate（2026-09-11）

**判定：GO，但撤回“视觉更早发现独立失效前兆”的解释。** 冻结双相机 ResNet-50 特征的 vision predictor 在 test step 80 达到 AUPRC 0.982、AUROC 0.952，与 temporal MLP 在 step 120 的排序指标相同。这些测量结果仍成立；进一步审计改变的是对它们所代表信号的判断。

| 证据 | 对结论的约束 |
| --- | --- |
| Vision-step80 的 Brier 0.100、ECE 0.096；相对 initial-proprio difficulty baseline，AUPRC +0.183（95% CI [0.031, 0.450]）、AUROC +0.500（[0.125, 0.857]） | Outcome Prediction 的 learned-signal gate 通过；step 40 尚无可靠优势。 |
| 与 step-80 temporal MLP 直接比较，AUPRC/AUROC 配对区间包含 0；Brier/ECE 配对区间支持 vision 的概率误差与校准改善 | 不能据此声称视觉在排序意义上显著更早识别失败。 |
| RGB/error 审计：risk 主要随目标接近和成功构型进度变化；唯一 false negative 在 step 80 看起来仍顺利推进，之后接近完成却未满足成功条件，持续微调直到 step 280 超时 | 该案例与成功轨迹的可靠分叉约在 step 120–160，不能把 step-80 分数解释成独立的早期失效前兆。 |
| Leave-one-episode-out（LOEO）与逐 episode 预测审计：vision-step80、temporal-step120 错排同一 failure/success pair，LOEO 排名结果相同；概率相关为 Pearson `r = 0.9998`、Spearman `ρ = 0.9515` | 两条 pipeline 很可能读取同一种执行进度信号，不能当作相互独立的证据链；LOEO 本身也不足以单独证明信号机制。 |

因此保留的结论是：**vision 在 step 80 能读到与最终 outcome 相关的 execution progress**。撤回原“视觉把独立的 failure-discriminative signal 从 step 120 提前到 step 80”的结论；不将 outcome 分数称为 impending warning，也不据此触发 safe stop。该解释只基于 task 4 的 10 个 held-out initial-state test episodes，且失败轨迹均跑满 280 步，仍需审慎；按已通过的 gate 继续做主线收口，不升级 Transformer。

### 11.2 完整版路线（MVP 后保留）

MVP 完成后按证据逐层扩展，而不是同时开工：

1. **模型扩展：** 加入 π0.5，与 SmolVLA 比较风险模式与监控器迁移能力。
2. **故障扩展：** 从 2 类 A2 扩到 action permutation、unit、control semantics、camera mapping、chunk、latency、stale observation 全套 stress test。
3. **算法扩展：** 从 temporal MLP / frozen-vision predictor 扩到多模态 Transformer，并完成 state-only、+action、+image、window h、horizon K 等消融。
4. **控制扩展：** 在 safe stop 之外实现脚本化撤退或 fallback policy，再评估 fixed short chunk 与 adaptive chunking。
5. **平台扩展：** 切到 RoboTwin 验证双臂左右映射、异步动作和协同失败；不要在单臂 LIBERO 上伪造双臂结论。
6. **系统扩展：** 保留独立 model server / benchmark client，在可控条件下注入真实网络延迟、抖动和丢包。

## 12. 可行性、风险与防止“做到一半停摆”

### 12.1 可行性判断

- **工程闭环完成概率：高。** 仿真、公开 checkpoint、日志、故障注入、规则 baseline 和视频都能独立交付。
- **学习模型得到有效信号的概率：中等。** 第 1 周必须先判断数据究竟支持 Impending 还是 Outcome，不能默认两个任务都有信号。
- **学习模型显著超过规则的概率：中等。** 这是待验证研究假设，不作为项目是否成功的唯一标准。
- **adaptive chunk / recovery 产生正收益的概率：不确定。** 因此保持为 stretch goal。

项目成功分为两层：

1. **工程成功：** 建成可复现的闭环评测、故障注入和运行时诊断系统。
2. **研究成功：** 数据选定的唯一主线在 held-out episode/seed/initial state 上打败对应强 baseline：要么是经过校准的 Outcome Prediction，要么是对明确 unsafe event 有实用 lead time 的 Impending Detection。四周内不要求两者同时成立。

即使第二层结论为阴性，第一层仍是可以独立展示和投递的完整项目；阴性结果必须被诚实分析，而不是通过合并 regime 或挑选 seed 掩盖。

### 12.2 风险登记表

| 风险 | 早期征兆 | 缓解措施 | 降级交付 |
| --- | --- | --- | --- |
| 仿真与模型依赖冲突 | 官方 smoke test 无法稳定复现 | 模型服务与 benchmark 用 Docker / 进程隔离；固定版本与镜像 digest | 先交付独立 LIBERO recorder 与规则 monitor |
| Plumbing 吞掉第一周 | 第 3 天仍无法同步 action/timing/video | Logger 从第一条 rollout built-in；先通过 20-episode 验收再开发算法 | 暂停四周算法目标，单独完成可观测性框架 |
| 自然失败不触发 unsafe event | `p_event_given_failure` 低或 event-positive episode 不足 | 第 1 周完成 pilot 并执行 Safety/Outcome 分流 | 放弃 lead-time/safe-stop 主张，切 Outcome 主线 |
| 自然失败过少 | 任务成功率接近 100% | 换较弱 checkpoint、困难任务或合法初始化分布 | 转为 A1/A2 故障诊断项目，不伪造 B 类数据 |
| 自然成功过少 | 任务成功率接近 0% | 换任务/checkpoint，先确认官方基线 | 只做失败分类与回放分析，不训练二分类器 |
| unsafe 标签不可靠 | impact 阈值对轻微接触或数值噪声过敏 | MVP 只保留 self-collision、joint/workspace violation 与明显 impact，并做阈值敏感性分析 | 暂时移除 impact，只做前两类明确事件 |
| Outcome 被误当作 runtime detection | 初始帧/task ID baseline 已接近完整历史模型 | 两个任务独立评测；加入 difficulty baseline；只让 impending risk 驱动 stop | 将 Outcome 降为离线诊断结果 |
| predictor 读取特权信息 | 移除 object pose 后性能骤降 | 数据 schema 强制区分 deploy inputs 与 label-only fields，并写测试 | 删除污染实验，重训部署可用输入版本 |
| 滑动窗口数据泄漏 | validation 指标异常接近满分 | 先按 episode/task/init state 切分，再生成窗口 | 重新切分并废弃原结果 |
| rollout 成本过高 | 20 episodes 已耗时或占盘过大 | 先做成本测量；压缩视频、缓存视觉特征、减少无效帧 | 缩至 2 个任务与一个模型 |
| stop 让 unsafe 指标虚假变好 | unsafe 降低但成功率归零 | 同时报 false stop、sacrificed successes、intervention success | 只报告检测，不宣称改善控制 |
| learned 不超过强 baseline | 第 2 周 episode-bootstrap 结果不支持正提升 | Frozen vision 与 state/action baseline 在第 2 周内并行完成，不弱化 baseline | 执行硬 pivot，以“规则适用边界与负结果”收尾 |

### 12.3 阶段闸门（Go / Pivot / Stop）

| 时间点 | Go 条件 | 未满足时的动作 |
| --- | --- | --- |
| 第 2–3 天 | 同步 logger 从第一条 rollout 工作；稳定获得一个成功 + 一个失败；完成 20-episode 成本基准 | 回退官方原生配置；若第 3 天仍不成立，停止四周算法计划，先只做 plumbing |
| 第 1 周末 | Pilot ≥50 episodes、自然失败 ≥20；算出 `p_event_given_failure` 与数据成本；书面选择 Safety 或 Outcome 唯一主线 | 更换任务/checkpoint；仍无合适分布则 pivot 到 A1/A2 诊断框架 |
| 第 2 周末（硬闸门） | 混合成功率任务 + 冻结的无泄漏 split + 至少一个 learned baseline 在 validation 上打败对应强 baseline | 立即转保底框架/负结果项目，不再用后两周追“learned 一定赢” |
| 第 3 周末 | 唯一主线的 held-out 评测、校准和消融完成；Safety 线额外完成 stop 代价 | 删除 supporting experiments，保住主表与 demo |
| 第 4 周 | 一张诚实主结果表、一个风险叠加视频、一个可复现 release | 不扩矩阵；集中修复展示与复现 |

**“打败强 baseline”的预注册口径：**

- Safety 线：在固定 false-alarm budget 下提高 event recall，或在相同 recall 下提供更长 lead time；与 tuned rule monitor 比较。
- Outcome 线：在固定 episode-progress 切片上提高 AUPRC / calibration；与 task ID + initial-frame difficulty baseline 以及 state/action temporal baseline 比较。
- 所有差值按 episode bootstrap；若主要指标差值的 95% CI 不支持正提升，不写“超过 baseline”。

**视觉 fallback 现在就定死：** Frozen image feature 不是第 3 周临时补救，而是第 2 周 baseline 阶段与 state/action temporal MLP 并行准备。第 2 周末如果两者都没有信号，直接执行 pivot；不再升级 Transformer，也不再追加两周调参。

### 12.4 保底项目形态

如果 B 类 failure predictor 最终没有可靠泛化，项目不废弃，而是收敛为：

> **VLA-SafeBench: A Reproducible Fault-Injection, Observability and Runtime-Diagnostics Framework for Closed-Loop VLA Evaluation**

保底成果包括统一 rollout schema、同步日志与视频、A1/A2 故障注入、强规则监控、command-effect consistency、风险/动作曲线、失败分类与完整复现实验。该版本不声称学习式监控优于规则，但仍完整展示 VLA serving、仿真闭环、控制接口、可观测性和实验设计能力。

## 13. 停手标准

> **本次 release 停手点：** SmolVLA + LIBERO task 4 闭环、同步日志/视频、50-episode 冻结 cohort、held-out initial-state Outcome Prediction、预注册 baseline、校准与 episode bootstrap、唯一主结果表、最小相机消融、offline efficiency 分析和 outcome-risk 叠加视频。RQ1、Impending、held-out task、多模型、双臂、recovery 与 adaptive chunking 均留作明确的后续工作，不阻塞当前单任务 MVP 发布。

## 14. 简历 / README 表述

**项目名：** VLA-SafeBench: Learning-Based Failure Prediction and Risk-Aware Closed-Loop Control for VLA Policies

**一句话：** Built a reproducible SmolVLA–LIBERO evaluation pipeline and trained episode-outcome predictors on deployment-available observation–action histories, separating outcome-associated execution progress from imminent safety-event detection.

**Bullets：**

- Built a SmolVLA + LIBERO closed-loop pipeline with synchronized dual-camera video, proprioception, predicted/executed actions, timing, provenance, and label-only simulator diagnostics from the first rollout step.
- Froze a 50-episode task-4 cohort and group-disjoint 30/10/10 split by initial state; compared initial-difficulty, temporal, and frozen-vision outcome predictors without privileged simulator inputs.
- Found frozen dual-camera vision reached test AUPRC 0.982 / AUROC 0.952 at step 80, then used RGB/error and LOEO audits to limit the claim to outcome-associated execution progress rather than imminent failure prediction.
- Evaluated a validation-thresholded offline policy that identified 4/7 held-out failures with 0/3 observed sacrificed successes and a 756-step counterfactual saving; did not present it as a safety intervention.

## 15. 实现与仓库结构

### 15.1 当前目录

```text
vla-safe-bench/
├── src/vlasafe/
│   ├── rollout/             # schema、recorder、validator、simulator diagnostics
│   ├── outcome_dataset.py
│   ├── outcome_metrics.py
│   └── predictor_inputs.py  # deployment-input allowlist
├── scripts/                 # collect、train、evaluate、render 等入口
├── tests/                   # recorder、validator、dataset、metrics、leakage tests
├── docs/manifests/          # 冻结 episode split
├── docs/release-artifacts.md
├── PROGRESS.md              # 阶段决策、结果与限制
└── README.md
```

大规模 rollout、模型权重、缓存特征和完整视频不提交 Git；使用 manifest 记录 checkpoint revision、镜像 digest、任务、seed、配置哈希与产物位置。

### 15.2 技术栈

- **Simulation / benchmark：** LIBERO（第一阶段）；双臂故障如需覆盖切 RoboTwin。
- **Policy：** π0.5 / SmolVLA 等公开 VLA。
- **Serving：** 独立 model server + benchmark client（保留延迟/网络故障注入能力）——可直接用 `allenai/vla-evaluation-harness`：WebSocket + msgpack、Docker 隔离、内置录像与 per-step sqlite 日志、官方含 π0/OpenVLA/GR00T server、支持 LIBERO/LIBERO-Pro/RoboTwin/RoboCasa。
- **Learning：** PyTorch；MLP / temporal encoder / Transformer；冻结视觉 encoder。
- **Logging / Eval：** 结构化 episode 日志、视频、风险曲线、failure/unsafe timestamp；success、unsafe-rate、F1、false-stop、lead time、latency、queries/episode。

### 15.3 可复现性最低要求

- 每次 rollout 保存完整 resolved config、Git commit、checkpoint revision、task、initial state 与随机 seed。
- 同一配置至少能够重放 observation/action 日志，并确定性复现指标计算；物理仿真不保证逐浮点位一致时，允许在容差内复现结果。
- 数据切分 manifest 在生成训练窗口前冻结，并加入测试确保同一 episode 不跨 split。
- Predictor 的输入 schema 使用 allowlist；label-only privileged fields 在 dataloader 边界被显式拒绝。
- README 只展示由固定 release 配置生成的数字，失败实验和已知限制一并保留。

### 15.4 AutoDL 上的 GitHub 网络恢复

AutoDL 实例重启后若 `git pull` 卡住、`curl https://github.com` 超时或出现 `GnuTLS recv error (-110)`，先在当前 shell 启用平台代理并验证远端，再执行同步：

```bash
source /etc/network_turbo

GIT_TERMINAL_PROMPT=0 \
git -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1000 \
    -c http.lowSpeedTime=30 \
    ls-remote origin HEAD

git -c http.version=HTTP/1.1 pull --ff-only
```

`source` 只影响当前 shell，新开终端或实例重启后需要重新执行。`http.postBuffer` 主要影响大体积 HTTP push，不是连接超时或 pull 失败的必要修复。若曾通过 VS Code 手动覆盖仓库文件，先用 `git status --short` 检查并 `git stash push --include-untracked` 保存，再 pull；远端已包含相同修改时不要盲目 `stash pop`。

## 16. 已知风险与诚实边界

- LIBERO 是单臂，双臂故障不适用；需要就切 RoboTwin。
- 若 B 类自然失败样本太少，学习模型可能只和强规则打平——缓解：多任务多 seed、优先选不稳任务收集自然失败。
- 若自然成功或失败比例过于极端，先更换任务或 checkpoint，不通过重复窗口伪造样本量。
- 仿真 privileged state 只允许生成标签；一旦进入 predictor 输入，所有泛化结果作废。
- MVP 不定义任务相关的“非预期接触”，避免陷入 contact whitelist 规则工程；只使用可自动标注的明确事件。
- adaptive chunking 的效率-安全权衡未必成立，仅作选做。
- safe stop 必须服从控制空间语义；全零动作不能默认视为安全。
- recovery 需真实 fallback，否则退化为 stop。

## 17. 项目价值

- 比"fine-tune 一个 VLA"更有辨识度：明确问题定义、故障体系、自建数据、泛化协议。
- 比纯 safety checker 更贴算法岗：真正训练 failure-detection 模型并做时序/多模态消融与泛化评测。
- 比只做模型训练更贴机器人：指标来自闭环 rollout，而非离线 loss。
- 直击 VLA 部署真实痛点：schema、控制语义、chunk、延迟、observation mapping，以及最难的自然涌现失败。
