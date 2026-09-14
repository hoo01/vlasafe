# VLA-SafeBench

VLA-SafeBench 是一个基于 SmolVLA 和 LIBERO 的闭环评测项目，用来研究机器人策略失败时，系统能够观察到什么、预测什么，以及哪些结论不能从现有数据推出。

当前核心发现是：冻结视觉特征可以较早预测 task 4 的最终成功或失败；RGB/error 和 LOEO 审计提示模型可能主要读取**任务执行进度**，但这一机制解释还没有经过进度控制实验的定量验证。风险分数目前只用于 Outcome Prediction 和离线效率分析，不能描述为即将发生危险或安全停止依据。

## 四周执行状态

| 阶段 | 状态 | 已完成 | 尚未完成 |
| --- | --- | --- | --- |
| **Day 0–3 / Week 1** | **完成** | SmolVLA + LIBERO 闭环；从首步启用同步日志和双相机视频；20-episode 性能/成本验收；50 条 task-4 自然 rollout；事件覆盖统计；冻结 30/10/10 group-disjoint split；确定 Outcome 主线 | 无 Week-1 阻塞项 |
| **Week 2** | **完成** | Initial-proprio difficulty baseline；16-step state/action temporal MLP；冻结 ResNet-50 双相机 predictor；paired episode bootstrap；输入泄漏测试；RGB/error、LOEO 和关键 false-negative 审计；形成 execution-progress 假设 | 不再增加 Transformer 或继续调参 |
| **Week 3** | **进行中** | 主/腕/双相机消融；validation-thresholded offline utility；test risk 曲线；360×360 outcome-risk 叠加视频；18-file v0.1 checksum | **进度控制实验：定量检验控制执行进度后是否仍有 outcome 信号** |
| **Week 4** | **未开始** | v0.1 已提供可复现阶段快照 | 统一完成度表述；根据进度控制结果冻结最终主表、结论、demo 和 release |

Week 1–2 回答了“系统能否运行、数据支持哪条主线、学习模型是否存在信号”。Week 3 要回答“这个信号能否在独立数据中复现，以及控制进度后是否仍成立”。Week 4 才是完整项目的最终收口。当前 v0.1 不能替代后两步。

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

当前完成的是 **RQ2a 的单任务阶段结果和离线 efficiency 分析**。RQ1、RQ2b 和真实闭环干预尚未完成。

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

当前平台为 **SmolVLA + LIBERO `libero_spatial` task 4**。Rollout 路径从第一步开始同步记录：

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

## 5. 方法

当前比较三个 Outcome predictor：

1. **Initial-proprio difficulty baseline：** 只读取初始机器人状态，检验完整模型是否只是学习初始难度。
2. **State/action temporal MLP：** 读取最近 16 步的 proprioception、动作和 timing 历史。
3. **Frozen dual-camera vision：** 使用冻结的 `torchvision/resnet50 IMAGENET1K_V2` 提取主/腕相机特征，再训练 checkpoint-specific linear probe。

模型选择和 L2 正则只使用 validation。结果按完整 episode 进行 paired bootstrap，不把同一 episode 的相关窗口当成独立样本。

## 6. 当前主结果

以下是同一组 10 条 held-out initial-state test episodes 的结果。每格依次为 **AUPRC / AUROC / Brier / ECE**，后两项越低越好。

| Method | Step 0 | Step 40 | Step 80 | Step 120 |
| --- | --- | --- | --- | --- |
| Initial-proprio difficulty | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 | 0.799 / 0.452 / 0.238 / 0.173 |
| State/action temporal MLP | 0.609 / 0.286 / 0.246 / 0.036 | 0.856 / 0.571 / 0.230 / 0.167 | 0.909 / 0.762 / 0.229 / 0.252 | **0.982 / 0.952 / 0.096 / 0.095** |
| Frozen dual-camera vision | 0.652 / 0.190 / 0.563 / 0.636 | 0.856 / 0.571 / 0.377 / 0.399 | **0.982 / 0.952 / 0.100 / 0.096** | **0.982 / 0.952 / 0.100 / 0.101** |

Paired episode bootstrap 支持 vision-step80 相对 initial-proprio 的排序改善：AUPRC `+0.183 [0.031, 0.450]`，AUROC `+0.500 [0.125, 0.857]`。Temporal-step120 的对应差值为 AUPRC `+0.183 [0.033, 0.450]`、AUROC `+0.500 [0.143, 0.860]`。

这些结果证明 task 4 的冻结 test 上存在与最终 outcome 相关的信号，但 test 只有 10 条，不能据此声称任务级泛化。

## 7. 模型实际读取了什么

最初的表面结果是：vision 在 step 80 达到了 temporal MLP 在 step 120 才达到的排序指标。RGB/error、leave-one-episode-out 和逐 episode 审计限制了这一解释：

- 成功轨迹通常在 step 122–160 结束，中位数约为 133；所有失败轨迹都运行到 step 280 超时。
- Step 80 的低风险画面通常更接近成功构型，高风险画面通常进度较慢。
- Vision-step80 与 temporal-step120 错排同一对 failure/success，概率高度相关（Pearson `0.9998`，Spearman `0.9515`）。
- 关键 false negative 在 step 80 看起来仍顺利推进，之后进入近完成构型，却持续微调至 step 280 超时；可靠分叉约在 step 120–160。

因此当前最稳妥的工作假设是：**模型主要读取 outcome-associated execution progress**。保留“预测最终结果”的测量结论，撤回“视觉更早发现独立失效前兆”的机制解释。下一步必须用进度控制实验判断：控制进度后信号消失、部分保留，还是现有样本不足以区分两种解释。

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

## 9. 当前状态与下一阶段

2026-09-14 冻结的 18-file checksum 是 **task-4 Outcome v0.1 阶段快照**，不是四周项目终点。它完成了单任务 pipeline 和主假设审计，但证据仍然偏薄。

当前只推进一项研究工作：**进度控制实验**。使用现有 50 条轨迹构造预先定义的进度代理，比较进度相近但最终 outcome 不同的 episode，并检验加入冻结 outcome risk 后是否仍改善 held-out 判别或概率误差。三种结果都如实报告：信号消失、信号部分保留，或样本不足以区分。

独立 confirmatory cohort、第二任务和 RQ1 supporting study 延期。它们分别只能增强 task-4 稳定性、增加一个独立任务 regime 或展开另一条故障诊断战线，当前都不如进度控制实验直接回答主结论。Transformer、π0.5、RoboTwin、recovery 和 adaptive chunking 同样不进入当前阶段。

## 10. 复现

完整 artifact 索引与命令见 [`docs/release-artifacts.md`](docs/release-artifacts.md)。

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

大规模 rollout、视频、模型权重和 feature cache 不提交 Git，通过 manifest、artifact index 和 SHA-256 清单追踪。

## 11. 已知限制

- 当前正式数据只覆盖一个 LIBERO task。
- Test 只有 10 条 episode，7 条失败全部是 280-step timeout。
- Outcome predictor 没有明确 failure timestamp，不能报告 unsafe lead time。
- Self-collision/joint-violation 标签覆盖不足，workspace/impact 尚未冻结。
- RQ1 的主动故障注入、command-effect consistency 和 A2 learned monitor 尚未测试。
- Offline utility 没有执行真实闭环干预。
- 当前结果不能支持 held-out task、物体或平台泛化主张。

## 12. 项目定位

这是一个关于 VLA 失败预测的**单任务方法论研究**。核心贡献是泄漏受控的闭环数据与评测流程，以及两次基于证据的结论收缩：先因 unsafe-event 覆盖不足从 Impending 转向 Outcome，再因机制审计撤回“独立失效前兆”的解释。项目不把当前结果包装成可泛化的失败检测方法。
