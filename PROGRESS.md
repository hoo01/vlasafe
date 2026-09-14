# VLA-SafeBench Progress

> 最后更新：2026-09-11。这里只保留影响当前决策、阶段闸门和复现的事实；完整研究边界见 `README.md`。

## 1. 当前状态

| 项目 | 当前结论 |
| --- | --- |
| 阶段 | **Week 2 learned-signal gate 已通过**；进入 decision utility 与最小消融 |
| Policy / Simulator | SmolVLA + LIBERO，闭环 rollout 已跑通 |
| 当前主任务 | `libero_spatial` task 4 |
| 正式 benchmark | **20/20 验证通过；8 成功 / 12 失败（40% success）** |
| 主线决策 | **Outcome Prediction** |
| 关键原因 | 正式 50-episode cohort 中仅 1/32 自然失败触发可用 unsafe event（3.125%） |
| 正式 Week-1 cohort | **50 episodes；18 success / 32 failure；全部固定 provenance** |
| 冻结 split | **30 train / 10 validation / 10 test；initial state 分组不相交** |
| Week-1 结论 | temporal MLP 在 step 120 显著优于 initial-proprio difficulty baseline；step 0–80 尚无可靠优势 |
| Week-2 结论 | learned-signal gate 通过；vision 与 temporal 主要读取 execution-progress 信号，不支持独立的早期失败前兆主张 |
| 当前剩余 | 冻结主结果表、README 最终叙事与可复现 release |

## 2. 阶段闸门

| Day 0–3 验收项 | 状态 | 证据 / 缺口 |
| --- | :---: | --- |
| CUDA、仿真、checkpoint 可运行 | ✅ | RTX 4090 D；SmolVLA + LIBERO 闭环运行稳定 |
| 从第 0 步同步记录 | ✅ | RGB、proprio、raw chunk、executed action、timing、label-only diagnostics |
| 双相机 MP4 与 JSONL 对齐 | ✅ | validator 检查所有 `*_camera.mp4` 与 step 数 |
| 至少一条成功和一条失败 | ✅ | task 4 同任务内已取得成功与失败 |
| 20-episode intended-config benchmark | ✅ | task 4；seed 700–719；init state 0–19；20/20 artifact 验证通过 |
| 吞吐、延迟、总存储 | ✅ | 246.37 episodes/hour；延迟与 52.75 MiB 总 artifact 已实测 |
| Recording overhead 与存储拆分 | ✅ | 同 seed 5+5 对照；按 step 归一化总开销 13.8%；已完成分项统计 |
| 可用于分流的 unsafe-event 标签 | ✅ | self/joint 可用；失败覆盖 1/12，已据此选择 Outcome 主线 |

**Day 0–3 Gate：GO。** Impact/workspace 不阻塞 Outcome 主线。

### 与初始 README 的偏差

| 初始设定 | 当前状态 | 处理 |
| --- | --- | --- |
| 用 Week-1 数据在 Outcome / Impending 中二选一 | self/joint event 仅覆盖 1/12 自然失败 | 按预注册阈值冻结 **Outcome** 主线；不是路线偏离 |
| Impending risk 可驱动 safe stop | 当前数据不支持可靠 impending 标签 | safe stop 与 lead time 从核心交付移除，仅保留 exploratory |
| MVP 最终覆盖 2–3 个任务 | 当前只冻结 task 4 作为首个 regime | Day 0–3 合理；第二任务后续补，held-out task 仍不作核心主张 |
| self/joint/workspace/impact 标签 | self/joint 已可靠；workspace/impact 未冻结 | 后两者降为 supporting work，不阻塞 Outcome |
| 精确可复现 revision / environment | 主要依赖已记录 | LeRobot commit、checkpoint revision 与环境 lock 尚需固化 |

总体属于数据驱动的范围收缩：工程路径不重做，核心叙事从“近期 unsafe-event 预警”收敛为“基于部署可用历史的 episode outcome 预测”。不得将 Outcome 指标表述为 runtime safety detection。

## 3. 已实现系统

| 模块 | 已验证能力 |
| --- | --- |
| Rollout runner | 模型只加载一次；多 episode 连续 reset；seed 与 initial-state ID 递增 |
| Sidecar recorder | `.incomplete` → 原子 finalize；稳定 episode/step/frame ID |
| Action logging | 保留 raw predicted chunk；执行前按 `[-1, 1]` 裁剪；记录 executed action 与裁剪幅度 |
| Video | 360×360、20 FPS、主相机 + 腕部相机，无 368×368 隐式 resize |
| Artifact validator | COMPLETE、schema、必填字段、有限数值、连续 ID、时间戳、result/JSONL/video 对齐 |
| Label-only diagnostics | joint limits/margins、EEF xyz、机器人 contact force/pair、self-collision pair |
| 信息边界 | privileged diagnostics 只进入 `label_only`，不得进入 predictor |
| 自动化测试 | 25/25 通过 |

### Checkpoint 兼容性

官方 `lerobot/smolvla_libero` checkpoint 的 JSON 将 `observation.state` 写成 6 维，但 dataset metadata、normalizer tensor 和当前 LIBERO processor 均为 8 维。运行时使用 `smolvla_libero_compat`，只修正 state metadata；模型权重哈希与原 checkpoint 相同。

### Action clipping

一次 81-step 成功轨迹中，裁剪主要来自夹爪维（72 步，最大 0.03778）；机械臂仅 Z 维轻微越界（7 步，最大 0.02121）。当前判定为反归一化后的边界 overshoot；保留裁剪和 A1 range-monitor 日志。

## 4. 任务筛选结果

| Task / 数据 | Outcome | 决策 |
| --- | --- | --- |
| task 0，已知完整轨迹 | 9/9 成功 | 过易，排除主任务 |
| task 1–3、6、8–9 初筛 | 各 1/1 成功 | 暂不扩测 |
| task 5 | 0/6 成功 | 过难；仅保留失败回放 / supporting analysis |
| **task 4** | **正式 benchmark：8/20 成功** | **冻结为 Outcome Prediction 首个 regime** |
| task 7 | 6/6 成功 | 过易，排除主任务 |

禁止把 task 0 的全成功与 task 5 的全失败直接混合训练，否则 predictor 可通过 task identity 作弊。

## 5. Pilot 性能

| Pilot | Success | Episodes/hour | Query latency p50/p95/p99 | Control latency p50/p95/p99 |
| --- | ---: | ---: | --- | --- |
| task 0，5 episodes | 5/5 | 456.53 | 342.01 / 731.88 / 980.28 ms | 35.73 / 47.75 / 53.66 ms |
| task 4，5 episodes | 1/5 | 216.05 | 330.37 / 345.79 / 834.10 ms | 39.11 / 45.74 / 51.87 ms |
| task 5，5 episodes | 0/5 | 220.40 | 326.11 / 338.10 / 799.48 ms | 33.01 / 45.42 / 54.94 ms |

这些是 pilot 数字，不作为最终 benchmark 结果。

## 6. 20-episode 正式验收

| 指标 | 结果 |
| --- | ---: |
| Artifact validation | **20/20 通过** |
| Outcome | **8 success / 12 failure** |
| Success rate | **40.0%** |
| 总 environment steps | 4,435 |
| Throughput | 246.37 episodes/hour |
| Query latency p50 / p95 / p99 | 344.59 / 363.18 / 400.22 ms |
| Control latency p50 / p95 / p99 | 38.94 / 46.49 / 50.33 ms |
| Artifact 总量 / 每 episode | 52.75 / 2.64 MiB |
| 100 episodes 存储投影 | 263.75 MiB |
| Action-clipped steps | 2,805 / 4,435（63.25%） |
| Rollout / video encoding 总时长 | 215.51 / 27.22 s |
| 失败且有明确 self/joint event | **1/12（8.33%）** |

结论：同一 task、同一 checkpoint、不同 seed/init state 下同时具备足量成功与失败，适合作为 Outcome Prediction 的首个 regime。明确 self/joint event 对自然失败的覆盖过低，不能把这些失败统一叙述为 impending unsafe event。

### Recording overhead 与成本投影

录像开/关各跑 5 个匹配 seed；两组 outcome 均为 2/5 success。成功 episode 的终止步数有轻微差异（1,104 vs 1,110 steps），因此采用按 step 归一化结果，不直接把原始 episodes/hour 差值当作录像开销。

| 指标 | 开录像 | 关录像 | 开销 |
| --- | ---: | ---: | ---: |
| Rollout ms/step | 49.973 | 48.281 | +3.5% |
| 总采集 ms/step（含编码） | 70.042 | 61.539 | **+13.8%** |
| Episodes/hour（仅供参考） | 232.78 | 263.51 | -11.7% |
| Encoding wall / CPU | 6.474 s / 1.481 s（5 episodes） | 0 / 0 | 1.295 / 0.296 s per episode |

| 存储项 | 20 episodes | 投影 / 100 episodes |
| --- | ---: | ---: |
| 双相机 MP4 | 11.30 MiB | 56.49 MiB |
| JSONL + metadata | 41.45 MiB | 207.26 MiB |
| 合计 | 52.75 MiB | **263.75 MiB** |

按实测 246.37 episodes/hour 和当前租价 ¥2.08/hour 粗略投影：100 episodes 约 0.41 GPU-hour、¥0.84、0.258 GiB；200 episodes 约 0.81 GPU-hour、¥1.69、0.515 GiB。未计模型首次下载和人工操作间隔。采集远低于一周与算力预算阈值，无需降低分辨率或取消录像。

## 7. 事件协议状态

| 事件 / 诊断量 | 状态 | 当前语义 |
| --- | :---: | --- |
| Joint violation | ✅ | MuJoCo robot joint range；记录逐关节 margin 与 violation index |
| Self-collision | ✅ | arm–arm / arm–gripper；排除正常 gripper–gripper 闭合接触 |
| Robot contact force | ✅ 原始量 | 只统计机器人参与的 contact；记录最大力与 geom pair |
| Impact | ⏳ | 阈值未冻结，布尔值保持 `null` |
| Workspace violation | ⏳ | task-independent bounds 未冻结，布尔值保持 `null` |

### 已废弃结论

event schema 0.1.0 曾把左右 gripper pad 闭合接触误标为 self-collision，从而得到错误的 `p_event_given_failure = 1.0`。该结论永久废弃，不得进入报告。

event schema 0.2.0 已结构性排除 gripper–gripper 接触。相同 task 4 失败条件重跑后：18 步内部夹爪接触、0 步 self-collision、0 步 joint violation。

正式 20-episode benchmark 中，可靠 self/joint event 只覆盖 1/12 自然失败。因此主线正式冻结为 Outcome Prediction；Impending Safety 仅保留为 event-positive 子集上的 exploratory analysis。

### Week-1 正式事件覆盖

| 项目 | 结果 |
| --- | ---: |
| 自然 episodes / failures | 50 / 32 |
| Failure + 任一可用事件 | **1 / 32** |
| `p_event_given_failure` | **3.125%** |
| Self-collision | 0 episodes / 0 steps |
| Joint violation | 1 episode / 14 steps |
| Workspace / impact | 未插桩，不参与当前分流统计 |
| Event-positive split 分布 | train 1；validation 0；test 0 |

这低于预注册的 20% Safety 门槛，且 validation/test 没有 event-positive episode，无法可信评测 impending recall、lead time 或校准。**Outcome 是唯一主线；停止 unsafe lead-time 和 safe-stop 核心主张。** 当前 `p_event_given_failure` 严格指已插桩且可用的 self-collision / joint-violation 事件，不能把未插桩的 workspace/impact 当作已验证的 0。

## 8. Week 1 收口

### Week-1 正式 cohort 与冻结切分

| Split | Episodes | Success | Failure | Success rate | Initial-state groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 30 | 11 | 19 | 36.7% | 17 |
| Validation | 10 | 4 | 6 | 40.0% | 7 |
| Test | 10 | 3 | 7 | 30.0% | 6 |
| **Total** | **50** | **18** | **32** | **36.0%** | **30 unique** |

冻结 manifest：`docs/manifests/week1_task4_split.json`（commit `6b3f445`）。切分 seed 为 `20260911`；按 `initial_state_id` 分组、按 outcome 分层。任何时间窗口必须继承 episode split，禁止重新随机切窗口或使用 test set 选择阈值。

正式 cohort 使用 seed 1000–1049，VLA-SafeBench revision `d16985f`，LeRobot tree SHA-256 `a48da863...f39a`，policy weights SHA-256 `9a9f6413...fca8`。旧 Day 0–3 benchmark 保留作成本与系统验收，不并入 predictor 训练集。

### Outcome 数据入口与首个 difficulty baseline

| 项目 | 状态 / 结果 |
| --- | --- |
| Deploy-input allowlist | ✅ 只允许 proprio、action/chunk、timing 与 frame index；拒绝 label/outcome 字段 |
| Leakage tests | ✅ 总测试 25/25 通过 |
| Outcome dataset | ✅ 200 samples；steps 0/40/80/120；window 16；0 skipped |
| Prevalence baseline（test） | AUPRC 0.700；AUROC 0.500；Brier 0.214；ECE 0.067 |
| Initial-proprio logistic（validation） | AUPRC/AUROC 1.000；Brier 0.206；ECE 0.109 |
| Initial-proprio logistic（test） | AUPRC 0.799；AUROC 0.452；Brier 0.238；ECE 0.173 |

Initial-proprio logistic 的 validation 满分未迁移到 held-out initial-state test；test AUROC 低于随机，Brier/ECE 也差于 prevalence。仅 AUPRC 较高不足以支持泛化结论，且 test 只有 10 episodes。当前把它记录为**初步负结果/难度对照**，不据此调 split 或反复选择超参数。

### State/action temporal MLP

模型只使用部署可得的 16-step proprio/action/timing 历史；训练归一化仅来自 train，五个固定 seed 的 checkpoint 只按 validation BCE 选择。Test 仍为冻结的 10 个 held-out initial-state episode。

| Step | Test temporal AUPRC / AUROC | Test Brier / ECE | 相对 initial-proprio 的配对 bootstrap 结论 |
| ---: | ---: | ---: | --- |
| 0 | 0.609 / 0.286 | 0.246 / 0.036 | 无优势；AUPRC、AUROC 差值均为负且 CI 跨 0 |
| 40 | 0.856 / 0.571 | 0.230 / 0.167 | 正向点估计很小，全部 CI 跨 0 |
| 80 | 0.909 / 0.762 | 0.229 / 0.252 | 有排序趋势，但相对 difficulty 的 AUPRC/AUROC CI 仍跨 0 |
| 120 | **0.982 / 0.952** | **0.096 / 0.095** | **AUPRC +0.183，95% CI [0.033, 0.450]；AUROC +0.500，[0.143, 0.860]** |

step 120 的 Brier 差值为 -0.143、ECE 差值为 -0.078，但置信区间均跨 0，不能声称校准显著改善。可靠排序优势出现较晚；这是 **Outcome Prediction** 的运行历史信号，不是 impending failure detection，也不构成 safe-stop 依据。Test 仅 10 episodes，所有区间仍较宽。

### Week-1 完成判定

| 交付项 | 状态 |
| --- | :---: |
| ≥50 natural episodes 且 ≥20 failures | ✅ 50 episodes / 32 failures |
| 正式事件覆盖与主线分流 | ✅ 1/32 = 3.125%；冻结 Outcome 主线 |
| Provenance 固化 | ✅ project revision、LeRobot tree hash、policy weights hash 已写入 rollout metadata |
| Group-disjoint split 先于窗口生成 | ✅ 30/10/10，按 initial state 分组 |
| Predictor 输入边界与 leakage tests | ✅ |
| Difficulty baseline | ✅ 初始 proprio 泛化不足 |
| State/action temporal baseline | ✅ step 120 有显著排序优势，早期优势不足 |

**Week 1：完成。** 不再修改 cohort、split、checkpoint steps 或 test episode 来改善结果。

### Episode 时长审计（Week 2 前置）

冻结 manifest 已足以复现该统计；`remaining steps` 定义为观察完 checkpoint step 后仍未执行的已记录步骤。

| Outcome | Episodes | 终止步数 min / median / max | 平均终止步数 |
| --- | ---: | ---: | ---: |
| Success | 18 | 122 / 133 / 160 | 133.28 |
| Failure | 32 | 280 / 280 / 280 | 280.00 |
| All | 50 | 122 / 280 / 280 | 227.18 |

| Checkpoint | 成功轨迹剩余步数（median / mean） | 失败轨迹剩余步数（median / mean） |
| ---: | ---: | ---: |
| 40 | 92 / 92.28 | 239 / 239.00 |
| 80 | 52 / 52.28 | 199 / 199.00 |
| 120 | **12 / 12.28** | **159 / 159.00** |

所有 50 条轨迹在 step 120 仍可观测，但成功轨迹此时通常已接近结束，而失败轨迹全部持续到 280-step horizon。因而 step-120 高指标可能同时反映真实运行历史信号和“是否已呈现成功收尾”的进展/停滞信号。它有潜在计算节省价值，但现阶段只能称为 late-stage outcome/progress diagnosis，不能称为 early warning。脚本：`scripts/audit_episode_timing.py`。

### Frozen-vision baseline 与 Week-2 信号闸门

冻结 `torchvision/resnet50 IMAGENET1K_V2`，从主相机与腕部相机的固定 checkpoint 帧提取 2×2048 维特征；encoder 不训练，每个 checkpoint 的线性 probe 仅用 validation 选择 L2。缓存为 200 records、shape `(200, 2, 2048)`、2.2 MiB，全部有限。

| Step | Checkpoint vision AUPRC / AUROC | Brier / ECE | 相对强基线的配对结论 |
| ---: | ---: | ---: | --- |
| 0 | 0.652 / 0.190 | 0.563 / 0.636 | 无信号，且校准明显差 |
| 40 | 0.856 / 0.571 | 0.377 / 0.399 | 所有关键 CI 跨 0 |
| 80 | **0.982 / 0.952** | **0.100 / 0.096** | vs initial-proprio：AUPRC +0.183，[0.031, 0.450]；AUROC +0.500，[0.125, 0.857] |
| 120 | 0.982 / 0.952 | 0.100 / 0.101 | 与 temporal MLP 排序相同，无额外提升 |

在 step 80，checkpoint vision 相对 initial-frame vision 的 AUPRC、AUROC、Brier、ECE 四项 CI 均支持改善；相对 temporal MLP 的 AUPRC/AUROC 配对区间包含 0，不能用“vision 显著超过 difficulty、temporal 未显著超过 difficulty”间接推出 vision 显著超过 temporal。Brier 与 ECE 的直接配对区间支持改善，但这是概率误差与校准优势，不证明排序意义上的更早失败识别。

冻结审计后定档为 **B：vision 主要读取任务执行进度 / 是否接近成功构型，而不是独立的早期 failure precursor。** Step-80 RGB 中，高风险失败尚未形成成功轨迹常见的目标接近/操作构型，低风险成功的腕部视野已被目标占据；唯一 false negative 在 step 80 视觉上同样呈现顺利推进，却最终失败，直接限制了“提前预警”的解释。Vision-step80 与 temporal-step120 概率不相同，但错排的是同一 failure/success pair，因而在 10 个 test episode 上得到完全相同的 AUPRC/AUROC 与 LOEO 排名结果。

LOEO 的 vision 与 temporal 结果均为：AUPRC 0.976–1.000，AUROC 0.929–1.000；删除错排 pair 任一端后两项均变为 1.0。结果没有因删除普通单个 episode 而崩溃，但 test 只有 7×3=21 个正负配对，一个 pair 就决定 AUROC 的一个 0.0476 台阶，不足以支撑精确到 checkpoint 的“信号提前”主张。

逐 episode 概率审计得到 Pearson `r = 0.9998`、Spearman `ρ = 0.9515`。Vision-step80 与 temporal-step120 的概率值并不相同，但共同错排 failure `smolvla-20260911T013117405943Z` 与 success `smolvla-20260911T014100736108Z`；两条 pipeline 不应作为相互独立的证据链，更可能在测量同一种 execution-progress 潜在信号。

上述 false negative 未触发可用的 self-collision 或 joint-violation event，最终以 `max_steps` 结束。它在 step 80 与成功轨迹视觉上高度相似；约在 step 120–160 进入近完成构型后仍未满足成功条件，随后持续小幅调整，最低的 16-step 状态运动窗口集中于 step 268–277，最终在 280-step horizon 超时。因此首次可靠分叉约在 step 120–160，而非 step 80；现有证据不支持把 step-80 score 解释为独立失败前兆。

可保留的结论是：vision 在 step 80 已能可靠读取与最终 outcome 相关的 execution progress，并显著超过初始难度基线。它仍只适用于 task 4 的 held-out initial states；test 仅 10 episodes，且失败全部跑满 280 步。它不是 impending unsafe-event detection，也不授权 safe stop。

**Week-2 learned-signal gate：GO。** 不执行 negative-result pivot，也不升级 Transformer。审计显示两类模型很可能读取同一种 execution-progress 信号；增加参数更可能重拟合该信号，而不是发现新的早期失败前兆。

### 下一步

1. 保持当前方法与 0/40/80/120 checkpoint 冻结，不再根据 test 修改模型。
2. 用 validation 选择 outcome threshold，计算 test saved steps/time、false terminations 与 sacrificed successes；定位为 efficiency / faster confirmation，不表述为 safety warning。
3. 做最小必要消融：主相机 / 腕部相机 / 双相机；避免扩展模型矩阵。
4. 生成按 checkpoint 展示 risk 与最终 outcome 的可视化；措辞保持 outcome/progress diagnosis。

### RQ1（A1/A2）状态与后续验证

**未测试。** 现有 artifact validator 与正常 rollout 验证不等于 A1 故障的运行时拦截实验；`1/32` 自然失败事件覆盖率也不能回答 A1/A2 配置故障检出率。当前没有 A2 配对注入、command-effect consistency 覆盖率或学习监控的 A2 故障检测结果。现有 learned predictor 预测最终 episode outcome，不能直接充当 A2 故障检测器。

优先完成 Outcome 主表、离线 decision utility、最小相机消融、可视化及可复现 release；RQ1 保持 supporting result，不阻塞当前主线。主线稳定后，先做 A1 的 shape/dtype/NaN/时间戳等注入单测，明确区分执行前规则拦截与事后 artifact 校验；再选 1–2 类当前 SmolVLA + LIBERO 接口可注入、数值合法的 A2 故障（如动作维度置换、主/腕相机映射错位或 chunk 执行长度错误）做 stress test。单位混用须先核实控制维度语义；当前接口没有独立目标坐标配置入口，不把“错误目标坐标”列为首轮案例。

每个注入案例配同 seed、同 initial state 的正常对照，预先固定注入点、前 N 步检出窗口及 validation 阈值；按故障类型分别报告检出数/总数、正常对照误报数/总数和首次报警步数。规则监控与 command-effect consistency 分列；只有建立独立的 A2 故障标签、训练/验证切分和冻结阈值后，才报告学习监控的 A2 覆盖率。A1、A2、B 不合并指标，也不对从 t=0 注入的故障报告 unsafe-event lead time。

### 最小相机消融（2026-09-14）

使用同一 frozen ResNet-50 feature cache、冻结 split、checkpoint 和 validation 选 L2 协议，分别训练 main-only、wrist-only 与 dual-camera linear probe。Dual-camera 复跑与原结果一致。Step 80 的 test 点估计如下（10 episodes）：

| 相机输入 | AUPRC | AUROC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Main only | 0.844 | 0.667 | 0.306 | 0.330 |
| Wrist only | 1.000 | 1.000 | 0.001 | 0.017 |
| Dual camera | 0.982 | 0.952 | 0.100 | 0.096 |

配对 episode bootstrap 显示，wrist 与 dual 相对 main 的 Brier/ECE 差值 95% CI 均严格低于 0，支持二者在该 test 上具有更低概率误差和校准误差；AUPRC/AUROC 差值区间下界为 0，不能称为严格的排序提升。Wrist 相对 dual 的 AUPRC +0.018（CI [0, 0.107]）、AUROC +0.048（[0, 0.250]），Brier -0.099（[-0.296, 0.000]）、ECE -0.079（[-0.263, 0.008]），所有区间均触及或跨过 0，因此没有可靠证据表明 wrist 优于 dual。

结论限定为：step-80 outcome/progress 信号主要可由腕部视角读出；双相机没有显示出相对 wrist-only 的可靠增益，main-only 明显更弱的证据主要体现在 Brier/ECE。该消融是机制诊断，不用于在 test 上重新选择主模型；正式主结果仍保留预注册的 dual-camera pipeline。结果 artifact：`artifacts/results/week1_task4_camera_ablation_bootstrap.json`。

### Offline outcome decision utility（2026-09-14）

使用冻结的 dual-camera predictor 与 0/40/80/120 checkpoint。阈值只在 validation 上选择：要求零 sacrificed validation successes，在此约束下最大化失败轨迹节省步数；test 不参与选择。选定阈值为 `0.9998072982`。

| Split | 检出的失败 | Sacrificed successes | 节省步数 | 估算 wall time |
| --- | ---: | ---: | ---: | ---: |
| Validation | 5/6 | 0/4 | 1,115 | 78.10 s |
| Test | 4/7 | 0/3 | 756 | 52.95 s |

Test 中 3 条失败在 step 80 首次超过阈值，各节省 199 步；1 条在 step 120 首次超过阈值，节省 159 步；另外 3 条失败始终未达到阈值。按全部 7 条 test failure 计，平均每条节省 108 步。已审计的近完成后超时 false negative `smolvla-20260911T013117405943Z` 未被该策略提前终止，与此前的机制分析一致。

该结果定位为 **offline faster confirmation / efficiency**：在已记录轨迹不受干预的反事实假设下，保守阈值可提前确认一部分最终失败并减少后续 rollout。`52.95 s` 使用实测 `70.042 ms/step` 线性换算，只是 wall-time 估计。阈值非常接近 1，反映概率饱和且不应视为可迁移的通用阈值；test 仅 7 failures / 3 successes，`0/3` sacrificed successes 不足以证明真实误停率接近零。该分析不证明闭环干预效果，不称为 safe stop、impending warning 或安全改进。结果 artifact：`artifacts/results/week1_task4_outcome_utility.json`。

### Outcome-risk 可视化（2026-09-14）

已生成 10 条 held-out test episode 在 step 0/40/80/120 的 frozen dual-camera outcome-risk 曲线，以及失败 episode `smolvla-20260911T014308102939Z` 的风险叠加视频。该 episode 在离线策略中于 step 80 首次超过 validation 阈值；视频仍完整播放原始 280-step rollout，并明确标注风险语义为最终 episode failure probability、触发点为 offline counterfactual，不表现为真实 safe-stop 干预。

产物为 `artifacts/visualizations/week1_task4/test_outcome_risk_curves.png`、`artifacts/visualizations/week1_task4/smolvla-20260911T014308102939Z_outcome_risk_overlay.mp4` 与 `render_metadata.json`。FFprobe 验证输出视频为 360×360、20 FPS、280 帧；渲染器已禁止 ImageIO 将画面隐式缩放到 368×368。

## 9. 复现环境

| 组件 | 版本 / 路径 |
| --- | --- |
| GPU | NVIDIA RTX 4090 D 24 GB |
| Python | 3.12.9；`/root/autodl-tmp/envs/vlasafe312` |
| PyTorch | 2.11.0+cu128 |
| MuJoCo / robosuite | 3.8.1 / 1.4.0 |
| FFmpeg | 7.1.1 |
| Project | `/root/autodl-tmp/vlasafe` |

```bash
source /root/autodl-tmp/envs/vlasafe312/bin/activate
export PYTHONPATH=/root/autodl-tmp/vlasafe/src
export LIBERO_CONFIG_PATH=/root/autodl-tmp/vlasafe/.libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/vlasafe/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

### 非阻塞事项

- robosuite private macro 警告不影响当前 reset/step/render。
- LIBERO assets 当前位于 `/root/.cache/libero/assets`；关机前需确认持久性或迁移到数据盘。
