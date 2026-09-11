# VLA-SafeBench Progress

> 最后更新：2026-09-11。这里只保留影响当前决策、阶段闸门和复现的事实；完整研究边界见 `README.md`。

## 1. 当前状态

| 项目 | 当前结论 |
| --- | --- |
| 阶段 | **Week-1 数据门槛与 split 已通过**；进入 Outcome baseline 数据构建 |
| Policy / Simulator | SmolVLA + LIBERO，闭环 rollout 已跑通 |
| 当前主任务 | `libero_spatial` task 4 |
| 正式 benchmark | **20/20 验证通过；8 成功 / 12 失败（40% success）** |
| 主线决策 | **Outcome Prediction** |
| 关键原因 | 正式 50-episode cohort 中仅 1/32 自然失败触发可用 unsafe event（3.125%） |
| 正式 Week-1 cohort | **50 episodes；18 success / 32 failure；全部固定 provenance** |
| 冻结 split | **30 train / 10 validation / 10 test；initial state 分组不相交** |
| 当前剩余 | 构建部署可用输入，先做 difficulty baseline，再做 state/action temporal baseline |

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
| 自动化测试 | 13/13 通过 |

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

## 8. 下一步

### Week-1 正式 cohort 与冻结切分

| Split | Episodes | Success | Failure | Success rate | Initial-state groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 30 | 11 | 19 | 36.7% | 17 |
| Validation | 10 | 4 | 6 | 40.0% | 7 |
| Test | 10 | 3 | 7 | 30.0% | 6 |
| **Total** | **50** | **18** | **32** | **36.0%** | **30 unique** |

冻结 manifest：`docs/manifests/week1_task4_split.json`（commit `6b3f445`）。切分 seed 为 `20260911`；按 `initial_state_id` 分组、按 outcome 分层。任何时间窗口必须继承 episode split，禁止重新随机切窗口或使用 test set 选择阈值。

正式 cohort 使用 seed 1000–1049，VLA-SafeBench revision `d16985f`，LeRobot tree SHA-256 `a48da863...f39a`，policy weights SHA-256 `9a9f6413...fca8`。旧 Day 0–3 benchmark 保留作成本与系统验收，不并入 predictor 训练集。

### 下一步

1. 定义 predictor deploy-input allowlist，并加入 privileged-field leakage 测试。
2. 从冻结 manifest 构建按 episode 归属的 Outcome 样本；先不生成视觉特征。
3. 实现 task/initial-frame difficulty baseline 与 episode-progress 分桶评测。
4. 训练 state/action temporal MLP，并并行准备 frozen-image features。
5. 只在 validation 上选择模型与校准方式，test 保持封存。

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
- `lerobot_revision` 与 `policy_revision` 当前仍是占位描述；正式数据扩充前必须替换为可验证 commit/hash。
