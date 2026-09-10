# VLA-SafeBench Progress

> 最后更新：2026-09-10。这里只保留影响当前决策、阶段闸门和复现的事实；完整研究边界见 `README.md`。

## 1. 当前状态

| 项目 | 当前结论 |
| --- | --- |
| 阶段 | Day 0–3：Plumbing 与成本验收 |
| Policy / Simulator | SmolVLA + LIBERO，闭环 rollout 已跑通 |
| 当前主任务候选 | `libero_spatial` task 4 |
| 数据特征 | 同一 task 内已有混合 outcome：独立 pilot 约 2 成功 / 4 失败 |
| 主线倾向 | **Outcome Prediction**；尚未正式冻结 |
| 关键原因 | 修正事件语义后，task 4 自然失败暂未观察到可靠 self-collision / joint violation |
| 当前阻塞 | 尚未完成 task 4 的 20-episode benchmark；impact/workspace 协议尚未冻结 |

## 2. 阶段闸门

| Day 0–3 验收项 | 状态 | 证据 / 缺口 |
| --- | :---: | --- |
| CUDA、仿真、checkpoint 可运行 | ✅ | RTX 4090 D；SmolVLA + LIBERO 闭环运行稳定 |
| 从第 0 步同步记录 | ✅ | RGB、proprio、raw chunk、executed action、timing、label-only diagnostics |
| 双相机 MP4 与 JSONL 对齐 | ✅ | validator 检查所有 `*_camera.mp4` 与 step 数 |
| 至少一条成功和一条失败 | ✅ | task 4 同任务内已取得成功与失败 |
| 20-episode intended-config benchmark | ⏳ | 下一项；计划 task 4、seed 700–719、init state 0–19 |
| episodes/hour、延迟、录像与存储成本 | 🟡 | 已有 pilot 估计；等待 20 条正式汇总 |
| 可用于分流的 unsafe-event 标签 | 🟡 | self/joint 可用；workspace/impact 仍为 `null` |

在 20-episode benchmark 完成前，不进入 predictor 训练。

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
| **task 4** | **独立 pilot 约 2 成功 / 4 失败** | **选为 20-episode benchmark 主任务** |
| task 7 | 6/6 成功 | 过易，排除主任务 |

禁止把 task 0 的全成功与 task 5 的全失败直接混合训练，否则 predictor 可通过 task identity 作弊。

## 5. Pilot 性能

| Pilot | Success | Episodes/hour | Query latency p50/p95/p99 | Control latency p50/p95/p99 |
| --- | ---: | ---: | --- | --- |
| task 0，5 episodes | 5/5 | 456.53 | 342.01 / 731.88 / 980.28 ms | 35.73 / 47.75 / 53.66 ms |
| task 4，5 episodes | 1/5 | 216.05 | 330.37 / 345.79 / 834.10 ms | 39.11 / 45.74 / 51.87 ms |
| task 5，5 episodes | 0/5 | 220.40 | 326.11 / 338.10 / 799.48 ms | 33.01 / 45.42 / 54.94 ms |

这些是 pilot 数字，不作为最终 benchmark 结果。

## 6. 事件协议状态

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

当前 task 4 的已知失败中，可靠 self/joint event 初步覆盖接近 0，因此 Outcome 主线比 Impending Safety 主线更可能成立。最终分流仍以 Week-1 正式 pilot 为准。

## 7. 下一步

1. 运行 task 4 的 20-episode benchmark：seed 700–719、initial state 0–19。
2. 自动验证全部 episode，汇总成功/失败、吞吐、延迟、裁剪率、存储和 self/joint event 覆盖。
3. 用 pilot 原始 EEF/contact-force 分布形成并版本化 workspace/impact 协议；不得用 outcome 标签反向挑阈值。
4. 固化依赖版本、LeRobot revision 与运行配置。
5. 达到 Week-1 最低样本要求后，计算正式 `p_event_given_failure`，书面选择唯一主线。

## 8. 复现环境

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
