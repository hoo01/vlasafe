"""Run one recorded SmolVLA episode in LIBERO."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from vlasafe.rollout import EpisodeMetadata, SidecarRecorder, StepRecord


def _first_bool(value: Any) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def _proprioception(value: dict[str, Any], prefix: str = "") -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_proprioception(item, name))
        else:
            result[name] = np.asarray(item)[0].reshape(-1).astype(float).tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smolvla_libero_compat"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/episodes"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    checkpoint = args.checkpoint.resolve()
    episode_id = datetime.now(timezone.utc).strftime("smolvla-%Y%m%dT%H%M%SZ")
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        task="libero_spatial",
        task_id=args.task_id,
        seed=args.seed,
        initial_state_id=0,
        policy_id=str(checkpoint),
        policy_revision="local-compat-copy",
        lerobot_revision="codeload-main-unpinned",
        resolved_config={
            "fps": 20,
            "max_steps": args.max_steps,
            "observation_height": 360,
            "observation_width": 360,
            "control_mode": "relative",
        },
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    recorder = SidecarRecorder(args.output_root, metadata)
    env = None

    try:
        cfg = LiberoEnvConfig(
            task="libero_spatial",
            task_ids=[args.task_id],
            observation_height=360,
            observation_width=360,
            episode_length=args.max_steps,
        )
        env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][args.task_id]
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
        policy.reset()
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg,
            policy_cfg=policy.config,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )

        raw_obs, _ = env.reset(seed=args.seed)
        task_description = list(env.call("task_description"))
        frames: list[np.ndarray] = []
        action_queue: deque[np.ndarray] = deque()
        active_chunk: np.ndarray | None = None
        chunk_id = -1
        chunk_offset = 0
        success = False

        for step_id in range(args.max_steps):
            observation_timestamp_ns = time.monotonic_ns()
            frames.append(np.asarray(raw_obs["pixels"]["image"][0]).copy())
            proprio = _proprioception(raw_obs["robot_state"])
            inference_started_ns = time.monotonic_ns()
            queried_policy = not action_queue

            if queried_policy:
                observation = preprocess_observation(raw_obs)
                observation["task"] = task_description
                observation = env_preprocessor(observation)
                observation = preprocessor(observation)
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(observation)
                    chunk = postprocessor(chunk)
                    chunk = env_postprocessor({ACTION: chunk})[ACTION]
                active_chunk = chunk.detach().cpu().numpy()
                action_queue.extend(active_chunk[0])
                chunk_id += 1
                chunk_offset = 0

            inference_finished_ns = time.monotonic_ns()
            if active_chunk is None:
                raise RuntimeError("policy did not produce an action chunk")
            predicted_action = np.asarray(action_queue.popleft(), dtype=np.float32)[None, :]
            action_numpy = np.clip(
                predicted_action,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32, copy=False)
            clip_delta = float(np.max(np.abs(action_numpy - predicted_action)))
            action_executed_ns = time.monotonic_ns()
            next_obs, reward, terminated, truncated, info = env.step(action_numpy)
            step_finished_ns = time.monotonic_ns()
            success = _first_bool(info.get("is_success", False))

            recorder.add_step(
                StepRecord(
                    episode_id=episode_id,
                    step_id=step_id,
                    frame_id=step_id,
                    observation_timestamp_ns=observation_timestamp_ns,
                    inference_started_ns=inference_started_ns,
                    inference_finished_ns=inference_finished_ns,
                    action_executed_ns=action_executed_ns,
                    inference_latency_ms=(inference_finished_ns - inference_started_ns) / 1e6,
                    control_latency_ms=(step_finished_ns - action_executed_ns) / 1e6,
                    predicted_action_chunk=active_chunk[0].astype(float).tolist(),
                    executed_action=action_numpy[0].astype(float).tolist(),
                    proprioception=proprio,
                    reward=float(np.asarray(reward).reshape(-1)[0]),
                    terminated=_first_bool(terminated),
                    truncated=_first_bool(truncated),
                    success=success,
                    deploy_metadata={
                        "policy_queried": queried_policy,
                        "chunk_id": chunk_id,
                        "chunk_offset": chunk_offset,
                        "action_clipped": clip_delta > 0.0,
                        "action_clip_linf": clip_delta,
                    },
                    label_only={
                        "events_instrumented": False,
                        "self_collision": None,
                        "joint_violation": None,
                        "workspace_violation": None,
                        "impact": None,
                    },
                )
            )
            chunk_offset += 1
            raw_obs = next_obs
            if _first_bool(terminated) or _first_bool(truncated):
                break

        imageio.mimsave(
            recorder.partial_dir / "main_camera.mp4",
            frames,
            fps=20,
            codec="libx264",
            macro_block_size=8,
        )
        reason = "success" if success else "integration_horizon"
        final_dir = recorder.finalize(termination_reason=reason, success=success)
        print(f"RECORDED {len(frames)} SmolVLA steps")
        print(f"FINALIZED {final_dir}")
        print(f"PEAK VRAM MiB {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    except BaseException:
        recorder.abort()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
