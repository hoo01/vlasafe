"""Record a short zero-action LIBERO episode with a synchronized sidecar."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
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
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/episodes"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    episode_id = datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%SZ")
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        task="libero_spatial",
        task_id=0,
        seed=args.seed,
        initial_state_id=0,
        policy_id="smoke/zero-action",
        policy_revision="none",
        lerobot_revision="codeload-main-unpinned",
        resolved_config={
            "fps": 20,
            "steps": args.steps,
            "observation_height": 480,
            "observation_width": 640,
            "control_mode": "relative",
        },
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    recorder = SidecarRecorder(args.output_root, metadata)
    env = None

    try:
        cfg = LiberoEnvConfig(
            task="libero_spatial",
            task_ids=[0],
            observation_height=480,
            observation_width=640,
            episode_length=args.steps,
        )
        env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][0]
        obs, _ = env.reset(seed=args.seed)
        frames: list[np.ndarray] = []
        success = False

        for step_id in range(args.steps):
            observation_timestamp_ns = time.monotonic_ns()
            frames.append(np.asarray(obs["pixels"]["image"][0]).copy())
            proprio = _proprioception(obs["robot_state"])

            inference_started_ns = time.monotonic_ns()
            # Interface smoke policy only. This is not a safety intervention.
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            predicted_chunk = action.tolist()
            inference_finished_ns = time.monotonic_ns()

            action_executed_ns = time.monotonic_ns()
            next_obs, reward, terminated, truncated, info = env.step(action)
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
                    predicted_action_chunk=predicted_chunk,
                    executed_action=action[0].astype(float).tolist(),
                    proprioception=proprio,
                    reward=float(np.asarray(reward).reshape(-1)[0]),
                    terminated=_first_bool(terminated),
                    truncated=_first_bool(truncated),
                    success=success,
                    deploy_metadata={"policy_kind": "zero_action_smoke"},
                    label_only={
                        "events_instrumented": False,
                        "self_collision": None,
                        "joint_violation": None,
                        "workspace_violation": None,
                        "impact": None,
                    },
                )
            )
            obs = next_obs
            if _first_bool(terminated) or _first_bool(truncated):
                break

        imageio.mimsave(recorder.partial_dir / "main_camera.mp4", frames, fps=20, codec="libx264")
        final_dir = recorder.finalize(termination_reason="smoke_horizon", success=success)
        print(f"RECORDED {len(frames)} synchronized steps")
        print(f"FINALIZED {final_dir}")
    except BaseException:
        recorder.abort()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

