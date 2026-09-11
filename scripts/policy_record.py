"""Run one recorded SmolVLA episode in LIBERO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from vlasafe.rollout.sim_diagnostics import SimDiagnostics


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
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("docs/provenance.json"),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable frame copies and MP4 encoding for recording-overhead benchmarks.",
    )
    return parser.parse_args()


def _record_episode(
    *,
    args: argparse.Namespace,
    episode_index: int,
    checkpoint: Path,
    env: Any,
    policy: SmolVLAPolicy,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_description: list[str],
    provenance: dict[str, Any],
    vlasafe_revision: str,
) -> dict[str, Any]:
    episode_started = time.perf_counter()
    seed = args.seed + episode_index
    episode_id = datetime.now(timezone.utc).strftime("smolvla-%Y%m%dT%H%M%S%fZ")
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        task="libero_spatial",
        task_id=args.task_id,
        seed=seed,
        initial_state_id=episode_index,
        policy_id=str(checkpoint),
        policy_revision=provenance["policy_revision"],
        lerobot_revision=provenance["lerobot_revision"],
        resolved_config={
            "fps": 20,
            "max_steps": args.max_steps,
            "observation_height": 360,
            "observation_width": 360,
            "control_mode": "relative",
            "benchmark_episode_index": episode_index,
            "task_description": task_description[0],
            "video_enabled": not args.no_video,
            "vlasafe_revision": vlasafe_revision,
            "provenance": provenance,
        },
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    recorder = SidecarRecorder(args.output_root, metadata)
    query_latencies_ms: list[float] = []
    control_latencies_ms: list[float] = []
    clipped_steps = 0
    try:
        policy.reset()
        raw_obs, _ = env.reset(seed=seed)
        diagnostics = SimDiagnostics.from_vector_env(env)
        main_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        action_queue: deque[np.ndarray] = deque()
        active_chunk: np.ndarray | None = None
        chunk_id = -1
        chunk_offset = 0
        success = False
        num_steps = 0

        rollout_started = time.perf_counter()
        for step_id in range(args.max_steps):
            observation_timestamp_ns = time.monotonic_ns()
            if not args.no_video:
                main_frames.append(np.asarray(raw_obs["pixels"]["image"][0]).copy())
                wrist_frames.append(np.asarray(raw_obs["pixels"]["image2"][0]).copy())
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
            inference_latency_ms = (inference_finished_ns - inference_started_ns) / 1e6
            if queried_policy:
                query_latencies_ms.append(inference_latency_ms)
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
            control_latency_ms = (step_finished_ns - action_executed_ns) / 1e6
            control_latencies_ms.append(control_latency_ms)
            clipped_steps += int(clip_delta > 0.0)
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
                    inference_latency_ms=inference_latency_ms,
                    control_latency_ms=control_latency_ms,
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
                    label_only=diagnostics.sample(next_obs),
                )
            )
            chunk_offset += 1
            num_steps += 1
            raw_obs = next_obs
            if _first_bool(terminated) or _first_bool(truncated):
                break

        rollout_finished = time.perf_counter()
        encoding_started = time.perf_counter()
        encoding_cpu_started = time.process_time()
        if not args.no_video:
            imageio.mimsave(
                recorder.partial_dir / "main_camera.mp4",
                main_frames,
                fps=20,
                codec="libx264",
                macro_block_size=8,
            )
            imageio.mimsave(
                recorder.partial_dir / "wrist_camera.mp4",
                wrist_frames,
                fps=20,
                codec="libx264",
                macro_block_size=8,
            )
        encoding_cpu_finished = time.process_time()
        encoding_finished = time.perf_counter()
        reason = "success" if success else "max_steps"
        rollout_s = rollout_finished - rollout_started
        encoding_s = encoding_finished - encoding_started
        encoding_cpu_s = encoding_cpu_finished - encoding_cpu_started
        metrics = {
            "rollout_seconds": rollout_s,
            "video_encoding_seconds": encoding_s,
            "video_encoding_cpu_seconds": encoding_cpu_s,
            "video_enabled": not args.no_video,
            "steps_per_second": num_steps / rollout_s if rollout_s else None,
            "episode_seconds_before_finalize": encoding_finished - episode_started,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        final_dir = recorder.finalize(
            termination_reason=reason,
            success=success,
            metrics=metrics,
        )
        print(f"RECORDED {num_steps} SmolVLA steps")
        print(f"FINALIZED {final_dir}")
        print(f"ROLLOUT seconds {rollout_s:.3f}")
        print(f"VIDEO ENCODING seconds {encoding_s:.3f}")
        print(f"VIDEO ENCODING CPU seconds {encoding_cpu_s:.3f}")
        print(f"STEPS/s {metrics['steps_per_second']:.3f}")
        print(f"PEAK VRAM MiB {metrics['peak_vram_mib']:.1f}")
        return {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "initial_state_id": episode_index,
            "seed": seed,
            "success": success,
            "num_steps": num_steps,
            "clipped_steps": clipped_steps,
            "query_latencies_ms": query_latencies_ms,
            "control_latencies_ms": control_latencies_ms,
            "rollout_seconds": rollout_s,
            "video_encoding_seconds": encoding_s,
            "video_encoding_cpu_seconds": encoding_cpu_s,
            "artifact_bytes": sum(path.stat().st_size for path in final_dir.rglob("*") if path.is_file()),
        }
    except BaseException:
        recorder.abort()
        raise


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    result = np.percentile(np.asarray(values), [50, 95, 99])
    return {"p50": float(result[0]), "p95": float(result[1]), "p99": float(result[2])}


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    checkpoint = args.checkpoint.resolve()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    required_provenance = {"lerobot_revision", "policy_revision"}
    missing_provenance = required_provenance - provenance.keys()
    if missing_provenance:
        raise ValueError(f"missing provenance keys: {sorted(missing_provenance)}")
    vlasafe_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_paths = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty_paths:
        raise RuntimeError(
            "tracked files are dirty; commit or stash them before formal collection"
        )
    benchmark_started = time.perf_counter()
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
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg, policy_cfg=policy.config
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        task_description = list(env.call("task_description"))
        setup_finished = time.perf_counter()
        episodes = []
        for episode_index in range(args.num_episodes):
            print(f"\n=== EPISODE {episode_index + 1}/{args.num_episodes} ===")
            episodes.append(
                _record_episode(
                    args=args,
                    episode_index=episode_index,
                    checkpoint=checkpoint,
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_description=task_description,
                    provenance=provenance,
                    vlasafe_revision=vlasafe_revision,
                )
            )
        benchmark_finished = time.perf_counter()
    finally:
        if env is not None:
            env.close()

    query_latencies = [x for ep in episodes for x in ep["query_latencies_ms"]]
    control_latencies = [x for ep in episodes for x in ep["control_latencies_ms"]]
    collection_seconds = benchmark_finished - setup_finished
    summary = {
        "schema_version": "0.1.0",
        "task": "libero_spatial",
        "task_id": args.task_id,
        "seed_start": args.seed,
        "vlasafe_revision": vlasafe_revision,
        "provenance": provenance,
        "video_enabled": not args.no_video,
        "num_episodes": len(episodes),
        "successes": sum(ep["success"] for ep in episodes),
        "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
        "total_steps": sum(ep["num_steps"] for ep in episodes),
        "total_clipped_steps": sum(ep["clipped_steps"] for ep in episodes),
        "setup_seconds": setup_finished - benchmark_started,
        "collection_seconds": collection_seconds,
        "episodes_per_hour": len(episodes) * 3600 / collection_seconds,
        "query_latency_ms": _percentiles(query_latencies),
        "control_latency_ms": _percentiles(control_latencies),
        "total_artifact_bytes": sum(ep["artifact_bytes"] for ep in episodes),
        "episodes": episodes,
    }
    benchmark_root = Path("artifacts/benchmarks")
    benchmark_root.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_root / datetime.now(timezone.utc).strftime(
        "benchmark-%Y%m%dT%H%M%S%fZ.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBENCHMARK SUMMARY {summary_path}")
    print(f"SUCCESS {summary['successes']}/{summary['num_episodes']} ({summary['success_rate']:.1%})")
    print(f"EPISODES/hour {summary['episodes_per_hour']:.2f}")
    print(f"QUERY latency ms {summary['query_latency_ms']}")
    print(f"CONTROL latency ms {summary['control_latency_ms']}")


if __name__ == "__main__":
    main()
