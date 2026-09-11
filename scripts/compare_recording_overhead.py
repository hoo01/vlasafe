"""Compare matched policy_record benchmarks with video enabled and disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_matched(video: dict[str, Any], no_video: dict[str, Any]) -> None:
    keys = ("task", "task_id", "seed_start", "num_episodes")
    mismatches = [key for key in keys if video.get(key) != no_video.get(key)]
    if mismatches:
        raise ValueError(f"benchmarks are not matched on: {', '.join(mismatches)}")
    if video.get("video_enabled") is not True:
        raise ValueError("first benchmark must have video_enabled=true")
    if no_video.get("video_enabled") is not False:
        raise ValueError("second benchmark must have video_enabled=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_summary", type=Path)
    parser.add_argument("no_video_summary", type=Path)
    args = parser.parse_args()

    video = _load(args.video_summary)
    no_video = _load(args.no_video_summary)
    _check_matched(video, no_video)

    video_rate = float(video["episodes_per_hour"])
    no_video_rate = float(no_video["episodes_per_hour"])
    video_seconds = float(video["collection_seconds"])
    no_video_seconds = float(no_video["collection_seconds"])
    video_steps = int(video["total_steps"])
    no_video_steps = int(no_video["total_steps"])
    video_rollout_seconds = sum(item["rollout_seconds"] for item in video["episodes"])
    no_video_rollout_seconds = sum(
        item["rollout_seconds"] for item in no_video["episodes"]
    )
    video_rollout_seconds_per_step = video_rollout_seconds / video_steps
    no_video_rollout_seconds_per_step = no_video_rollout_seconds / no_video_steps
    video_total_seconds_per_step = video_seconds / video_steps
    no_video_total_seconds_per_step = no_video_seconds / no_video_steps
    overhead = {
        "task": video["task"],
        "task_id": video["task_id"],
        "seed_start": video["seed_start"],
        "num_episodes": video["num_episodes"],
        "video_episodes_per_hour": video_rate,
        "no_video_episodes_per_hour": no_video_rate,
        "throughput_reduction_fraction": 1.0 - video_rate / no_video_rate,
        "video_collection_seconds": video_seconds,
        "no_video_collection_seconds": no_video_seconds,
        "extra_wall_seconds_per_episode": (
            video_seconds - no_video_seconds
        ) / video["num_episodes"],
        "video_total_steps": video_steps,
        "no_video_total_steps": no_video_steps,
        "video_rollout_ms_per_step": video_rollout_seconds_per_step * 1000,
        "no_video_rollout_ms_per_step": no_video_rollout_seconds_per_step * 1000,
        "frame_copy_rollout_overhead_fraction": (
            video_rollout_seconds_per_step / no_video_rollout_seconds_per_step - 1.0
        ),
        "video_total_ms_per_step": video_total_seconds_per_step * 1000,
        "no_video_total_ms_per_step": no_video_total_seconds_per_step * 1000,
        "normalized_total_overhead_fraction": (
            video_total_seconds_per_step / no_video_total_seconds_per_step - 1.0
        ),
        "video_encoding_wall_seconds": sum(
            item["video_encoding_seconds"] for item in video["episodes"]
        ),
        "video_encoding_cpu_seconds": sum(
            item.get("video_encoding_cpu_seconds", 0.0) for item in video["episodes"]
        ),
        "video_artifact_mib": video["total_artifact_bytes"] / 1024**2,
        "no_video_artifact_mib": no_video["total_artifact_bytes"] / 1024**2,
        "matched_outcomes": [item["success"] for item in video["episodes"]]
        == [item["success"] for item in no_video["episodes"]],
        "matched_step_counts": [item["num_steps"] for item in video["episodes"]]
        == [item["num_steps"] for item in no_video["episodes"]],
    }
    output = args.video_summary.with_name(
        f"recording-overhead-{args.video_summary.stem}-{args.no_video_summary.stem}.json"
    )
    output.write_text(json.dumps(overhead, indent=2), encoding="utf-8")
    print(f"OVERHEAD {output}")
    for key, value in overhead.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
