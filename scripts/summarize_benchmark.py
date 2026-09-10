"""Validate and summarize a policy_record benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlasafe.rollout.validator import validate_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--episodes-root", type=Path, default=Path("artifacts/episodes"))
    args = parser.parse_args()

    summary: dict[str, Any] = json.loads(args.summary.read_text(encoding="utf-8"))
    episode_rows = []
    all_steps: list[dict[str, Any]] = []
    for expected_index, item in enumerate(summary["episodes"]):
        episode_dir = args.episodes_root / item["episode_id"]
        validation = validate_episode(episode_dir)
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (episode_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected_seed = summary["seed_start"] + expected_index
        if metadata["seed"] != expected_seed:
            raise ValueError(f"{item['episode_id']}: seed {metadata['seed']} != {expected_seed}")
        if metadata["initial_state_id"] != expected_index:
            raise ValueError(
                f"{item['episode_id']}: initial_state_id {metadata['initial_state_id']} != {expected_index}"
            )
        labels = [row["label_only"] for row in rows]
        clear_event = any(
            label["self_collision"] or label["joint_violation"] for label in labels
        )
        episode_rows.append(
            {
                "episode_id": item["episode_id"],
                "seed": metadata["seed"],
                "initial_state_id": metadata["initial_state_id"],
                "success": validation.success,
                "num_steps": validation.num_steps,
                "clipped_steps": validation.clipped_steps,
                "clear_self_or_joint_event": clear_event,
                "max_robot_contact_force": max(
                    label["max_robot_contact_force"] for label in labels
                ),
            }
        )
        all_steps.extend(rows)

    failures = [row for row in episode_rows if not row["success"]]
    failures_with_event = [row for row in failures if row["clear_self_or_joint_event"]]
    total_steps = len(all_steps)
    total_bytes = int(summary["total_artifact_bytes"])
    analysis = {
        "benchmark_summary": str(args.summary),
        "validated_episodes": len(episode_rows),
        "successes": sum(row["success"] for row in episode_rows),
        "failures": len(failures),
        "success_rate": sum(row["success"] for row in episode_rows) / len(episode_rows),
        "total_steps": total_steps,
        "episodes_per_hour": summary["episodes_per_hour"],
        "query_latency_ms": summary["query_latency_ms"],
        "control_latency_ms": summary["control_latency_ms"],
        "total_artifact_mib": total_bytes / 1024**2,
        "mean_artifact_mib_per_episode": total_bytes / len(episode_rows) / 1024**2,
        "projected_mib_per_100_episodes": total_bytes / len(episode_rows) * 100 / 1024**2,
        "clipped_steps": sum(row["clipped_steps"] for row in episode_rows),
        "clipped_step_rate": sum(row["clipped_steps"] for row in episode_rows) / total_steps,
        "failures_with_self_or_joint_event": len(failures_with_event),
        "p_self_or_joint_event_given_failure": (
            len(failures_with_event) / len(failures) if failures else None
        ),
        "rollout_seconds": sum(item["rollout_seconds"] for item in summary["episodes"]),
        "video_encoding_seconds": sum(
            item["video_encoding_seconds"] for item in summary["episodes"]
        ),
        "episode_rows": episode_rows,
    }
    output = args.summary.with_name(f"{args.summary.stem}-analysis.json")
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("idx seed init success steps clipped clear_event max_force")
    for index, row in enumerate(episode_rows):
        print(
            index,
            row["seed"],
            row["initial_state_id"],
            row["success"],
            row["num_steps"],
            row["clipped_steps"],
            row["clear_self_or_joint_event"],
            f"{row['max_robot_contact_force']:.3f}",
        )
    print(f"\nANALYSIS {output}")
    for key, value in analysis.items():
        if key != "episode_rows":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
