"""Validate an independently collected confirmation cohort without splitting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlasafe.rollout.validator import validate_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-state-start", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    episodes = []
    for item in summary["episodes"]:
        episode_dir = args.episodes_root / item["episode_id"]
        validation = validate_episode(episode_dir)
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        episodes.append(
            {
                "episode_id": item["episode_id"],
                "path": episode_dir.as_posix(),
                "seed": int(metadata["seed"]),
                "initial_state_id": int(metadata["initial_state_id"]),
                "success": validation.success,
                "num_steps": validation.num_steps,
            }
        )
    expected_ids = list(
        range(args.expected_state_start, args.expected_state_start + args.expected_episodes)
    )
    actual_ids = sorted(row["initial_state_id"] for row in episodes)
    if len(episodes) != args.expected_episodes or actual_ids != expected_ids:
        raise ValueError(
            f"confirmation cohort mismatch: episodes={len(episodes)}, states={actual_ids}"
        )
    manifest = {
        "schema_version": "0.1.0",
        "role": "independent_confirmation_only",
        "selection": "preset before outcome inspection",
        "identity": {
            "task": summary["task"],
            "task_id": summary["task_id"],
            "vlasafe_revision": summary["vlasafe_revision"],
            "provenance": summary["provenance"],
        },
        "totals": {
            "episodes": len(episodes),
            "successes": sum(row["success"] for row in episodes),
            "failures": sum(not row["success"] for row in episodes),
        },
        "splits": {"confirmation": episodes},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("CONFIRMATION MANIFEST", args.output)
    print("episodes", len(episodes), "success", manifest["totals"]["successes"], "failure", manifest["totals"]["failures"])
    print("initial_state_ids", actual_ids)


if __name__ == "__main__":
    main()
