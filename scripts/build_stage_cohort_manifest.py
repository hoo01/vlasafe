"""Freeze and validate the predeclared v0.3 task-stage cohort manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlasafe.rollout.validator import validate_episode


def expected_grid(protocol: dict[str, Any]) -> set[tuple[int, int]]:
    sampling = protocol["sampling"]
    start = int(sampling["preset_initial_state_ids"]["start_inclusive"])
    stop = int(sampling["preset_initial_state_ids"]["stop_exclusive"])
    seed_starts = [int(value) for value in sampling["seed_starts"]]
    return {
        (state, seed_start + state - start)
        for seed_start in seed_starts
        for state in range(start, stop)
    }


def validate_declared_grid(
    protocol: dict[str, Any], episodes: list[dict[str, Any]]
) -> None:
    declared = expected_grid(protocol)
    observed = [(int(row["initial_state_id"]), int(row["seed"])) for row in episodes]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate (initial_state_id, seed) pair in stage cohort")
    observed_set = set(observed)
    if observed_set != declared:
        missing = sorted(declared - observed_set)
        extra = sorted(observed_set - declared)
        raise ValueError(f"stage cohort grid mismatch: missing={missing}, extra={extra}")
    expected_total = int(protocol["sampling"]["total_episodes"])
    if len(episodes) != expected_total:
        raise ValueError(f"expected {expected_total} episodes, got {len(episodes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    identity = protocol["identity"]
    episode_dirs = sorted(
        path
        for path in args.episodes_root.glob("smolvla-*")
        if path.is_dir() and not path.name.endswith(".incomplete")
    )
    episodes = []
    revisions: set[str] = set()
    for episode_dir in episode_dirs:
        validation = validate_episode(episode_dir)
        metadata = json.loads(
            (episode_dir / "metadata.json").read_text(encoding="utf-8")
        )
        first_row = json.loads(
            (episode_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        config = metadata["resolved_config"]
        if metadata["task"] != identity["task"] or int(metadata["task_id"]) != int(identity["task_id"]):
            raise ValueError(f"{episode_dir.name}: task identity mismatch")
        if config.get("fault_mode") != identity["fault_mode"]:
            raise ValueError(f"{episode_dir.name}: fault mode mismatch")
        if int(config.get("max_steps")) != int(identity["max_steps"]):
            raise ValueError(f"{episode_dir.name}: max_steps mismatch")
        event_version = first_row["label_only"].get("event_schema_version")
        if event_version != identity["event_schema_version"]:
            raise ValueError(
                f"{episode_dir.name}: event schema {event_version!r} does not match protocol"
            )
        revision = str(config["vlasafe_revision"])
        revisions.add(revision)
        episodes.append(
            {
                "episode_id": str(metadata["episode_id"]),
                "path": episode_dir.as_posix(),
                "seed": int(metadata["seed"]),
                "initial_state_id": int(metadata["initial_state_id"]),
                "success": bool(validation.success),
                "num_steps": int(validation.num_steps),
                "event_schema_version": event_version,
            }
        )

    validate_declared_grid(protocol, episodes)
    if len(revisions) != 1:
        raise ValueError(f"stage cohort spans multiple code revisions: {sorted(revisions)}")
    episodes.sort(key=lambda row: (row["initial_state_id"], row["seed"]))
    totals = {
        "episodes": len(episodes),
        "successes": sum(row["success"] for row in episodes),
        "failures": sum(not row["success"] for row in episodes),
        "initial_states": len({row["initial_state_id"] for row in episodes}),
        "repetitions_per_initial_state": int(
            protocol["sampling"]["repetitions_per_initial_state"]
        ),
    }
    manifest = {
        "schema_version": "0.1.0",
        "role": "stage_aligned_mechanism_confirmation_only",
        "source_protocol": args.protocol.as_posix(),
        "selection": protocol["sampling"]["selection_rule"],
        "group_key": "initial_state_id",
        "identity": {
            **identity,
            "vlasafe_revision": next(iter(revisions)),
        },
        "totals": totals,
        "splits": {"stage_confirmation": episodes},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("STAGE COHORT MANIFEST", args.output)
    print("totals", totals)
    print("vlasafe_revision", next(iter(revisions)))
    print(
        "step120_coverage",
        sum(row["num_steps"] > 120 for row in episodes),
        "/",
        len(episodes),
    )


if __name__ == "__main__":
    main()
