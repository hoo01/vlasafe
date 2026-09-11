"""Validate rollout cohorts and freeze a group-disjoint outcome split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

from vlasafe.rollout.validator import validate_episode


SPLIT_NAMES = ("train", "validation", "test")


def _take_exact(
    group_ids: list[str], group_sizes: dict[str, int], target: int
) -> tuple[list[str], list[str]] | None:
    chosen: list[str] = []
    chosen_set: set[str] = set()
    remaining = target
    for group_id in group_ids:
        size = group_sizes[group_id]
        if size <= remaining:
            chosen.append(group_id)
            chosen_set.add(group_id)
            remaining -= size
        if remaining == 0:
            return chosen, [item for item in group_ids if item not in chosen_set]
    return None


def choose_group_split(
    episodes: list[dict[str, Any]], seed: int, attempts: int = 20_000
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[str(episode["initial_state_id"])].append(episode)
    group_ids = sorted(groups)
    group_sizes = {group_id: len(groups[group_id]) for group_id in group_ids}
    targets = {
        "train": round(len(episodes) * 0.6),
        "validation": round(len(episodes) * 0.2),
        "test": len(episodes) - round(len(episodes) * 0.6) - round(len(episodes) * 0.2),
    }
    overall_rate = sum(item["success"] for item in episodes) / len(episodes)
    rng = Random(seed)
    best: tuple[float, dict[str, list[dict[str, Any]]]] | None = None

    for _ in range(attempts):
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)
        train_pick = _take_exact(shuffled, group_sizes, targets["train"])
        if train_pick is None:
            continue
        train_ids, remaining = train_pick
        validation_pick = _take_exact(remaining, group_sizes, targets["validation"])
        if validation_pick is None:
            continue
        validation_ids, test_ids = validation_pick
        assignment = {
            "train": [item for group_id in train_ids for item in groups[group_id]],
            "validation": [item for group_id in validation_ids for item in groups[group_id]],
            "test": [item for group_id in test_ids for item in groups[group_id]],
        }
        if any(len(assignment[name]) != targets[name] for name in SPLIT_NAMES):
            continue
        if any(
            len({item["success"] for item in assignment[name]}) < 2
            for name in SPLIT_NAMES
        ):
            continue
        score = sum(
            abs(
                sum(item["success"] for item in assignment[name])
                / len(assignment[name])
                - overall_rate
            )
            for name in SPLIT_NAMES
        )
        if best is None or score < best[0]:
            best = (score, assignment)

    if best is None:
        raise RuntimeError("could not find an exact group-disjoint stratified split")
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        nargs=2,
        action="append",
        metavar=("SUMMARY", "EPISODES_ROOT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    episodes: list[dict[str, Any]] = []
    cohort_identity: dict[str, Any] | None = None
    seen_episode_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for summary_text, root_text in args.cohort:
        summary_path = Path(summary_text)
        episodes_root = Path(root_text)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = {
            "task": summary["task"],
            "task_id": summary["task_id"],
            "vlasafe_revision": summary["vlasafe_revision"],
            "provenance": summary["provenance"],
        }
        if cohort_identity is None:
            cohort_identity = identity
        elif identity != cohort_identity:
            raise ValueError(f"cohort identity mismatch: {summary_path}")

        for item in summary["episodes"]:
            episode_id = item["episode_id"]
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            episode_dir = episodes_root / episode_id
            validation = validate_episode(episode_dir)
            metadata = json.loads(
                (episode_dir / "metadata.json").read_text(encoding="utf-8")
            )
            seed = int(metadata["seed"])
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed: {seed}")
            seen_episode_ids.add(episode_id)
            seen_seeds.add(seed)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "path": episode_dir.as_posix(),
                    "seed": seed,
                    "initial_state_id": metadata["initial_state_id"],
                    "success": validation.success,
                    "num_steps": validation.num_steps,
                }
            )

    if len(episodes) < 50:
        raise ValueError(f"Week-1 gate requires at least 50 episodes, got {len(episodes)}")
    failures = sum(not item["success"] for item in episodes)
    if failures < 20:
        raise ValueError(f"Week-1 gate requires at least 20 failures, got {failures}")

    assignment = choose_group_split(episodes, args.seed)
    manifest = {
        "schema_version": "0.1.0",
        "split_seed": args.seed,
        "group_key": "initial_state_id",
        "stratify_key": "success",
        "identity": cohort_identity,
        "totals": {
            "episodes": len(episodes),
            "successes": sum(item["success"] for item in episodes),
            "failures": failures,
        },
        "splits": assignment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"FROZEN SPLIT {args.output}")
    print("split episodes success failure initial_state_ids")
    for name in SPLIT_NAMES:
        rows = assignment[name]
        successes = sum(item["success"] for item in rows)
        initial_states = sorted({item["initial_state_id"] for item in rows})
        print(name, len(rows), successes, len(rows) - successes, initial_states)


if __name__ == "__main__":
    main()
