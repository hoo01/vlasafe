"""Build the frozen, first-approach-aligned Phase-2 pickup-stall pilot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from audit_failure_event_feasibility import trajectory
except ModuleNotFoundError:
    from scripts.audit_failure_event_feasibility import trajectory


RULE = "pickup_stall_approach0.10_w40_move0.04"


def allocate_groups(
    mixed_groups: list[int], all_positive_groups: list[int], negative_only_groups: list[int], seed: int
) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    result: dict[int, str] = {}
    for groups in (mixed_groups, negative_only_groups):
        values = np.asarray(sorted(groups), dtype=np.int64)
        rng.shuffle(values)
        count = len(values)
        evaluation_count = max(1, int(round(0.2 * count))) if count >= 3 else 0
        train_end = count - 2 * evaluation_count
        validation_end = train_end + evaluation_count
        for index, group in enumerate(values.tolist()):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            result[int(group)] = split
    for group in all_positive_groups:
        result[int(group)] = "train"
    return result


def aligned_sample(
    data: dict[str, Any], history: int, wait: int, horizon: int, movement: float
) -> dict[str, Any] | None:
    approached = np.flatnonzero(data["bowl_eef_distance"] <= 0.10)
    if not len(approached):
        return None
    approach = int(approached[0])
    event = approach + wait
    checkpoint = event - horizon
    if event >= len(data["bowl"]) or checkpoint - history + 1 < 0:
        return None
    displacement = np.linalg.norm(
        data["bowl"][approach : event + 1] - data["bowl"][approach], axis=1
    )
    maximum = float(np.max(displacement))
    crossings = np.flatnonzero(displacement >= movement)
    crossing_step = approach + int(crossings[0]) if len(crossings) else None
    return {
        "first_approach_step": approach,
        "checkpoint_step": checkpoint,
        "event_step": event if maximum < movement else None,
        "pickup_stall": int(maximum < movement),
        "maximum_target_movement_m": maximum,
        "target_movement_crossing_step": crossing_step,
        "checkpoint_stage_eligible": crossing_step is None or crossing_step > checkpoint,
    }


def alignment_exclusion_reason(
    data: dict[str, Any], history: int, wait: int, horizon: int
) -> str | None:
    approached = np.flatnonzero(data["bowl_eef_distance"] <= 0.10)
    if not len(approached):
        return "no_first_approach"
    approach = int(approached[0])
    event = approach + wait
    checkpoint = event - horizon
    if event >= len(data["bowl"]):
        return "incomplete_approach_window"
    if checkpoint - history + 1 < 0:
        return "insufficient_history"
    return None


def split_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(int(row["initial_state_id"]), []).append(row)
    mixed = {
        group
        for group, values in by_group.items()
        if any(row["pickup_stall"] for row in values)
        and any(not row["pickup_stall"] for row in values)
    }
    positives = [row for row in rows if row["pickup_stall"]]
    matched = [row for row in positives if row["initial_state_id"] in mixed]
    return {
        "samples": len(rows),
        "positives": len(positives),
        "negatives": len(rows) - len(positives),
        "groups": len(by_group),
        "mixed_outcome_groups": len(mixed),
        "positives_with_same_state_negative": len(matched),
        "positive_same_state_negative_fraction": len(matched) / len(positives)
        if positives
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort_manifest", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["event_rule"]["name"] != RULE:
        raise ValueError(f"protocol must freeze {RULE}")
    cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    source_rows = [row for rows in cohort["splits"].values() for row in rows]
    samples = []
    exclusions = []
    for row in source_rows:
        episode_path = Path(row["path"])
        steps = [
            json.loads(line)
            for line in (episode_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        data = trajectory(steps)
        reason = alignment_exclusion_reason(data, history=16, wait=40, horizon=20)
        sample = aligned_sample(data, history=16, wait=40, horizon=20, movement=0.04)
        if sample is None:
            exclusions.append(
                {
                    "episode_id": str(row["episode_id"]),
                    "initial_state_id": int(row["initial_state_id"]),
                    "reason": reason,
                }
            )
            continue
        if not sample["checkpoint_stage_eligible"]:
            exclusions.append(
                {
                    "episode_id": str(row["episode_id"]),
                    "initial_state_id": int(row["initial_state_id"]),
                    "reason": "target_moved_before_checkpoint",
                    "checkpoint_step": int(sample["checkpoint_step"]),
                    "target_movement_crossing_step": int(sample["target_movement_crossing_step"]),
                }
            )
            continue
        samples.append(
            {
                "episode_id": str(row["episode_id"]),
                "path": row["path"],
                "initial_state_id": int(row["initial_state_id"]),
                "seed": int(row["seed"]),
                "episode_success": bool(row["success"]),
                **sample,
            }
        )
    all_groups = sorted({row["initial_state_id"] for row in samples})
    rows_by_group: dict[int, list[dict[str, Any]]] = {}
    for row in samples:
        rows_by_group.setdefault(row["initial_state_id"], []).append(row)
    mixed_groups = sorted(
        group for group, rows in rows_by_group.items()
        if any(row["pickup_stall"] for row in rows)
        and any(not row["pickup_stall"] for row in rows)
    )
    all_positive_groups = sorted(
        group for group, rows in rows_by_group.items()
        if all(row["pickup_stall"] for row in rows)
    )
    negative_only_groups = sorted(
        group for group, rows in rows_by_group.items()
        if not any(row["pickup_stall"] for row in rows)
    )
    group_split = allocate_groups(
        mixed_groups,
        all_positive_groups,
        negative_only_groups,
        int(protocol["split"]["seed"]),
    )
    splits = {"train": [], "validation": [], "test": []}
    for row in samples:
        split = group_split[row["initial_state_id"]]
        splits[split].append(row)
    for rows in splits.values():
        rows.sort(key=lambda row: (row["initial_state_id"], row["seed"], row["episode_id"]))
    support = {split: split_support(rows) for split, rows in splits.items()}
    report = {
        "schema_version": "0.1.0",
        "role": "phase2_pickup_stall_pilot",
        "protocol": args.protocol.as_posix(),
        "source_cohort": args.cohort_manifest.as_posix(),
        "group_key": "initial_state_id",
        "samples": len(samples),
        "positives": sum(row["pickup_stall"] for row in samples),
        "groups": len(all_groups),
        "group_strata": {
            "mixed": mixed_groups,
            "all_positive": all_positive_groups,
            "negative_only": negative_only_groups,
        },
        "excluded": exclusions,
        "support": support,
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PICKUP STALL PILOT", args.output)
    print("samples", report["samples"], "positives", report["positives"], "groups", report["groups"])
    for split, rows in splits.items():
        summary = support[split]
        print(split, summary)
    print("excluded", len(exclusions), {reason: sum(row["reason"] == reason for row in exclusions) for reason in sorted({row["reason"] for row in exclusions})})


if __name__ == "__main__":
    main()
