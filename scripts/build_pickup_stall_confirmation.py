"""Apply the frozen pickup-stall label rule to an independent confirmation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from build_pickup_stall_pilot import (
        RULE,
        aligned_sample,
        alignment_exclusion_reason,
        split_support,
    )
    from audit_failure_event_feasibility import trajectory
except ModuleNotFoundError:
    from scripts.build_pickup_stall_pilot import (
        RULE,
        aligned_sample,
        alignment_exclusion_reason,
        split_support,
    )
    from scripts.audit_failure_event_feasibility import trajectory


def build_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
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
    samples.sort(key=lambda row: (row["initial_state_id"], row["seed"], row["episode_id"]))
    return samples, exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort_manifest", type=Path)
    parser.add_argument("--event-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    event_protocol = json.loads(args.event_protocol.read_text(encoding="utf-8"))
    confirmation_protocol = json.loads(
        args.confirmation_protocol.read_text(encoding="utf-8")
    )
    if event_protocol["event_rule"]["name"] != RULE:
        raise ValueError(f"event protocol must freeze {RULE}")
    if confirmation_protocol["frozen_event_and_window"]["event"] != RULE:
        raise ValueError(f"confirmation protocol must freeze {RULE}")

    cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    source_rows = [row for rows in cohort["splits"].values() for row in rows]
    expected = int(confirmation_protocol["sampling"]["total_episodes"])
    if len(source_rows) != expected:
        raise ValueError(f"expected {expected} source episodes, got {len(source_rows)}")
    samples, exclusions = build_samples(source_rows)
    support = split_support(samples)
    positive_groups = len(
        {int(row["initial_state_id"]) for row in samples if row["pickup_stall"]}
    )
    gate = confirmation_protocol["formal_gate"]
    gate_status = {
        "minimum_event_positive_episodes": support["positives"]
        >= int(gate["minimum_event_positive_episodes"]),
        "minimum_event_positive_initial_state_groups": positive_groups
        >= int(gate["minimum_event_positive_initial_state_groups"]),
        "minimum_same_state_matched_positive_fraction": (
            support["positive_same_state_negative_fraction"] is not None
            and support["positive_same_state_negative_fraction"]
            >= float(gate["minimum_same_state_matched_positive_fraction"])
        ),
    }
    report = {
        "schema_version": "0.1.0",
        "role": "phase2_pickup_stall_frozen_confirmation",
        "event_protocol": args.event_protocol.as_posix(),
        "confirmation_protocol": args.confirmation_protocol.as_posix(),
        "source_cohort": args.cohort_manifest.as_posix(),
        "group_key": "initial_state_id",
        "samples": len(samples),
        "positives": support["positives"],
        "groups": support["groups"],
        "positive_groups": positive_groups,
        "support": support,
        "formal_gate": {**gate_status, "passed": all(gate_status.values())},
        "excluded": exclusions,
        "splits": {"confirmation": samples},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    counts = {
        reason: sum(row["reason"] == reason for row in exclusions)
        for reason in sorted({row["reason"] for row in exclusions})
    }
    print("PICKUP STALL CONFIRMATION", args.output)
    print("support", support)
    print("positive_groups", positive_groups)
    print("excluded", len(exclusions), counts)
    print("formal_gate", report["formal_gate"])


if __name__ == "__main__":
    main()
