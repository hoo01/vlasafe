"""Evaluate paired A2 fault cohorts with generic rules and command-effect consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlasafe.monitors.command_effect import (
    consistency_errors,
    episode_consistency_score,
    first_alarm_step,
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    lower = 0.0 if successes == 0 else float(max(0.0, centre - radius))
    upper = 1.0 if successes == total else float(min(1.0, centre + radius))
    return [lower, upper]


def load_cohort(root: Path, expected_mode: str, detection_horizon: int) -> dict[int, dict[str, Any]]:
    cohort = {}
    for episode_dir in sorted(root.iterdir()):
        if not episode_dir.is_dir() or not (episode_dir / "COMPLETE").exists():
            continue
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        result = json.loads((episode_dir / "result.json").read_text(encoding="utf-8"))
        mode = metadata["resolved_config"].get("fault_mode", "none")
        if mode != expected_mode:
            raise ValueError(f"{episode_dir}: fault_mode={mode}, expected {expected_mode}")
        initial_state_id = int(metadata["initial_state_id"])
        rows = [json.loads(line) for line in (episode_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) <= detection_horizon:
            raise ValueError(f"{episode_dir}: need at least {detection_horizon + 1} rows to evaluate {detection_horizon} actions")
        early_rows = rows[: detection_horizon + 1]
        errors = consistency_errors(early_rows)
        score = episode_consistency_score(errors)
        if not np.isfinite(score):
            raise ValueError(f"{episode_dir}: insufficient active commands for consistency score")
        range_alarm_steps = [
            int(row["step_id"])
            for row in rows[:detection_horizon]
            if row.get("deploy_metadata", {}).get("action_monitor", {}).get("range_violation")
        ]
        protocol_alarm_steps = [
            int(row["step_id"])
            for row in rows[:detection_horizon]
            if row.get("deploy_metadata", {}).get("action_monitor", {}).get("schema_valid") is False
        ]
        cohort[initial_state_id] = {
            "episode_id": metadata["episode_id"],
            "initial_state_id": initial_state_id,
            "seed": int(metadata["seed"]),
            "success": bool(result["success"]),
            "num_steps": int(result["num_steps"]),
            "consistency_score": score,
            "errors": errors,
            "protocol_rule_alarm": bool(protocol_alarm_steps),
            "protocol_rule_first_alarm_step": protocol_alarm_steps[0] if protocol_alarm_steps else None,
            "range_warning": bool(range_alarm_steps),
            "range_warning_first_step": range_alarm_steps[0] if range_alarm_steps else None,
        }
    if not cohort:
        raise ValueError(f"no complete episodes found in {root}")
    return cohort


def summarize(cohort: dict[int, dict[str, Any]], ids: list[int], threshold: float) -> dict[str, Any]:
    rows = []
    for initial_state_id in ids:
        source = cohort[initial_state_id]
        first = first_alarm_step(source["errors"], threshold)
        rows.append({
            key: value for key, value in source.items() if key != "errors"
        } | {
            "consistency_alarm": first is not None,
            "consistency_first_alarm_step": first,
        })
    consistency_steps = [row["consistency_first_alarm_step"] for row in rows if row["consistency_first_alarm_step"] is not None]
    consistency_detections = sum(row["consistency_alarm"] for row in rows)
    protocol_detections = sum(row["protocol_rule_alarm"] for row in rows)
    range_warnings = sum(row["range_warning"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "consistency_detections": consistency_detections,
        "consistency_detection_rate": consistency_detections / len(rows),
        "consistency_detection_ci95_wilson": wilson_interval(consistency_detections, len(rows)),
        "consistency_first_alarm_median": float(np.median(consistency_steps)) if consistency_steps else None,
        "protocol_rule_detections": protocol_detections,
        "protocol_rule_detection_rate": protocol_detections / len(rows),
        "protocol_rule_detection_ci95_wilson": wilson_interval(protocol_detections, len(rows)),
        "range_warnings": range_warnings,
        "range_warning_rate": range_warnings / len(rows),
        "range_warning_ci95_wilson": wilson_interval(range_warnings, len(rows)),
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--action-swap-root", type=Path, required=True)
    parser.add_argument("--camera-swap-root", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--detection-horizon", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calibration_count <= 0:
        raise ValueError("calibration-count must be positive")
    if args.detection_horizon <= 2:
        raise ValueError("detection-horizon must exceed the three-step window")

    cohorts = {
        "normal": load_cohort(args.normal_root, "none", args.detection_horizon),
        "action_swap_xy": load_cohort(args.action_swap_root, "action_swap_xy", args.detection_horizon),
        "camera_swap": load_cohort(args.camera_swap_root, "camera_swap", args.detection_horizon),
    }
    common_ids = sorted(set.intersection(*(set(value) for value in cohorts.values())))
    if len(common_ids) <= args.calibration_count:
        raise ValueError("not enough paired episodes after reserving normal calibration episodes")
    for initial_state_id in common_ids:
        seeds = {cohort[initial_state_id]["seed"] for cohort in cohorts.values()}
        if len(seeds) != 1:
            raise ValueError(f"paired seed mismatch for initial_state_id={initial_state_id}: {seeds}")
    calibration_ids = common_ids[: args.calibration_count]
    evaluation_ids = common_ids[args.calibration_count :]
    calibration_scores = [cohorts["normal"][index]["consistency_score"] for index in calibration_ids]
    threshold = float(np.nextafter(max(calibration_scores), np.inf))
    output = {
        "schema_version": "0.1.0",
        "protocol": {
            "pairing": "same seed and initial_state_id across regimes",
            "calibration": "first initial_state IDs from normal only",
            "threshold_rule": "next float above maximum normal calibration episode score",
            "threshold": threshold,
            "calibration_ids": calibration_ids,
            "evaluation_ids": evaluation_ids,
            "window": 3,
            "detection_horizon_actions": args.detection_horizon,
            "faults_start_at_step": 0,
            "lead_time_reported": False,
        },
        "calibration_normal": summarize(cohorts["normal"], calibration_ids, threshold),
        "evaluation": {
            name: summarize(cohort, evaluation_ids, threshold)
            for name, cohort in cohorts.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("A2 CONSISTENCY", args.output)
    print("threshold", threshold)
    for name, result in output["evaluation"].items():
        print(
            name,
            "success", f"{result['successes']}/{result['episodes']}",
            "consistency", f"{result['consistency_detections']}/{result['episodes']}",
            "protocol_rule", f"{result['protocol_rule_detections']}/{result['episodes']}",
            "range_warning", f"{result['range_warnings']}/{result['episodes']}",
            "first_alarm_median", result["consistency_first_alarm_median"],
        )


if __name__ == "__main__":
    main()
