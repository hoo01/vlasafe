"""Evaluate validation-selected offline outcome decision utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def index_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(episode["episode_id"]): {**episode, "split": split}
        for split, episodes in manifest["splits"].items()
        for episode in episodes
    }


def index_predictions(report: dict[str, Any]) -> dict[str, dict[int, float]]:
    indexed: dict[str, dict[int, float]] = {}
    for row in report["predictions"]:
        indexed.setdefault(str(row["episode_id"]), {})[
            int(row["checkpoint_step"])
        ] = float(row["checkpoint_vision_probability"])
    return indexed


def evaluate_policy(
    episodes: list[dict[str, Any]],
    predictions: dict[str, dict[int, float]],
    checkpoints: list[int],
    threshold: float,
) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        num_steps = int(episode["num_steps"])
        failure = not bool(episode["success"])
        available = [step for step in checkpoints if step < num_steps]
        missing = [step for step in available if step not in predictions.get(episode_id, {})]
        if missing:
            raise ValueError(f"{episode_id} missing predictions at checkpoints {missing}")
        stop_step = next(
            (
                step
                for step in available
                if predictions[episode_id][step] >= threshold
            ),
            None,
        )
        saved_steps = num_steps - (stop_step + 1) if stop_step is not None else 0
        rows.append(
            {
                "episode_id": episode_id,
                "failure": failure,
                "num_steps": num_steps,
                "stop_step": stop_step,
                "saved_steps": saved_steps,
            }
        )
    failures = [row for row in rows if row["failure"]]
    successes = [row for row in rows if not row["failure"]]
    stopped = [row for row in rows if row["stop_step"] is not None]
    stopped_failures = [row for row in failures if row["stop_step"] is not None]
    sacrificed = [row for row in successes if row["stop_step"] is not None]
    total_saved = sum(row["saved_steps"] for row in rows)
    failure_saved = sum(row["saved_steps"] for row in failures)
    return {
        "episodes": len(rows),
        "failures": len(failures),
        "successes": len(successes),
        "terminated_episodes": len(stopped),
        "termination_rate": len(stopped) / len(rows) if rows else None,
        "detected_failures": len(stopped_failures),
        "failure_detection_rate": len(stopped_failures) / len(failures) if failures else None,
        "sacrificed_successes": len(sacrificed),
        "false_termination_rate": len(sacrificed) / len(successes) if successes else None,
        "total_saved_steps": total_saved,
        "mean_saved_steps_all_episodes": total_saved / len(rows) if rows else None,
        "failure_saved_steps": failure_saved,
        "mean_saved_steps_per_failure": failure_saved / len(failures) if failures else None,
        "episodes_detail": rows,
    }


def choose_threshold(
    episodes: list[dict[str, Any]],
    predictions: dict[str, dict[int, float]],
    checkpoints: list[int],
) -> tuple[float, dict[str, Any], int]:
    probabilities = [
        predictions[str(episode["episode_id"])][step]
        for episode in episodes
        for step in checkpoints
        if step < int(episode["num_steps"])
    ]
    if not probabilities:
        raise ValueError("validation split contains no available predictions")
    candidates = sorted(set(probabilities + [float(np.nextafter(max(probabilities), np.inf))]))
    feasible = []
    for threshold in candidates:
        result = evaluate_policy(episodes, predictions, checkpoints, threshold)
        if result["sacrificed_successes"] == 0:
            feasible.append((threshold, result))
    if not feasible:
        raise RuntimeError("no threshold satisfies zero sacrificed validation successes")
    threshold, result = max(
        feasible,
        key=lambda item: (
            item[1]["failure_saved_steps"],
            item[1]["detected_failures"],
            item[0],
        ),
    )
    return threshold, result, len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vision_report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 40, 80, 120])
    parser.add_argument("--milliseconds-per-step", type=float, required=True)
    args = parser.parse_args()

    if args.milliseconds_per_step <= 0:
        raise ValueError("milliseconds-per-step must be positive")
    checkpoints = sorted(set(args.checkpoint_steps))
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("checkpoint steps must be non-negative")

    report = json.loads(args.vision_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episode_index = index_manifest(manifest)
    predictions = index_predictions(report)
    validation = [episode for episode in episode_index.values() if episode["split"] == "validation"]
    test = [episode for episode in episode_index.values() if episode["split"] == "test"]
    threshold, validation_result, candidate_count = choose_threshold(
        validation, predictions, checkpoints
    )
    test_result = evaluate_policy(test, predictions, checkpoints, threshold)
    for result in (validation_result, test_result):
        result["estimated_saved_wall_seconds"] = (
            result["total_saved_steps"] * args.milliseconds_per_step / 1000.0
        )

    output = {
        "vision_report": str(args.vision_report),
        "manifest": str(args.manifest),
        "policy": "stop at the first scheduled checkpoint with failure probability >= threshold",
        "interpretation": "offline counterfactual efficiency bound; not a safety intervention",
        "checkpoint_steps": checkpoints,
        "threshold_selection": {
            "split": "validation",
            "constraint": "zero sacrificed validation successes",
            "objective": "maximize failure saved steps; tie-break detected failures, then higher threshold",
            "candidate_thresholds": candidate_count,
            "selected_threshold": threshold,
            "test_used_for_selection": False,
        },
        "milliseconds_per_step": args.milliseconds_per_step,
        "validation": validation_result,
        "test": test_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"OUTCOME UTILITY {args.output}")
    print("selected validation threshold:", threshold)
    for split in ("validation", "test"):
        values = output[split]
        print(
            split,
            "detected_failures", f"{values['detected_failures']}/{values['failures']}",
            "sacrificed_successes", f"{values['sacrificed_successes']}/{values['successes']}",
            "saved_steps", values["total_saved_steps"],
            "estimated_saved_wall_seconds", values["estimated_saved_wall_seconds"],
        )


if __name__ == "__main__":
    main()
