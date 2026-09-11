"""Paired episode bootstrap for frozen checkpoint-vision outcome probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_metrics import binary_metrics


METRICS = ("auprc", "auroc", "brier", "ece")


def indexed_test_predictions(report: dict, probability_key: str) -> dict[tuple[str, int], dict]:
    return {
        (row["episode_id"], int(row["checkpoint_step"])): {
            "failure": int(row["failure"]),
            "probability": float(row[probability_key]),
        }
        for row in report["predictions"]
        if row["split"] == "test"
    }


def paired_delta(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    observed_candidate = binary_metrics(target, candidate)
    observed_baseline = binary_metrics(target, baseline)
    distributions = {name: [] for name in METRICS}
    for _ in range(samples):
        indexes = rng.integers(0, len(target), size=len(target))
        sampled_target = target[indexes]
        if len(np.unique(sampled_target)) < 2:
            continue
        candidate_metrics = binary_metrics(sampled_target, candidate[indexes])
        baseline_metrics = binary_metrics(sampled_target, baseline[indexes])
        for name in METRICS:
            distributions[name].append(candidate_metrics[name] - baseline_metrics[name])
    return {
        "observed_candidate": observed_candidate,
        "observed_baseline": observed_baseline,
        "valid_bootstrap_samples": len(distributions["auprc"]),
        "delta_candidate_minus_baseline": {
            name: {
                "point": observed_candidate[name] - observed_baseline[name],
                "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
            }
            for name, values in distributions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vision_report", type=Path)
    parser.add_argument("temporal_report", type=Path)
    parser.add_argument("difficulty_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    vision_report = json.loads(args.vision_report.read_text(encoding="utf-8"))
    temporal_report = json.loads(args.temporal_report.read_text(encoding="utf-8"))
    difficulty_report = json.loads(args.difficulty_report.read_text(encoding="utf-8"))
    vision_rows = [row for row in vision_report["predictions"] if row["split"] == "test"]
    temporal = indexed_test_predictions(temporal_report, "temporal_mlp_probability")
    difficulty = {
        row["episode_id"]: {
            "failure": int(row["failure"]),
            "probability": float(row["initial_proprio_probability"]),
        }
        for row in difficulty_report["predictions"]
        if row["split"] == "test"
    }
    rng = np.random.default_rng(args.seed)
    result = {
        "vision_report": str(args.vision_report),
        "temporal_report": str(args.temporal_report),
        "difficulty_report": str(args.difficulty_report),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "unit": "episode (paired within checkpoint_step)",
        "metric_direction": "positive favors vision for AUPRC/AUROC; negative favors vision for Brier/ECE",
        "checkpoints": {},
    }
    for checkpoint in sorted({int(row["checkpoint_step"]) for row in vision_rows}):
        rows = [row for row in vision_rows if int(row["checkpoint_step"]) == checkpoint]
        keys = [(row["episode_id"], checkpoint) for row in rows]
        missing_temporal = [key for key in keys if key not in temporal]
        missing_difficulty = [row["episode_id"] for row in rows if row["episode_id"] not in difficulty]
        if missing_temporal or missing_difficulty:
            raise ValueError(
                f"missing paired predictions: temporal={missing_temporal}, difficulty={missing_difficulty}"
            )
        target = np.asarray([int(row["failure"]) for row in rows])
        for row, key in zip(rows, keys, strict=True):
            if temporal[key]["failure"] != int(row["failure"]):
                raise ValueError(f"label mismatch for {key}")
            if difficulty[row["episode_id"]]["failure"] != int(row["failure"]):
                raise ValueError(f"difficulty label mismatch for {row['episode_id']}")
        vision_probability = np.asarray(
            [float(row["checkpoint_vision_probability"]) for row in rows]
        )
        baselines = {
            "initial_frame_vision": np.asarray(
                [float(row["initial_frame_vision_probability"]) for row in rows]
            ),
            "initial_proprio": np.asarray(
                [difficulty[row["episode_id"]]["probability"] for row in rows]
            ),
            "temporal_mlp": np.asarray([temporal[key]["probability"] for key in keys]),
        }
        result["checkpoints"][str(checkpoint)] = {
            name: paired_delta(
                target, vision_probability, probability, args.samples, rng
            )
            for name, probability in baselines.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"VISION BOOTSTRAP {args.output}")
    for checkpoint, comparisons in result["checkpoints"].items():
        for baseline, values in comparisons.items():
            print(
                "step", checkpoint,
                "vs", baseline,
                values["delta_candidate_minus_baseline"],
            )


if __name__ == "__main__":
    main()
