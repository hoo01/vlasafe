"""Paired episode bootstrap for temporal MLP versus initial difficulty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("temporal_report", type=Path)
    parser.add_argument("difficulty_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    temporal_report = json.loads(args.temporal_report.read_text(encoding="utf-8"))
    difficulty_report = json.loads(args.difficulty_report.read_text(encoding="utf-8"))
    difficulty = {
        row["episode_id"]: row
        for row in difficulty_report["predictions"]
        if row["split"] == "test"
    }
    temporal = [row for row in temporal_report["predictions"] if row["split"] == "test"]
    rng = np.random.default_rng(args.seed)
    result = {
        "temporal_report": str(args.temporal_report),
        "difficulty_report": str(args.difficulty_report),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "unit": "episode (paired within each checkpoint_step)",
        "checkpoints": {},
    }
    for checkpoint in sorted({row["checkpoint_step"] for row in temporal}):
        rows = [row for row in temporal if row["checkpoint_step"] == checkpoint]
        missing = {row["episode_id"] for row in rows} - difficulty.keys()
        if missing:
            raise ValueError(f"difficulty report is missing episodes: {sorted(missing)}")
        target = np.asarray([row["failure"] for row in rows])
        temporal_probability = np.asarray([row["temporal_mlp_probability"] for row in rows])
        difficulty_probability = np.asarray(
            [difficulty[row["episode_id"]]["initial_proprio_probability"] for row in rows]
        )
        observed_temporal = binary_metrics(target, temporal_probability)
        observed_difficulty = binary_metrics(target, difficulty_probability)
        deltas = {name: [] for name in ("auprc", "auroc", "brier", "ece")}
        for _ in range(args.samples):
            indexes = rng.integers(0, len(rows), size=len(rows))
            sampled_target = target[indexes]
            if len(np.unique(sampled_target)) < 2:
                continue
            temporal_metrics = binary_metrics(sampled_target, temporal_probability[indexes])
            difficulty_metrics = binary_metrics(sampled_target, difficulty_probability[indexes])
            for name in deltas:
                deltas[name].append(temporal_metrics[name] - difficulty_metrics[name])
        comparison = {
            name: {
                "point": observed_temporal[name] - observed_difficulty[name],
                "ci95": np.percentile(np.asarray(values), [2.5, 97.5]).tolist(),
            }
            for name, values in deltas.items()
        }
        result["checkpoints"][str(checkpoint)] = {
            "episodes": len(rows),
            "valid_bootstrap_samples": len(deltas["auprc"]),
            "observed": {
                "initial_proprio": observed_difficulty,
                "temporal_mlp": observed_temporal,
            },
            "delta_temporal_minus_initial_proprio": comparison,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PAIRED BOOTSTRAP {args.output}")
    for checkpoint, values in result["checkpoints"].items():
        print("step", checkpoint, values["delta_temporal_minus_initial_proprio"])


if __name__ == "__main__":
    main()
