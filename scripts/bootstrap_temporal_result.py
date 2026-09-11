"""Paired episode bootstrap for temporal MLP versus prevalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    predictions = [row for row in report["predictions"] if row["split"] == "test"]
    rng = np.random.default_rng(args.seed)
    result = {
        "source_report": str(args.report),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "unit": "episode (paired within each checkpoint_step)",
        "checkpoints": {},
    }
    for checkpoint in sorted({row["checkpoint_step"] for row in predictions}):
        rows = [row for row in predictions if row["checkpoint_step"] == checkpoint]
        target = np.asarray([row["failure"] for row in rows])
        baseline = np.asarray([row["prevalence_probability"] for row in rows])
        temporal = np.asarray([row["temporal_mlp_probability"] for row in rows])
        observed_baseline = binary_metrics(target, baseline)
        observed_temporal = binary_metrics(target, temporal)
        deltas = {name: [] for name in ("auprc", "auroc", "brier", "ece")}
        for _ in range(args.samples):
            indexes = rng.integers(0, len(rows), size=len(rows))
            sampled_target = target[indexes]
            if len(np.unique(sampled_target)) < 2:
                continue
            sampled_baseline = binary_metrics(sampled_target, baseline[indexes])
            sampled_temporal = binary_metrics(sampled_target, temporal[indexes])
            for name in deltas:
                delta = sampled_temporal[name] - sampled_baseline[name]
                deltas[name].append(delta)
        checkpoint_result = {
            "episodes": len(rows),
            "valid_bootstrap_samples": len(deltas["auprc"]),
            "observed": {
                "prevalence": observed_baseline,
                "temporal_mlp": observed_temporal,
            },
            "delta_temporal_minus_prevalence": {},
        }
        for name, values in deltas.items():
            array = np.asarray(values)
            checkpoint_result["delta_temporal_minus_prevalence"][name] = {
                "point": observed_temporal[name] - observed_baseline[name],
                "ci95": np.percentile(array, [2.5, 97.5]).tolist(),
            }
        result["checkpoints"][str(checkpoint)] = checkpoint_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"BOOTSTRAP {args.output}")
    for checkpoint, values in result["checkpoints"].items():
        print("step", checkpoint, values["delta_temporal_minus_prevalence"])


if __name__ == "__main__":
    main()
