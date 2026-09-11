"""Train prevalence and initial-proprio difficulty baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlasafe.outcome_metrics import binary_metrics, fit_logistic_l2, predict_logistic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset)
    initial_rows = data["checkpoint_step"] == 0
    features = data["initial_state"][initial_rows].astype(np.float64)
    target = data["failure"][initial_rows].astype(np.int64)
    splits = data["split"][initial_rows]
    episode_ids = data["episode_id"][initial_rows]
    if len(set(episode_ids.tolist())) != len(episode_ids):
        raise ValueError("checkpoint-0 dataset must contain one row per episode")

    train = splits == "train"
    validation = splits == "validation"
    test = splits == "test"
    mean = features[train].mean(axis=0)
    std = features[train].std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (features - mean) / std
    prevalence = float(target[train].mean())

    candidates = []
    for l2 in (0.001, 0.01, 0.1, 1.0, 10.0):
        weights, bias = fit_logistic_l2(normalized[train], target[train], l2=l2)
        validation_probability = predict_logistic(normalized[validation], weights, bias)
        metrics = binary_metrics(target[validation], validation_probability)
        candidates.append(
            {
                "l2": l2,
                "validation": metrics,
                "weights": weights,
                "bias": bias,
            }
        )
    chosen = max(
        candidates,
        key=lambda item: (item["validation"]["auprc"], -item["validation"]["brier"]),
    )

    report = {
        "dataset": str(args.dataset),
        "selection_rule": "max validation AUPRC; tie-break min validation Brier",
        "chosen_l2": chosen["l2"],
        "candidate_validation_metrics": [
            {"l2": item["l2"], **item["validation"]} for item in candidates
        ],
        "results": {},
    }
    for split_name, selected in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        labels = target[selected]
        constant_probability = np.full(labels.shape, prevalence)
        learned_probability = predict_logistic(
            normalized[selected], chosen["weights"], chosen["bias"]
        )
        report["results"][split_name] = {
            "episodes": int(selected.sum()),
            "failure_rate": float(labels.mean()),
            "prevalence": binary_metrics(labels, constant_probability),
            "initial_proprio_logistic": binary_metrics(labels, learned_probability),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"DIFFICULTY BASELINE {args.output}")
    print(f"chosen_l2: {report['chosen_l2']}")
    for split_name, result in report["results"].items():
        print(split_name, result)


if __name__ == "__main__":
    main()
