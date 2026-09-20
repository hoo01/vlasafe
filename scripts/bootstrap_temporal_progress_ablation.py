"""Paired episode bootstrap for the temporal checkpoint-progress feature ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlasafe.outcome_metrics import binary_metrics


METRICS = ("auprc", "auroc", "brier", "ece")


def _checkpoint_progress_feature(report: dict[str, Any]) -> bool | None:
    """Read the feature declaration, including reports created before the flag existed."""
    if "checkpoint_progress_feature" in report:
        return bool(report["checkpoint_progress_feature"])
    normalization = report.get("normalization", {})
    if normalization.get("progress_denominator") is not None:
        return True
    return None


def _test_index(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["episode_id"]), int(row["checkpoint_step"])): row
        for row in report["predictions"]
        if row["split"] == "test"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("with_progress", type=Path)
    parser.add_argument("without_progress", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    with_report = json.loads(args.with_progress.read_text(encoding="utf-8"))
    without_report = json.loads(args.without_progress.read_text(encoding="utf-8"))
    if _checkpoint_progress_feature(with_report) is not True:
        raise ValueError("with_progress report does not declare checkpoint progress")
    if _checkpoint_progress_feature(without_report) is not False:
        raise ValueError("without_progress report still declares checkpoint progress")
    with_index = _test_index(with_report)
    without_index = _test_index(without_report)
    if with_index.keys() != without_index.keys():
        raise ValueError("paired reports contain different test episode/checkpoint keys")

    rng = np.random.default_rng(args.seed)
    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "with_progress_report": str(args.with_progress),
        "without_progress_report": str(args.without_progress),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "unit": "episode paired within checkpoint_step",
        "selection": "diagnostic ablation; test is not used to select a replacement model",
        "checkpoints": {},
    }
    checkpoints = sorted({checkpoint for _, checkpoint in with_index})
    for checkpoint in checkpoints:
        keys = sorted(key for key in with_index if key[1] == checkpoint)
        target = np.asarray([int(with_index[key]["failure"]) for key in keys])
        paired_target = np.asarray([int(without_index[key]["failure"]) for key in keys])
        if not np.array_equal(target, paired_target):
            raise ValueError(f"label mismatch at checkpoint {checkpoint}")
        with_probability = np.asarray(
            [float(with_index[key]["temporal_mlp_probability"]) for key in keys]
        )
        without_probability = np.asarray(
            [float(without_index[key]["temporal_mlp_probability"]) for key in keys]
        )
        observed_with = binary_metrics(target, with_probability)
        observed_without = binary_metrics(target, without_probability)
        samples = {name: [] for name in METRICS}
        for _ in range(args.samples):
            indexes = rng.integers(0, len(keys), size=len(keys))
            sampled_target = target[indexes]
            if len(np.unique(sampled_target)) < 2:
                continue
            sampled_with = binary_metrics(sampled_target, with_probability[indexes])
            sampled_without = binary_metrics(sampled_target, without_probability[indexes])
            for name in METRICS:
                samples[name].append(sampled_without[name] - sampled_with[name])
        deltas = {}
        for name, values in samples.items():
            array = np.asarray(values)
            deltas[name] = {
                "point": observed_without[name] - observed_with[name],
                "ci95": np.percentile(array, [2.5, 97.5]).tolist(),
            }
        result["checkpoints"][str(checkpoint)] = {
            "episodes": len(keys),
            "valid_bootstrap_samples": len(samples["auprc"]),
            "with_checkpoint_progress": observed_with,
            "without_checkpoint_progress": observed_without,
            "delta_without_minus_with": deltas,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TEMPORAL PROGRESS ABLATION", args.output)
    for checkpoint, values in result["checkpoints"].items():
        print("step", checkpoint, values["delta_without_minus_with"])


if __name__ == "__main__":
    main()
