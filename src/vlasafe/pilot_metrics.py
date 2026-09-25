"""Grouped evaluation helpers for the pickup-stall pilot."""

from __future__ import annotations

from typing import Any

import numpy as np

from .outcome_metrics import binary_metrics


METRICS = ("auprc", "auroc", "brier", "ece")


def matched_pair_accuracy(
    target: np.ndarray, probability: np.ndarray, group: np.ndarray
) -> dict[str, Any]:
    correct = total = 0
    mixed = 0
    for value in np.unique(group):
        selected = group == value
        positive = probability[selected & (target == 1)]
        negative = probability[selected & (target == 0)]
        if not len(positive) or not len(negative):
            continue
        mixed += 1
        comparison = positive[:, None] - negative[None, :]
        correct += int((comparison > 0).sum())
        total += int(comparison.size)
    return {
        "mixed_groups": mixed,
        "pairs": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def cluster_bootstrap(
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    distributions = {
        name: {metric: [] for metric in METRICS} for name in probabilities
    }
    deltas = {
        name: {metric: [] for metric in METRICS}
        for name in probabilities
        if name not in {"prevalence", "approach_time"}
    }
    valid = 0
    for _ in range(samples):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(groups == value) for value in sampled_groups]
        )
        sampled_target = target[indexes]
        if len(np.unique(sampled_target)) < 2:
            continue
        valid += 1
        metrics = {
            name: binary_metrics(sampled_target, probability[indexes])
            for name, probability in probabilities.items()
        }
        for name in probabilities:
            for metric in METRICS:
                distributions[name][metric].append(metrics[name][metric])
        for name in deltas:
            for metric in METRICS:
                deltas[name][metric].append(
                    metrics[name][metric] - metrics["approach_time"][metric]
                )
    observed = {
        name: binary_metrics(target, probability)
        for name, probability in probabilities.items()
    }
    return {
        "unit": "initial_state_id cluster",
        "requested_samples": samples,
        "valid_samples": valid,
        "metrics": {
            name: {
                metric: {
                    "point": observed[name][metric],
                    "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
                }
                for metric, values in distributions[name].items()
            }
            for name in probabilities
        },
        "delta_vs_approach_time": {
            name: {
                metric: {
                    "point": observed[name][metric] - observed["approach_time"][metric],
                    "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
                }
                for metric, values in metric_values.items()
            }
            for name, metric_values in deltas.items()
        },
    }
