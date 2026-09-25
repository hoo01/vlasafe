"""Cluster-resampled uncertainty estimates for episode-level binary metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .outcome_metrics import binary_metrics


def cluster_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    indexes_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    point = binary_metrics(labels, probability)
    distributions = {name: [] for name in ("auprc", "auroc", "brier", "ece")}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([indexes_by_group[group] for group in sampled_groups])
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        metrics = binary_metrics(sampled_labels, probability[index])
        for name in distributions:
            distributions[name].append(metrics[name])
    return {
        name: {
            "point": point[name],
            "valid_samples": len(distributions[name]),
            "ci95": np.percentile(distributions[name], [2.5, 97.5]).tolist(),
        }
        for name in distributions
    }
