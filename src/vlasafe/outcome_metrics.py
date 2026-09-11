"""Small dependency-free metrics for episode-level binary outcome evaluation."""

from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, probability: np.ndarray, ece_bins: int = 5) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if target.size != probability.size or target.size == 0:
        raise ValueError("target and probability must be non-empty and equally sized")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability must be finite and within [0, 1]")

    positives = probability[target == 1]
    negatives = probability[target == 0]
    if positives.size and negatives.size:
        comparisons = positives[:, None] - negatives[None, :]
        auroc = float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())
    else:
        auroc = float("nan")

    positive_count = int(target.sum())
    if positive_count:
        thresholds = np.unique(probability)[::-1]
        previous_recall = 0.0
        auprc = 0.0
        for threshold in thresholds:
            predicted = probability >= threshold
            true_positive = int((target[predicted] == 1).sum())
            precision = true_positive / int(predicted.sum())
            recall = true_positive / positive_count
            auprc += (recall - previous_recall) * precision
            previous_recall = recall
    else:
        auprc = float("nan")

    brier = float(np.mean((probability - target) ** 2))
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for index in range(ece_bins):
        if index == ece_bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return {"auprc": float(auprc), "auroc": auroc, "brier": brier, "ece": float(ece)}


def fit_logistic_l2(
    features: np.ndarray,
    target: np.ndarray,
    l2: float,
    steps: int = 3000,
    learning_rate: float = 0.03,
) -> tuple[np.ndarray, float]:
    """Fit deterministic L2 logistic regression with Adam."""

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = 0.0
    second_b = 0.0
    for iteration in range(1, steps + 1):
        logits = np.clip(features @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - target
        gradient_w = features.T @ error / len(target) + l2 * weights
        gradient_b = float(error.mean())
        first_w = 0.9 * first_w + 0.1 * gradient_w
        second_w = 0.999 * second_w + 0.001 * gradient_w**2
        first_b = 0.9 * first_b + 0.1 * gradient_b
        second_b = 0.999 * second_b + 0.001 * gradient_b**2
        correction_w = first_w / (1.0 - 0.9**iteration)
        correction_w2 = second_w / (1.0 - 0.999**iteration)
        correction_b = first_b / (1.0 - 0.9**iteration)
        correction_b2 = second_b / (1.0 - 0.999**iteration)
        weights -= learning_rate * correction_w / (np.sqrt(correction_w2) + 1e-8)
        bias -= learning_rate * correction_b / (np.sqrt(correction_b2) + 1e-8)
    return weights, bias


def predict_logistic(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(np.asarray(features) @ weights + bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))
