"""Test whether frozen outcome risk retains signal after a train-defined progress proxy."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from vlasafe.outcome_metrics import binary_metrics


def load_predictions(path: Path, probability_key: str) -> dict[tuple[str, int], float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["episode_id"]), int(row["checkpoint_step"])): float(row[probability_key])
        for row in report["predictions"]
    }


def build_progress_proxy(data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    raw = data["features"].astype(np.float64).reshape(len(data["features"]), -1)
    split = data["split"]
    step = data["checkpoint_step"].astype(int)
    train = split == "train"
    mean = raw[train].mean(axis=0)
    std = raw[train].std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (raw - mean) / std
    start = normalized[train & (step == 0)].mean(axis=0)
    end = normalized[train & (step == 120)].mean(axis=0)
    axis = end - start
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("train-defined progress axis has zero norm")
    axis /= norm
    score = normalized @ axis
    return score, {
        "definition": "projection onto train mean(step120)-mean(step0) frozen-feature direction",
        "uses_outcome_labels": False,
        "uses_test_to_define_proxy": False,
        "feature_dim": int(raw.shape[1]),
        "axis_norm_before_unit_scaling": norm,
    }


def fit_residualization(train_progress: np.ndarray, train_risk: np.ndarray) -> list[float]:
    design = np.column_stack([np.ones(len(train_progress)), train_progress])
    coefficients = np.linalg.lstsq(design, train_risk, rcond=None)[0]
    return coefficients.tolist()


def apply_residualization(progress: np.ndarray, risk: np.ndarray, coefficients: list[float]) -> np.ndarray:
    return risk - (coefficients[0] + coefficients[1] * progress)


def rank_probability(score: np.ndarray) -> np.ndarray:
    """Map an unconstrained score to (0, 1) without changing its ranking."""
    clipped = np.clip(np.asarray(score, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def ranking_metrics(outcome: np.ndarray, score: np.ndarray) -> dict[str, float]:
    metrics = binary_metrics(outcome, rank_probability(score))
    return {"auprc": metrics["auprc"], "auroc": metrics["auroc"]}


def partial_correlation(risk: np.ndarray, outcome: np.ndarray, progress: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(progress)), progress])
    risk_residual = risk - design @ np.linalg.lstsq(design, risk, rcond=None)[0]
    outcome_residual = outcome - design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
    if np.std(risk_residual) < 1e-12 or np.std(outcome_residual) < 1e-12:
        return float("nan")
    return float(np.corrcoef(risk_residual, outcome_residual)[0, 1])


def closest_one_to_one_matches(
    episode_ids: np.ndarray,
    outcome: np.ndarray,
    progress: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, Any]]:
    successes = np.flatnonzero(outcome == 0)
    failures = np.flatnonzero(outcome == 1)
    if len(successes) > len(failures):
        raise ValueError("matching expects no more successes than failures")
    best: tuple[float, tuple[int, ...]] | None = None
    for selected in itertools.permutations(failures.tolist(), len(successes)):
        cost = sum(abs(progress[s] - progress[f]) for s, f in zip(successes, selected, strict=True))
        if best is None or cost < best[0]:
            best = (float(cost), selected)
    if best is None:
        return []
    return [
        {
            "success_episode": str(episode_ids[s]),
            "failure_episode": str(episode_ids[f]),
            "absolute_progress_gap": float(abs(progress[s] - progress[f])),
            "success_risk": float(risk[s]),
            "failure_risk": float(risk[f]),
            "risk_orders_pair_correctly": bool(risk[f] > risk[s]),
        }
        for s, f in zip(successes, best[1], strict=True)
    ]


def bootstrap_metrics(
    outcome: np.ndarray,
    progress_failure_score: np.ndarray,
    residual_risk: np.ndarray,
    risk: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distributions = {"raw_auroc": [], "progress_auroc": [], "residual_auroc": [], "partial_r": []}
    for _ in range(samples):
        index = rng.integers(0, len(outcome), len(outcome))
        sampled_outcome = outcome[index]
        if len(np.unique(sampled_outcome)) < 2:
            continue
        distributions["raw_auroc"].append(binary_metrics(sampled_outcome, risk[index])["auroc"])
        distributions["progress_auroc"].append(ranking_metrics(sampled_outcome, progress_failure_score[index])["auroc"])
        distributions["residual_auroc"].append(ranking_metrics(sampled_outcome, residual_risk[index])["auroc"])
        value = partial_correlation(risk[index], sampled_outcome.astype(float), -progress_failure_score[index])
        if np.isfinite(value):
            distributions["partial_r"].append(value)
    return {
        name: {
            "valid_samples": len(values),
            "ci95": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
        }
        for name, values in distributions.items()
    }


def analyze_model(
    name: str,
    checkpoint: int,
    probability: dict[tuple[str, int], float],
    data: Any,
    progress: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = data["checkpoint_step"].astype(int) == checkpoint
    episode_ids = data["episode_id"][selected]
    split = data["split"][selected]
    outcome = data["failure"][selected].astype(int)
    model_risk = np.asarray([probability[(str(eid), checkpoint)] for eid in episode_ids])
    model_progress = progress[selected]
    train = split == "train"
    test = split == "test"
    coefficients = fit_residualization(model_progress[train], model_risk[train])
    residual = apply_residualization(model_progress, model_risk, coefficients)
    test_outcome = outcome[test]
    test_progress = model_progress[test]
    test_risk = model_risk[test]
    test_residual = residual[test]
    progress_failure_score = -test_progress
    matches = closest_one_to_one_matches(
        episode_ids[test], test_outcome, test_progress, test_risk
    )
    return {
        "name": name,
        "checkpoint": checkpoint,
        "residualization": {
            "fit_split": "train",
            "model": "risk = intercept + slope * progress_proxy",
            "coefficients": coefficients,
            "uses_outcome_labels": False,
        },
        "test": {
            "episodes": int(test.sum()),
            "raw_risk": binary_metrics(test_outcome, test_risk),
            "progress_only_failure_score": ranking_metrics(test_outcome, progress_failure_score),
            "progress_residualized_risk": ranking_metrics(test_outcome, test_residual),
            "partial_pearson_r_risk_outcome_given_progress": partial_correlation(
                test_risk, test_outcome.astype(float), test_progress
            ),
            "closest_one_to_one_matches": matches,
            "matched_pairs": len(matches),
            "matched_pairs_correct": sum(row["risk_orders_pair_correctly"] for row in matches),
        },
        "episode_bootstrap": bootstrap_metrics(
            test_outcome,
            progress_failure_score,
            test_residual,
            test_risk,
            samples,
            seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("vision_report", type=Path)
    parser.add_argument("temporal_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    data = np.load(args.features)
    progress, proxy_metadata = build_progress_proxy(data)
    vision = load_predictions(args.vision_report, "checkpoint_vision_probability")
    temporal = load_predictions(args.temporal_report, "temporal_mlp_probability")
    output = {
        "analysis_type": "post-hoc mechanism diagnostic; does not alter frozen predictor results",
        "progress_proxy": proxy_metadata,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "models": {
            "vision_step80": analyze_model("vision", 80, vision, data, progress, args.samples, args.seed),
            "temporal_step120": analyze_model("temporal", 120, temporal, data, progress, args.samples, args.seed + 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("PROGRESS CONTROL", args.output)
    for key, result in output["models"].items():
        test = result["test"]
        print(
            key,
            "raw_auroc", test["raw_risk"]["auroc"],
            "progress_auroc", test["progress_only_failure_score"]["auroc"],
            "residual_auroc", test["progress_residualized_risk"]["auroc"],
            "partial_r", test["partial_pearson_r_risk_outcome_given_progress"],
            "matches", f"{test['matched_pairs_correct']}/{test['matched_pairs']}",
        )
        print("bootstrap", result["episode_bootstrap"])


if __name__ == "__main__":
    main()
