"""Apply the frozen v0.1 progress-control audit to the v0.2 confirmation cohort."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from analyze_progress_control import (
        apply_residualization,
        bootstrap_metrics,
        fit_residualization,
        load_predictions,
        partial_correlation,
        ranking_metrics,
    )
except ModuleNotFoundError:  # Imported as scripts.* by the unit tests.
    from scripts.analyze_progress_control import (
        apply_residualization,
        bootstrap_metrics,
        fit_residualization,
        load_predictions,
        partial_correlation,
        ranking_metrics,
    )
from vlasafe.outcome_metrics import binary_metrics


def fit_progress_transform(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalization and an outcome-label-free time axis using reference train only."""
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
        raise ValueError("reference train progress axis has zero norm")
    axis /= norm
    return mean, std, axis, normalized @ axis


def apply_progress_transform(data: Any, mean: np.ndarray, std: np.ndarray, axis: np.ndarray) -> np.ndarray:
    raw = data["features"].astype(np.float64).reshape(len(data["features"]), -1)
    if raw.shape[1] != len(mean):
        raise ValueError(f"feature dimension mismatch: confirmation={raw.shape[1]}, reference={len(mean)}")
    return ((raw - mean) / std) @ axis


def optimal_matches(
    episode_ids: np.ndarray,
    outcome: np.ndarray,
    progress: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, Any]]:
    """Minimum-total-gap one-to-one matching for either class balance direction."""
    success = np.flatnonzero(outcome == 0).tolist()
    failure = np.flatnonzero(outcome == 1).tolist()
    if not success or not failure:
        return []
    smaller, larger = (success, failure) if len(success) <= len(failure) else (failure, success)

    @lru_cache(maxsize=None)
    def solve(position: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if position == len(smaller):
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        source = smaller[position]
        for larger_position, target in enumerate(larger):
            if used_mask & (1 << larger_position):
                continue
            remaining_cost, assignment = solve(
                position + 1, used_mask | (1 << larger_position)
            )
            candidate = (
                abs(float(progress[source] - progress[target])) + remaining_cost,
                (target,) + assignment,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise RuntimeError("matching failed")
        return best

    _, assigned = solve(0, 0)
    pairs = []
    for source, target in zip(smaller, assigned, strict=True):
        success_index, failure_index = (
            (source, target) if outcome[source] == 0 else (target, source)
        )
        pairs.append(
            {
                "success_episode": str(episode_ids[success_index]),
                "failure_episode": str(episode_ids[failure_index]),
                "absolute_progress_gap": float(
                    abs(progress[success_index] - progress[failure_index])
                ),
                "success_risk": float(risk[success_index]),
                "failure_risk": float(risk[failure_index]),
                "risk_orders_pair_correctly": bool(
                    risk[failure_index] > risk[success_index]
                ),
            }
        )
    return sorted(pairs, key=lambda row: row["absolute_progress_gap"])


def analyze(
    *,
    name: str,
    checkpoint: int,
    reference_data: Any,
    confirmation_data: Any,
    reference_progress: np.ndarray,
    confirmation_progress: np.ndarray,
    reference_probability: dict[tuple[str, int], float],
    confirmation_probability: dict[tuple[str, int], float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    reference_selected = reference_data["checkpoint_step"].astype(int) == checkpoint
    reference_train = reference_selected & (reference_data["split"] == "train")
    reference_ids = reference_data["episode_id"][reference_train]
    train_risk = np.asarray(
        [reference_probability[(str(eid), checkpoint)] for eid in reference_ids]
    )
    train_progress = reference_progress[reference_train]
    coefficients = fit_residualization(train_progress, train_risk)
    train_sd = float(np.std(train_progress, ddof=1))
    if train_sd <= 0:
        raise ValueError("reference train progress has zero standard deviation")

    selected = confirmation_data["checkpoint_step"].astype(int) == checkpoint
    episode_ids = confirmation_data["episode_id"][selected]
    outcome = confirmation_data["failure"][selected].astype(int)
    progress = confirmation_progress[selected]
    risk = np.asarray(
        [confirmation_probability[(str(eid), checkpoint)] for eid in episode_ids]
    )
    residual = apply_residualization(progress, risk, coefficients)
    progress_failure_score = -progress
    matches = optimal_matches(episode_ids, outcome, progress, risk)
    for row in matches:
        row["progress_gap_reference_train_sd"] = (
            row["absolute_progress_gap"] / train_sd
        )
    overlap = {
        str(caliper): {
            "pairs": sum(
                row["progress_gap_reference_train_sd"] <= caliper for row in matches
            ),
            "correct": sum(
                row["progress_gap_reference_train_sd"] <= caliper
                and row["risk_orders_pair_correctly"]
                for row in matches
            ),
        }
        for caliper in (0.25, 0.5, 1.0)
    }
    standardized = (progress - float(np.mean(train_progress))) / train_sd
    return {
        "name": name,
        "checkpoint": checkpoint,
        "reference_fit": {
            "split": "v0.1 train",
            "risk_on_progress_coefficients": coefficients,
            "progress_mean": float(np.mean(train_progress)),
            "progress_sd": train_sd,
            "progress_min": float(np.min(train_progress)),
            "progress_max": float(np.max(train_progress)),
        },
        "confirmation": {
            "episodes": int(selected.sum()),
            "failures": int(outcome.sum()),
            "raw_risk": binary_metrics(outcome, risk),
            "progress_only_failure_score": ranking_metrics(
                outcome, progress_failure_score
            ),
            "progress_residualized_risk": ranking_metrics(outcome, residual),
            "partial_pearson_r_risk_outcome_given_progress": partial_correlation(
                risk, outcome.astype(float), progress
            ),
            "progress_standardized_to_reference_train": {
                "min": float(np.min(standardized)),
                "max": float(np.max(standardized)),
                "outside_reference_train_range": int(
                    np.sum(
                        (progress < np.min(train_progress))
                        | (progress > np.max(train_progress))
                    )
                ),
            },
            "optimal_one_to_one_matches": matches,
            "matched_pairs": len(matches),
            "matched_pairs_correct": sum(
                row["risk_orders_pair_correctly"] for row in matches
            ),
            "matched_overlap_by_caliper_reference_train_sd": overlap,
        },
        "episode_bootstrap": bootstrap_metrics(
            outcome,
            progress_failure_score,
            residual,
            risk,
            samples,
            seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_features", type=Path)
    parser.add_argument("confirmation_features", type=Path)
    parser.add_argument("reference_vision_report", type=Path)
    parser.add_argument("reference_temporal_report", type=Path)
    parser.add_argument("confirmation_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    reference = np.load(args.reference_features)
    confirmation = np.load(args.confirmation_features)
    if [str(value) for value in reference["camera"]] != [
        str(value) for value in confirmation["camera"]
    ]:
        raise ValueError("reference and confirmation camera order differs")
    mean, std, axis, reference_progress = fit_progress_transform(reference)
    confirmation_progress = apply_progress_transform(confirmation, mean, std, axis)
    reference_vision = load_predictions(
        args.reference_vision_report, "checkpoint_vision_probability"
    )
    reference_temporal = load_predictions(
        args.reference_temporal_report, "temporal_mlp_probability"
    )
    confirmation_vision = load_predictions(
        args.confirmation_report, "checkpoint_vision_probability"
    )
    confirmation_temporal = load_predictions(
        args.confirmation_report, "temporal_mlp_probability"
    )
    output = {
        "schema_version": "0.1.0",
        "analysis_type": "frozen v0.1 progress control applied once to v0.2 confirmation",
        "progress_proxy": {
            "fit_split": "v0.1 train",
            "definition": "projection onto mean(step120)-mean(step0) frozen-feature direction",
            "uses_outcome_labels": False,
            "uses_confirmation_to_fit": False,
            "feature_dim": int(len(mean)),
        },
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "models": {
            "vision_step80": analyze(
                name="vision",
                checkpoint=80,
                reference_data=reference,
                confirmation_data=confirmation,
                reference_progress=reference_progress,
                confirmation_progress=confirmation_progress,
                reference_probability=reference_vision,
                confirmation_probability=confirmation_vision,
                samples=args.samples,
                seed=args.seed,
            ),
            "temporal_step120": analyze(
                name="temporal",
                checkpoint=120,
                reference_data=reference,
                confirmation_data=confirmation,
                reference_progress=reference_progress,
                confirmation_progress=confirmation_progress,
                reference_probability=reference_temporal,
                confirmation_probability=confirmation_temporal,
                samples=args.samples,
                seed=args.seed + 1,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("CONFIRMATION PROGRESS CONTROL", args.output)
    for key, result in output["models"].items():
        values = result["confirmation"]
        print(
            key,
            "raw_auroc", values["raw_risk"]["auroc"],
            "progress_auroc", values["progress_only_failure_score"]["auroc"],
            "residual_auroc", values["progress_residualized_risk"]["auroc"],
            "partial_r", values["partial_pearson_r_risk_outcome_given_progress"],
            "matches", f"{values['matched_pairs_correct']}/{values['matched_pairs']}",
            "caliper_0.5", values["matched_overlap_by_caliper_reference_train_sd"]["0.5"],
        )
        print("bootstrap", result["episode_bootstrap"])


if __name__ == "__main__":
    main()
