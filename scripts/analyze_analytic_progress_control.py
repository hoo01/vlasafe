"""Control frozen outcome risk with deploy-side kinematic progress summaries.

All transforms are fitted on the v0.1 train split. The independent v0.2
confirmation cohort is used only once for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from analyze_progress_control import load_predictions, ranking_metrics
except ModuleNotFoundError:
    from scripts.analyze_progress_control import load_predictions, ranking_metrics
from vlasafe.outcome_metrics import binary_metrics


FEATURE_NAMES = (
    "eef_delta_x",
    "eef_delta_y",
    "eef_delta_z",
    "eef_path_length",
    "eef_vertical_range",
    "gripper_qpos_mean",
    "gripper_qpos_change",
)


def _entries(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    try:
        return list(manifest["splits"][split])
    except KeyError as exc:
        raise KeyError(f"manifest has no split {split!r}") from exc


def _episode_path(entry: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(entry["path"])
    if path.is_absolute():
        return path
    # Repository manifests store paths relative to the repository root.
    candidates = [Path.cwd() / path, manifest_path.parent.parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_rows(entry: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    path = _episode_path(entry, manifest_path) / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing rollout sidecar: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty rollout sidecar: {path}")
    return rows


def analytic_features(rows: list[dict[str, Any]], checkpoint: int) -> np.ndarray:
    """Summarize causal deploy-side kinematics through ``checkpoint``."""
    if checkpoint < 0 or checkpoint >= len(rows):
        raise IndexError(f"checkpoint {checkpoint} outside rollout of length {len(rows)}")
    selected = rows[: checkpoint + 1]
    eef = np.asarray(
        [row["proprioception"]["eef.pos"] for row in selected], dtype=np.float64
    )
    gripper = np.asarray(
        [row["proprioception"]["gripper.qpos"] for row in selected], dtype=np.float64
    )
    if eef.shape[1:] != (3,) or gripper.shape[1:] != (2,):
        raise ValueError("unexpected eef.pos or gripper.qpos shape")
    if not np.isfinite(eef).all() or not np.isfinite(gripper).all():
        raise ValueError("non-finite analytic progress input")
    delta = eef[-1] - eef[0]
    path_length = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
    vertical_range = float(np.ptp(eef[:, 2]))
    gripper_mean = float(gripper[-1].mean())
    gripper_change = float(gripper[-1].mean() - gripper[0].mean())
    return np.asarray(
        [*delta, path_length, vertical_range, gripper_mean, gripper_change],
        dtype=np.float64,
    )


def build_cohort(
    manifest_path: Path, split: str, checkpoint: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _entries(manifest, split)
    ids, labels, features = [], [], []
    for entry in entries:
        ids.append(str(entry["episode_id"]))
        labels.append(0 if bool(entry["success"]) else 1)
        features.append(analytic_features(load_rows(entry, manifest_path), checkpoint))
    return np.asarray(ids), np.asarray(labels, dtype=int), np.stack(features)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_linear(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def partial_correlation(risk: np.ndarray, outcome: np.ndarray, controls: np.ndarray) -> float:
    risk_residual = risk - predict_linear(controls, fit_linear(controls, risk))
    outcome_residual = outcome - predict_linear(controls, fit_linear(controls, outcome))
    if np.std(risk_residual) < 1e-12 or np.std(outcome_residual) < 1e-12:
        return float("nan")
    return float(np.corrcoef(risk_residual, outcome_residual)[0, 1])


def bootstrap(
    labels: np.ndarray,
    raw_risk: np.ndarray,
    analytic_score: np.ndarray,
    residual_risk: np.ndarray,
    controls: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in ("raw_auroc", "analytic_auroc", "residual_auroc", "partial_r")}
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        distributions["raw_auroc"].append(binary_metrics(sampled, raw_risk[index])["auroc"])
        distributions["analytic_auroc"].append(ranking_metrics(sampled, analytic_score[index])["auroc"])
        distributions["residual_auroc"].append(ranking_metrics(sampled, residual_risk[index])["auroc"])
        value = partial_correlation(raw_risk[index], sampled.astype(float), controls[index])
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
    *,
    name: str,
    checkpoint: int,
    reference_manifest: Path,
    confirmation_manifest: Path,
    reference_predictions: dict[tuple[str, int], float],
    confirmation_predictions: dict[tuple[str, int], float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    train_ids, train_labels, train_raw = build_cohort(reference_manifest, "train", checkpoint)
    confirmation_ids, labels, confirmation_raw = build_cohort(
        confirmation_manifest, "confirmation", checkpoint
    )
    mean, std = fit_standardizer(train_raw)
    train_x = (train_raw - mean) / std
    confirmation_x = (confirmation_raw - mean) / std
    train_risk = np.asarray([reference_predictions[(str(eid), checkpoint)] for eid in train_ids])
    risk = np.asarray([confirmation_predictions[(str(eid), checkpoint)] for eid in confirmation_ids])

    outcome_coefficients = fit_linear(train_x, train_labels.astype(float))
    risk_coefficients = fit_linear(train_x, train_risk)
    analytic_score = predict_linear(confirmation_x, outcome_coefficients)
    expected_risk = predict_linear(confirmation_x, risk_coefficients)
    residual_risk = risk - expected_risk
    outside = (confirmation_raw < train_raw.min(axis=0)) | (confirmation_raw > train_raw.max(axis=0))

    return {
        "name": name,
        "checkpoint": checkpoint,
        "controls": list(FEATURE_NAMES),
        "reference_fit": {
            "split": "v0.1 train",
            "episodes": int(len(train_ids)),
            "standardization_mean": mean.tolist(),
            "standardization_std": std.tolist(),
            "analytic_outcome_coefficients": outcome_coefficients.tolist(),
            "risk_on_controls_coefficients": risk_coefficients.tolist(),
        },
        "confirmation": {
            "episodes": int(len(labels)),
            "failures": int(labels.sum()),
            "raw_risk": binary_metrics(labels, risk),
            "analytic_progress_baseline": ranking_metrics(labels, analytic_score),
            "progress_residualized_risk": ranking_metrics(labels, residual_risk),
            "partial_pearson_r_risk_outcome_given_controls": partial_correlation(
                risk, labels.astype(float), confirmation_x
            ),
            "support": {
                "episodes_with_any_feature_outside_reference_train_range": int(outside.any(axis=1).sum()),
                "outside_reference_train_range_by_feature": {
                    feature: int(outside[:, index].sum())
                    for index, feature in enumerate(FEATURE_NAMES)
                },
            },
        },
        "episode_bootstrap": bootstrap(
            labels, risk, analytic_score, residual_risk, confirmation_x, samples, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("confirmation_manifest", type=Path)
    parser.add_argument("reference_vision_report", type=Path)
    parser.add_argument("reference_temporal_report", type=Path)
    parser.add_argument("confirmation_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    reference_vision = load_predictions(args.reference_vision_report, "checkpoint_vision_probability")
    reference_temporal = load_predictions(args.reference_temporal_report, "temporal_mlp_probability")
    confirmation_vision = load_predictions(args.confirmation_report, "checkpoint_vision_probability")
    confirmation_temporal = load_predictions(args.confirmation_report, "temporal_mlp_probability")
    output = {
        "schema_version": "0.1.0",
        "analysis_type": "frozen v0.1 deploy-side analytic progress controls applied once to v0.2 confirmation",
        "information_boundary": "uses only causal proprioception fields available to deployment; controls are analysis-only",
        "models": {
            "vision_step80": analyze_model(
                name="vision", checkpoint=80,
                reference_manifest=args.reference_manifest,
                confirmation_manifest=args.confirmation_manifest,
                reference_predictions=reference_vision,
                confirmation_predictions=confirmation_vision,
                samples=args.samples, seed=args.seed,
            ),
            "temporal_step120": analyze_model(
                name="temporal", checkpoint=120,
                reference_manifest=args.reference_manifest,
                confirmation_manifest=args.confirmation_manifest,
                reference_predictions=reference_temporal,
                confirmation_predictions=confirmation_temporal,
                samples=args.samples, seed=args.seed + 1,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("ANALYTIC PROGRESS CONTROL", args.output)
    for key, result in output["models"].items():
        values = result["confirmation"]
        print(
            key,
            "raw_auroc", values["raw_risk"]["auroc"],
            "analytic_auroc", values["analytic_progress_baseline"]["auroc"],
            "residual_auroc", values["progress_residualized_risk"]["auroc"],
            "partial_r", values["partial_pearson_r_risk_outcome_given_controls"],
            "support_ood", values["support"]["episodes_with_any_feature_outside_reference_train_range"],
        )
        print("bootstrap", result["episode_bootstrap"])


if __name__ == "__main__":
    main()
