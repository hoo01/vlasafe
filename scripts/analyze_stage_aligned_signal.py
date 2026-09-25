"""Test frozen outcome scores after controlling privileged task-stage state."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from analyze_progress_control import ranking_metrics
except ModuleNotFoundError:
    from scripts.analyze_progress_control import ranking_metrics
from vlasafe.outcome_metrics import binary_metrics


TARGET_CANDIDATES = ("akita_black_bowl_1_main", "akita_black_bowl_2_main")
PLATE_BODY = "plate_1_main"
DRAWER_BODY = "wooden_cabinet_1_cabinet_top"
TOP_REGION_SITE = "wooden_cabinet_1_top_region"
FEATURE_NAMES = (
    "bowl_delta_x",
    "bowl_delta_y",
    "bowl_delta_z",
    "bowl_displacement_norm",
    "bowl_to_eef_distance",
    "bowl_to_plate_xy_distance",
    "bowl_to_plate_3d_distance",
    "top_drawer_displacement_norm",
    "gripper_qpos_mean",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _body_position(label: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(label["scene_body_poses"][name]["position"], dtype=np.float64)


def select_target(first_label: dict[str, Any]) -> str:
    region = np.asarray(first_label["scene_site_positions"][TOP_REGION_SITE], dtype=np.float64)
    return min(
        TARGET_CANDIDATES,
        key=lambda name: float(np.linalg.norm(_body_position(first_label, name)[:2] - region[:2])),
    )


def stage_features(rows: list[dict[str, Any]], checkpoint: int) -> tuple[np.ndarray, str]:
    if checkpoint <= 0 or checkpoint >= len(rows):
        raise IndexError(f"checkpoint {checkpoint} outside rollout of length {len(rows)}")
    initial = rows[0]["label_only"]
    # label_only is sampled after the action in its row. Row checkpoint-1 is
    # therefore aligned to the observation and camera frame at checkpoint.
    current = rows[checkpoint - 1]["label_only"]
    target = select_target(initial)
    bowl_initial = _body_position(initial, target)
    bowl = _body_position(current, target)
    plate = _body_position(current, PLATE_BODY)
    drawer_initial = _body_position(initial, DRAWER_BODY)
    drawer = _body_position(current, DRAWER_BODY)
    eef = np.asarray(rows[checkpoint]["proprioception"]["eef.pos"], dtype=np.float64)
    gripper = np.asarray(rows[checkpoint]["proprioception"]["gripper.qpos"], dtype=np.float64)
    delta = bowl - bowl_initial
    features = np.asarray(
        [
            *delta,
            np.linalg.norm(delta),
            np.linalg.norm(bowl - eef),
            np.linalg.norm((bowl - plate)[:2]),
            np.linalg.norm(bowl - plate),
            np.linalg.norm(drawer - drawer_initial),
            gripper.mean(),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("non-finite stage features")
    return features, target


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_linear(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def cross_fitted_controls(
    features: np.ndarray,
    labels: np.ndarray,
    risk: np.ndarray,
    groups: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    stage_score = np.empty(len(labels), dtype=np.float64)
    residual_risk = np.empty(len(labels), dtype=np.float64)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std < 1e-8] = 1.0
        train_x = (features[train] - mean) / std
        test_x = (features[test] - mean) / std
        outcome_coefficients = fit_linear(train_x, labels[train].astype(float), ridge)
        risk_coefficients = fit_linear(train_x, risk[train], ridge)
        stage_score[test] = predict_linear(test_x, outcome_coefficients)
        residual_risk[test] = risk[test] - predict_linear(test_x, risk_coefficients)
    return stage_score, residual_risk


def cluster_ranking_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    indexes = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    distributions = {name: {metric: [] for metric in ("auprc", "auroc")} for name in scores}
    for _ in range(samples):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        selected = np.concatenate([indexes[group] for group in sampled_groups])
        sampled_labels = labels[selected]
        if len(np.unique(sampled_labels)) < 2:
            continue
        for name, score in scores.items():
            metrics = ranking_metrics(sampled_labels, score[selected])
            for metric in distributions[name]:
                distributions[name][metric].append(metrics[metric])
    return {
        name: {
            metric: {
                "point": ranking_metrics(labels, scores[name])[metric],
                "valid_samples": len(values),
                "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
            }
            for metric, values in metrics.items()
        }
        for name, metrics in distributions.items()
    }


def optimal_pairs(
    episode_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    standardized_features: np.ndarray,
    risk: np.ndarray,
    stage_score: np.ndarray,
    residual_risk: np.ndarray,
) -> list[dict[str, Any]]:
    pairs = []
    for group in np.unique(groups):
        group_indexes = np.flatnonzero(groups == group)
        success = [index for index in group_indexes if labels[index] == 0]
        failure = [index for index in group_indexes if labels[index] == 1]
        if not success or not failure:
            continue
        smaller, larger = (success, failure) if len(success) <= len(failure) else (failure, success)
        best = None
        for assigned in itertools.permutations(larger, len(smaller)):
            cost = sum(
                np.linalg.norm(standardized_features[source] - standardized_features[target])
                for source, target in zip(smaller, assigned, strict=True)
            )
            if best is None or cost < best[0]:
                best = (float(cost), assigned)
        assert best is not None
        for source, target in zip(smaller, best[1], strict=True):
            success_index, failure_index = (
                (source, target) if labels[source] == 0 else (target, source)
            )
            pairs.append(
                {
                    "initial_state_id": int(group),
                    "success_episode": str(episode_ids[success_index]),
                    "failure_episode": str(episode_ids[failure_index]),
                    "stage_distance": float(
                        np.linalg.norm(
                            standardized_features[success_index]
                            - standardized_features[failure_index]
                        )
                    ),
                    "success_risk": float(risk[success_index]),
                    "failure_risk": float(risk[failure_index]),
                    "raw_risk_correct": bool(
                        risk[failure_index] > risk[success_index]
                    ),
                    "stage_only_correct": bool(
                        stage_score[failure_index] > stage_score[success_index]
                    ),
                    "residual_risk_correct": bool(
                        residual_risk[failure_index] > residual_risk[success_index]
                    ),
                }
            )
    return sorted(pairs, key=lambda row: (row["initial_state_id"], row["stage_distance"]))


def paired_accuracy(
    pairs: list[dict[str, Any]], key: str, samples: int, seed: int
) -> dict[str, Any]:
    if not pairs:
        return {"correct": 0, "pairs": 0, "accuracy": None, "ci95": None}
    groups = sorted({int(row["initial_state_id"]) for row in pairs})
    by_group = {
        group: [row for row in pairs if int(row["initial_state_id"]) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(groups, len(groups), replace=True)
        rows = [row for group in sampled for row in by_group[int(group)]]
        values.append(float(np.mean([row[key] for row in rows])))
    correct = sum(bool(row[key]) for row in pairs)
    return {
        "correct": correct,
        "pairs": len(pairs),
        "accuracy": correct / len(pairs),
        "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
    }


def event_summary(rows: list[dict[str, Any]]) -> tuple[str, dict[str, int | None]]:
    initial = rows[0]["label_only"]
    target = select_target(initial)
    start = _body_position(initial, target)
    events: dict[str, int | None] = {
        "displacement_gt_0.04": None,
        "height_increase_gt_0.04": None,
        "plate_xy_distance_lt_0.10": None,
    }
    for row in rows:
        label = row["label_only"]
        bowl = _body_position(label, target)
        plate = _body_position(label, PLATE_BODY)
        step = int(row["step_id"])
        if events["displacement_gt_0.04"] is None and np.linalg.norm(bowl - start) > 0.04:
            events["displacement_gt_0.04"] = step
        if events["height_increase_gt_0.04"] is None and bowl[2] - start[2] > 0.04:
            events["height_increase_gt_0.04"] = step
        if events["plate_xy_distance_lt_0.10"] is None and np.linalg.norm((bowl - plate)[:2]) < 0.10:
            events["plate_xy_distance_lt_0.10"] = step
    return target, events


def summarize_event_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for event in (
        "displacement_gt_0.04",
        "height_increase_gt_0.04",
        "plate_xy_distance_lt_0.10",
    ):
        event_rows = [row for row in rows if row["events"][event] is not None]
        result[event] = {
            "reached": len(event_rows),
            "total": len(rows),
            "reached_success": sum(row["failure"] == 0 for row in event_rows),
            "total_success": sum(row["failure"] == 0 for row in rows),
            "reached_failure": sum(row["failure"] == 1 for row in event_rows),
            "total_failure": sum(row["failure"] == 1 for row in rows),
            "median_step_if_reached": float(
                np.median([row["events"][event] for row in event_rows])
            ) if event_rows else None,
        }
    return result


def analyze_model(
    *,
    model: str,
    checkpoint: int,
    prediction_rows: list[dict[str, Any]],
    manifest_rows: dict[str, dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    ids, labels, groups, risk, feature_rows, targets, event_rows = [], [], [], [], [], [], []
    for prediction in prediction_rows:
        if prediction["model"] != model or int(prediction["checkpoint_step"]) != checkpoint:
            continue
        episode_id = str(prediction["episode_id"])
        manifest_row = manifest_rows[episode_id]
        rows = load_rows(Path(manifest_row["path"]))
        features, target = stage_features(rows, checkpoint)
        event_target, events = event_summary(rows)
        if target != event_target:
            raise RuntimeError("target selection changed within analysis")
        ids.append(episode_id)
        labels.append(int(prediction["failure"]))
        groups.append(int(prediction["initial_state_id"]))
        risk.append(float(prediction["probability"]))
        feature_rows.append(features)
        targets.append(target)
        event_rows.append({"failure": int(prediction["failure"]), "events": events})
    episode_ids = np.asarray(ids)
    labels_array = np.asarray(labels, dtype=int)
    groups_array = np.asarray(groups, dtype=int)
    risk_array = np.asarray(risk, dtype=np.float64)
    features_array = np.stack(feature_rows)
    mean = features_array.mean(axis=0)
    std = features_array.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (features_array - mean) / std

    ridge = 1e-3
    stage_score, residual_risk = cross_fitted_controls(
        features_array, labels_array, risk_array, groups_array, ridge
    )
    scores = {
        "raw_risk": risk_array,
        "stage_only": stage_score,
        "stage_residualized_risk": residual_risk,
    }
    sensitivity = {}
    for value in (1e-6, 1e-2, 1e-1, 1.0):
        _, candidate = cross_fitted_controls(
            features_array, labels_array, risk_array, groups_array, value
        )
        sensitivity[str(value)] = ranking_metrics(labels_array, candidate)
    pairs = optimal_pairs(
        episode_ids,
        labels_array,
        groups_array,
        standardized,
        risk_array,
        stage_score,
        residual_risk,
    )
    return {
        "model": model,
        "checkpoint_step": checkpoint,
        "episodes": len(episode_ids),
        "initial_state_groups": len(np.unique(groups_array)),
        "target_body_counts": {
            target: targets.count(target) for target in sorted(set(targets))
        },
        "stage_features": list(FEATURE_NAMES),
        "control_fit": {
            "method": "leave-one-initial-state-out linear ridge control",
            "primary_ridge": ridge,
            "uses_held_group_to_fit": False,
        },
        "cluster_bootstrap": cluster_ranking_bootstrap(
            labels_array, scores, groups_array, samples, seed
        ),
        "ridge_sensitivity_stage_residualized_risk": sensitivity,
        "same_initial_state_stage_matches": {
            "groups_with_mixed_outcomes": len({row["initial_state_id"] for row in pairs}),
            "pairs": len(pairs),
            "stage_distance": {
                "min": float(min(row["stage_distance"] for row in pairs)) if pairs else None,
                "median": float(np.median([row["stage_distance"] for row in pairs])) if pairs else None,
                "max": float(max(row["stage_distance"] for row in pairs)) if pairs else None,
            },
            "raw_risk": paired_accuracy(pairs, "raw_risk_correct", samples, seed + 10),
            "stage_only": paired_accuracy(pairs, "stage_only_correct", samples, seed + 11),
            "stage_residualized_risk": paired_accuracy(
                pairs, "residual_risk_correct", samples, seed + 12
            ),
            "details": pairs,
        },
        "event_reach_coverage": summarize_event_coverage(event_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prediction_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {
        str(row["episode_id"]): row
        for split in manifest["splits"].values()
        for row in split
    }
    predictions = json.loads(args.prediction_report.read_text(encoding="utf-8"))["predictions"]
    output = {
        "schema_version": "0.1.0",
        "analysis_type": "predeclared privileged task-stage control of frozen outcome scores",
        "information_boundary": "scene poses are analysis-only and never enter either predictor",
        "bootstrap_unit": "initial_state_id",
        "models": {
            "vision_step80": analyze_model(
                model="vision", checkpoint=80, prediction_rows=predictions,
                manifest_rows=manifest_rows, samples=args.samples, seed=args.seed,
            ),
            "temporal_step120": analyze_model(
                model="temporal", checkpoint=120, prediction_rows=predictions,
                manifest_rows=manifest_rows, samples=args.samples, seed=args.seed + 1,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("STAGE-ALIGNED SIGNAL", args.output)
    for name, result in output["models"].items():
        print(name, "episodes", result["episodes"], "targets", result["target_body_counts"])
        print("ranking", result["cluster_bootstrap"])
        print("matches", result["same_initial_state_stage_matches"] | {"details": "omitted"})
        print("events", result["event_reach_coverage"])


if __name__ == "__main__":
    main()
