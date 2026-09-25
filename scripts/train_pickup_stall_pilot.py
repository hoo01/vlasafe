"""Train and evaluate frozen Phase-2 pickup-stall pilot baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from train_frozen_vision_baseline import L2_CANDIDATES, fit_probe, predict, select_camera_features
    from train_temporal_mlp import TRAINING_SEEDS, _ensemble_predict, _fit_one
except ModuleNotFoundError:
    from scripts.train_frozen_vision_baseline import L2_CANDIDATES, fit_probe, predict, select_camera_features
    from scripts.train_temporal_mlp import TRAINING_SEEDS, _ensemble_predict, _fit_one
from vlasafe.outcome_dataset import normalize_and_flatten_temporal
from vlasafe.outcome_metrics import binary_metrics, fit_logistic_l2, predict_logistic
from vlasafe.pilot_metrics import cluster_bootstrap, matched_pair_accuracy


def fit_approach_baseline(
    step: np.ndarray, target: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = float(step[train].mean())
    std = float(step[train].std()) or 1.0
    features = ((step - mean) / std)[:, None]
    candidates = []
    for l2 in L2_CANDIDATES:
        weights, bias = fit_logistic_l2(features[train], target[train], l2)
        probability = predict_logistic(features, weights, bias)
        metrics = binary_metrics(target[validation], probability[validation])
        candidates.append((metrics["auprc"], -metrics["brier"], l2, weights, bias, probability, metrics))
    chosen = max(candidates, key=lambda row: (row[0], row[1]))
    return chosen[5], {
        "feature": "privileged first_approach_step; analysis baseline only",
        "train_mean": mean,
        "train_std": std,
        "chosen_l2": chosen[2],
        "validation_metrics": chosen[6],
    }


def fit_vision(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, torch.Tensor]]:
    mean = features[train].mean(axis=0)
    std = features[train].std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = ((features - mean) / std).astype(np.float32)
    train_x = torch.from_numpy(normalized[train]).to(device)
    train_y = torch.from_numpy(target[train].astype(np.float32)).to(device)
    validation_x = torch.from_numpy(normalized[validation]).to(device)
    validation_y = torch.from_numpy(target[validation].astype(np.float32)).to(device)
    candidates = []
    for l2 in L2_CANDIDATES:
        state, validation_bce = fit_probe(train_x, train_y, validation_x, validation_y, l2)
        probability = predict(state, normalized, device)
        metrics = binary_metrics(target[validation], probability[validation])
        candidates.append((metrics["auprc"], -metrics["brier"], l2, state, probability, validation_bce, metrics))
    chosen = max(candidates, key=lambda row: (row[0], row[1]))
    return chosen[4], {
        "chosen_l2": chosen[2],
        "selection": "maximum validation AUPRC; tie-break minimum Brier",
        "validation_bce": chosen[5],
        "validation_metrics": chosen[6],
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }, chosen[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("temporal_dataset", type=Path)
    parser.add_argument("vision_features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    args = parser.parse_args()
    temporal = np.load(args.temporal_dataset)
    vision = np.load(args.vision_features)
    for key in ("episode_id", "split", "pickup_stall", "initial_state_id"):
        if not np.array_equal(temporal[key], vision[key]):
            raise ValueError(f"temporal/vision alignment mismatch: {key}")
    target = temporal["pickup_stall"].astype(np.int64)
    split = temporal["split"]
    train, validation, test = split == "train", split == "validation", split == "test"
    device = torch.device(args.device)
    prevalence = float(target[train].mean())
    prevalence_probability = np.full(len(target), prevalence)
    approach_probability, approach_metadata = fit_approach_baseline(
        temporal["first_approach_step"].astype(float), target, train, validation
    )

    temporal_features, temporal_normalization = normalize_and_flatten_temporal(
        temporal, include_checkpoint_progress=False
    )
    temporal_states = []
    temporal_runs = []
    for seed in TRAINING_SEEDS:
        state, epoch, loss = _fit_one(temporal_features, target, train, validation, seed, device)
        temporal_states.append(state)
        temporal_runs.append({"seed": seed, "best_epoch": epoch, "validation_bce": loss})
    temporal_probability = _ensemble_predict(temporal_states, temporal_features, device)

    vision_features, cameras = select_camera_features(
        vision["features"].astype(np.float32), vision["camera"], "dual"
    )
    vision_probability, vision_metadata, vision_state = fit_vision(
        vision_features, target, train, validation, device
    )
    probabilities = {
        "prevalence": prevalence_probability,
        "approach_time": approach_probability,
        "temporal_mlp": temporal_probability,
        "frozen_vision": vision_probability,
    }
    results = {}
    for split_name, selected in (("train", train), ("validation", validation), ("test", test)):
        results[split_name] = {
            name: {
                **binary_metrics(target[selected], probability[selected]),
                "same_state_pairs": matched_pair_accuracy(
                    target[selected], probability[selected], temporal["initial_state_id"][selected]
                ),
            }
            for name, probability in probabilities.items()
        }
    test_probabilities = {name: value[test] for name, value in probabilities.items()}
    bootstrap = cluster_bootstrap(
        target[test], test_probabilities, temporal["initial_state_id"][test],
        args.bootstrap_samples, args.bootstrap_seed,
    )
    report = {
        "schema_version": "0.1.0",
        "task": "progress-aligned prediction of pickup-stall confirmation within 20 steps",
        "temporal_dataset": args.temporal_dataset.as_posix(),
        "vision_features": args.vision_features.as_posix(),
        "target": "pickup_stall",
        "absolute_checkpoint_feature": False,
        "approach_baseline": approach_metadata,
        "temporal": {"model": "MLP 128-64, 5-seed ensemble", "normalization": temporal_normalization, "runs": temporal_runs},
        "vision": {"model": "frozen ResNet-50 dual-camera linear probe", "cameras": cameras, **vision_metadata},
        "results": results,
        "test_cluster_bootstrap": bootstrap,
        "predictions": [
            {
                "episode_id": str(temporal["episode_id"][index]),
                "initial_state_id": int(temporal["initial_state_id"][index]),
                "split": str(split[index]),
                "pickup_stall": int(target[index]),
                **{name: float(probability[index]) for name, probability in probabilities.items()},
            }
            for index in range(len(target))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {"temporal_states": temporal_states, "temporal_normalization": temporal_normalization, "vision_state": vision_state, "vision_metadata": vision_metadata},
        args.output.with_suffix(".pt"),
    )
    print("PICKUP STALL PILOT RESULT", args.output)
    print("device", device, "test groups", len(np.unique(temporal["initial_state_id"][test])))
    for name, values in results["test"].items():
        print(name, {key: value for key, value in values.items() if key != "same_state_pairs"}, values["same_state_pairs"])
    print("delta vs approach", bootstrap["delta_vs_approach_time"])


if __name__ == "__main__":
    main()
