"""Evaluate the frozen v0.4 pickup-stall models once on v0.5 confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from train_frozen_vision_baseline import predict, select_camera_features
    from train_temporal_mlp import _ensemble_predict
except ModuleNotFoundError:
    from scripts.train_frozen_vision_baseline import predict, select_camera_features
    from scripts.train_temporal_mlp import _ensemble_predict
from vlasafe.outcome_metrics import binary_metrics, fit_logistic_l2, predict_logistic
from vlasafe.pilot_metrics import cluster_bootstrap, matched_pair_accuracy


def temporal_features(data: Any, normalization: dict[str, Any]) -> np.ndarray:
    mask = data["mask"].astype(np.float32)
    channels = np.concatenate(
        [data["state"], data["action"], data["timing"]], axis=-1
    ).astype(np.float32)
    mean = np.asarray(normalization["channel_mean"], dtype=np.float32)
    std = np.asarray(normalization["channel_std"], dtype=np.float32)
    normalized = (channels - mean) / std
    normalized *= mask[..., None]
    features = np.concatenate([normalized.reshape(len(normalized), -1), mask], axis=1)
    expected = int(normalization["feature_dim"])
    if features.shape[1] != expected:
        raise ValueError(f"temporal feature dimension {features.shape[1]} != {expected}")
    return features.astype(np.float32)


def stage_probabilities(
    pilot: Any,
    confirmation: Any,
    chosen_l2: float,
) -> np.ndarray:
    pilot_raw = np.column_stack(
        [pilot["first_approach_step"], pilot["checkpoint_target_movement_m"]]
    ).astype(float)
    confirmation_raw = np.column_stack(
        [
            confirmation["first_approach_step"],
            confirmation["checkpoint_target_movement_m"],
        ]
    ).astype(float)
    train = pilot["split"] == "train"
    mean = pilot_raw[train].mean(axis=0)
    std = pilot_raw[train].std(axis=0)
    std[std < 1e-8] = 1.0
    weights, bias = fit_logistic_l2(
        (pilot_raw[train] - mean) / std,
        pilot["pickup_stall"][train].astype(np.int64),
        float(chosen_l2),
    )
    return predict_logistic((confirmation_raw - mean) / std, weights, bias)


def evaluate_subset(
    target: np.ndarray,
    group: np.ndarray,
    probabilities: dict[str, np.ndarray],
    selected: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected_probabilities = {name: value[selected] for name, value in probabilities.items()}
    return {
        "samples": int(selected.sum()),
        "positives": int(target[selected].sum()),
        "negatives": int(selected.sum() - target[selected].sum()),
        "groups": int(len(np.unique(group[selected]))),
        "metrics": {
            name: {
                **binary_metrics(target[selected], probability),
                "same_state_pairs": matched_pair_accuracy(
                    target[selected], probability, group[selected]
                ),
            }
            for name, probability in selected_probabilities.items()
        },
        "cluster_bootstrap": cluster_bootstrap(
            target[selected],
            selected_probabilities,
            group[selected],
            bootstrap_samples,
            bootstrap_seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_temporal", type=Path)
    parser.add_argument("confirmation_vision", type=Path)
    parser.add_argument("--pilot-temporal", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    args = parser.parse_args()

    temporal = np.load(args.confirmation_temporal)
    vision = np.load(args.confirmation_vision)
    pilot = np.load(args.pilot_temporal)
    pilot_report = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.confirmation_manifest.read_text(encoding="utf-8"))
    for key in ("episode_id", "pickup_stall", "initial_state_id"):
        if not np.array_equal(temporal[key], vision[key]):
            raise ValueError(f"temporal/vision alignment mismatch: {key}")

    device = torch.device(args.device)
    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    target = temporal["pickup_stall"].astype(np.int64)
    group = temporal["initial_state_id"].astype(np.int64)
    pilot_train = pilot["split"] == "train"
    prevalence = float(pilot["pickup_stall"][pilot_train].mean())
    stage = stage_probabilities(
        pilot,
        temporal,
        pilot_report["stage_baseline"]["chosen_l2"],
    )
    temporal_x = temporal_features(temporal, bundle["temporal_normalization"])
    temporal_probability = _ensemble_predict(
        bundle["temporal_states"], temporal_x, device
    )
    vision_x, cameras = select_camera_features(
        vision["features"].astype(np.float32), vision["camera"], "dual"
    )
    vision_metadata = bundle["vision_metadata"]
    vision_x = (
        vision_x - np.asarray(vision_metadata["normalization_mean"], dtype=np.float32)
    ) / np.asarray(vision_metadata["normalization_std"], dtype=np.float32)
    vision_probability = predict(bundle["vision_state"], vision_x.astype(np.float32), device)
    probabilities = {
        "prevalence": np.full(len(target), prevalence),
        "stage_only": stage,
        "temporal_mlp": temporal_probability,
        "frozen_vision": vision_probability,
    }

    mixed_groups = {
        value
        for value in np.unique(group)
        if np.any(target[group == value] == 1) and np.any(target[group == value] == 0)
    }
    all_selected = np.ones(len(target), dtype=bool)
    mixed_selected = np.asarray([value in mixed_groups for value in group], dtype=bool)
    full = evaluate_subset(
        target, group, probabilities, all_selected,
        args.bootstrap_samples, args.bootstrap_seed,
    )
    matched = evaluate_subset(
        target, group, probabilities, mixed_selected,
        args.bootstrap_samples, args.bootstrap_seed + 1,
    )
    primary = full["cluster_bootstrap"]["delta_vs_stage_only"]["temporal_mlp"]
    ranking_gate = all(primary[name]["ci95"][0] > 0 for name in ("auprc", "auroc"))
    support_gate = bool(manifest["formal_gate"]["passed"])
    report = {
        "schema_version": "0.1.0",
        "role": "one_shot_frozen_phase2_confirmation",
        "confirmation_temporal": args.confirmation_temporal.as_posix(),
        "confirmation_vision": args.confirmation_vision.as_posix(),
        "pilot_temporal": args.pilot_temporal.as_posix(),
        "pilot_report": args.pilot_report.as_posix(),
        "frozen_model": args.model.as_posix(),
        "confirmation_manifest": args.confirmation_manifest.as_posix(),
        "device": str(device),
        "cameras": cameras,
        "support_gate_passed": support_gate,
        "primary_ranking_gate_passed": ranking_gate,
        "formal_claim_gate_passed": support_gate and ranking_gate,
        "full_confirmation": full,
        "mixed_initial_state_subset": matched,
        "predictions": [
            {
                "episode_id": str(temporal["episode_id"][index]),
                "initial_state_id": int(group[index]),
                "pickup_stall": int(target[index]),
                **{name: float(value[index]) for name, value in probabilities.items()},
            }
            for index in range(len(target))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FROZEN PICKUP STALL CONFIRMATION", args.output)
    print("support/ranking/formal", support_gate, ranking_gate, support_gate and ranking_gate)
    for subset_name, result in (("full", full), ("mixed", matched)):
        print(subset_name, result["samples"], result["positives"], result["groups"])
        for name, metrics in result["metrics"].items():
            print(subset_name, name, {key: value for key, value in metrics.items() if key != "same_state_pairs"}, metrics["same_state_pairs"])
        print(subset_name, "delta_vs_stage", result["cluster_bootstrap"]["delta_vs_stage_only"])


if __name__ == "__main__":
    main()
