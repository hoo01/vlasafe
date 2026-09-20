"""Apply frozen v0.1 temporal and vision predictors to an independent cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_frozen_vision_baseline import predict as vision_predict
from train_frozen_vision_baseline import select_camera_features
from train_temporal_mlp import _ensemble_predict
from vlasafe.outcome_metrics import binary_metrics


def _temporal_features(data: Any, normalization: dict[str, Any]) -> np.ndarray:
    mask = data["mask"].astype(np.float32)
    channels = np.concatenate(
        [data["state"], data["action"], data["timing"]], axis=-1
    ).astype(np.float32)
    mean = np.asarray(normalization["channel_mean"], dtype=np.float32)
    std = np.asarray(normalization["channel_std"], dtype=np.float32)
    normalized = (channels - mean) / std
    normalized *= mask[..., None]
    parts = [normalized.reshape(len(normalized), -1), mask]
    denominator = normalization.get("progress_denominator")
    if denominator is not None:
        parts.append((data["checkpoint_step"].astype(np.float32) / denominator)[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _bootstrap(labels: np.ndarray, probability: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in ("auprc", "auroc", "brier", "ece")}
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        metrics = binary_metrics(labels[index], probability[index])
        for name in distributions:
            distributions[name].append(metrics[name])
    return {
        name: {
            "point": binary_metrics(labels, probability)[name],
            "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
        }
        for name, values in distributions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("temporal_dataset", type=Path)
    parser.add_argument("vision_features", type=Path)
    parser.add_argument("--temporal-model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    device = torch.device(args.device)
    temporal_data = np.load(args.temporal_dataset)
    vision_data = np.load(args.vision_features)
    temporal_artifact = torch.load(args.temporal_model, map_location="cpu", weights_only=True)
    vision_artifact = torch.load(args.vision_model, map_location="cpu", weights_only=True)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    prevalence = float(reference["predictions"][0]["prevalence_probability"])

    temporal_x = _temporal_features(temporal_data, temporal_artifact["normalization"])
    temporal_probability = _ensemble_predict(
        temporal_artifact["model_states"], temporal_x, device
    )
    vision_x, cameras = select_camera_features(
        vision_data["features"].astype(np.float32), vision_data["camera"], "dual"
    )
    vision_probability = np.empty(len(vision_x), dtype=np.float64)
    for checkpoint_text, state in vision_artifact["models"].items():
        checkpoint = int(checkpoint_text)
        selected = vision_data["checkpoint_step"].astype(int) == checkpoint
        metadata = vision_artifact["metadata"][checkpoint_text]
        mean = np.asarray(metadata["mean"], dtype=np.float32)
        std = np.asarray(metadata["std"], dtype=np.float32)
        normalized = ((vision_x[selected] - mean) / std).astype(np.float32)
        vision_probability[selected] = vision_predict(state, normalized, device)

    temporal_key = {
        (str(eid), int(step)): i
        for i, (eid, step) in enumerate(
            zip(temporal_data["episode_id"], temporal_data["checkpoint_step"], strict=True)
        )
    }
    predictions = []
    results = {}
    for checkpoint in sorted(np.unique(vision_data["checkpoint_step"]).astype(int)):
        checkpoint = int(checkpoint)
        vision_selected = vision_data["checkpoint_step"].astype(int) == checkpoint
        labels = vision_data["failure"][vision_selected].astype(int)
        episode_ids = vision_data["episode_id"][vision_selected]
        temporal_indexes = np.asarray([temporal_key[(str(eid), checkpoint)] for eid in episode_ids])
        temporal_p = temporal_probability[temporal_indexes]
        vision_p = vision_probability[vision_selected]
        prevalence_p = np.full(len(labels), prevalence)
        results[str(checkpoint)] = {
            "episodes": len(labels),
            "failure_rate": float(labels.mean()),
            "prevalence": _bootstrap(labels, prevalence_p, args.bootstrap_samples, 1000 + checkpoint),
            "temporal_mlp": _bootstrap(labels, temporal_p, args.bootstrap_samples, 2000 + checkpoint),
            "frozen_vision": _bootstrap(labels, vision_p, args.bootstrap_samples, 3000 + checkpoint),
        }
        for eid, label, tp, vp in zip(episode_ids, labels, temporal_p, vision_p, strict=True):
            predictions.append({
                "episode_id": str(eid),
                "checkpoint_step": checkpoint,
                "failure": int(label),
                "prevalence_probability": prevalence,
                "temporal_mlp_probability": float(tp),
                "checkpoint_vision_probability": float(vp),
            })
    report = {
        "schema_version": "0.1.0",
        "role": "independent confirmation; no fitting or selection",
        "temporal_dataset": str(args.temporal_dataset),
        "vision_features": str(args.vision_features),
        "temporal_model": str(args.temporal_model),
        "vision_model": str(args.vision_model),
        "reference_report": str(args.reference_report),
        "camera_selection": cameras,
        "train_prevalence": prevalence,
        "results": results,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FROZEN CONFIRMATION", args.output)
    for checkpoint, values in results.items():
        print("step", checkpoint, "temporal", values["temporal_mlp"], "vision", values["frozen_vision"])


if __name__ == "__main__":
    main()
