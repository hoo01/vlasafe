"""Apply frozen v0.1 predictors to the predeclared v0.3 stage cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_frozen_confirmation import _temporal_features
from train_frozen_vision_baseline import predict as vision_predict
from train_frozen_vision_baseline import select_camera_features
from train_temporal_mlp import _ensemble_predict
from vlasafe.cluster_bootstrap import cluster_bootstrap


def load_group_map(manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for split in manifest["splits"].values() for row in split]
    return {str(row["episode_id"]): int(row["initial_state_id"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("temporal_dataset", type=Path)
    parser.add_argument("vision_features", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--temporal-model", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    device = torch.device(args.device)
    temporal_data = np.load(args.temporal_dataset)
    vision_data = np.load(args.vision_features)
    group_map = load_group_map(args.manifest)
    temporal_artifact = torch.load(args.temporal_model, map_location="cpu", weights_only=True)
    vision_artifact = torch.load(args.vision_model, map_location="cpu", weights_only=True)

    temporal_x = _temporal_features(temporal_data, temporal_artifact["normalization"])
    temporal_probability = _ensemble_predict(
        temporal_artifact["model_states"], temporal_x, device
    )
    vision_x, cameras = select_camera_features(
        vision_data["features"].astype(np.float32), vision_data["camera"], "dual"
    )
    vision_probability = np.empty(len(vision_x), dtype=np.float64)
    for checkpoint in np.unique(vision_data["checkpoint_step"]).astype(int):
        key = str(int(checkpoint))
        if key not in vision_artifact["models"]:
            raise KeyError(f"frozen vision artifact has no checkpoint {key}")
        selected = vision_data["checkpoint_step"].astype(int) == checkpoint
        metadata = vision_artifact["metadata"][key]
        mean = np.asarray(metadata["mean"], dtype=np.float32)
        std = np.asarray(metadata["std"], dtype=np.float32)
        normalized = ((vision_x[selected] - mean) / std).astype(np.float32)
        vision_probability[selected] = vision_predict(
            vision_artifact["models"][key], normalized, device
        )

    predictions = []
    results = {}
    specifications = (
        ("vision_step80", "vision", 80, vision_data, vision_probability),
        ("temporal_step120", "temporal", 120, temporal_data, temporal_probability),
    )
    for offset, (result_name, model_name, checkpoint, data, probability) in enumerate(specifications):
        selected = data["checkpoint_step"].astype(int) == checkpoint
        ids = data["episode_id"][selected]
        labels = data["failure"][selected].astype(int)
        values = probability[selected]
        groups = np.asarray([group_map[str(eid)] for eid in ids], dtype=int)
        results[result_name] = {
            "model": model_name,
            "checkpoint_step": checkpoint,
            "episodes": int(len(ids)),
            "initial_state_groups": int(len(np.unique(groups))),
            "failures": int(labels.sum()),
            "successes": int((labels == 0).sum()),
            "cluster_bootstrap": cluster_bootstrap(
                labels, values, groups, args.bootstrap_samples, args.seed + offset
            ),
        }
        for episode_id, label, group, value in zip(ids, labels, groups, values, strict=True):
            predictions.append(
                {
                    "model": model_name,
                    "checkpoint_step": checkpoint,
                    "episode_id": str(episode_id),
                    "initial_state_id": int(group),
                    "failure": int(label),
                    "probability": float(value),
                }
            )

    report = {
        "schema_version": "0.1.0",
        "role": "frozen predictors applied once to predeclared stage cohort",
        "manifest": args.manifest.as_posix(),
        "temporal_dataset": args.temporal_dataset.as_posix(),
        "vision_features": args.vision_features.as_posix(),
        "temporal_model": args.temporal_model.as_posix(),
        "vision_model": args.vision_model.as_posix(),
        "camera_selection": cameras,
        "bootstrap_unit": "initial_state_id",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "results": results,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FROZEN STAGE COHORT", args.output)
    for name, result in results.items():
        print(
            name,
            "episodes", result["episodes"],
            "groups", result["initial_state_groups"],
            "success/failure", f"{result['successes']}/{result['failures']}",
        )
        print("metrics", result["cluster_bootstrap"])


if __name__ == "__main__":
    main()
