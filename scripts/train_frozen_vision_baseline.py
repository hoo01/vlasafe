"""Train frozen initial-frame and checkpoint-vision linear probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vlasafe.outcome_metrics import binary_metrics


L2_CANDIDATES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def select_camera_features(
    raw_features: np.ndarray, camera_names: np.ndarray, selection: str
) -> tuple[np.ndarray, list[str]]:
    if raw_features.ndim != 3:
        raise ValueError(
            f"expected features with shape (samples, cameras, dim), got {raw_features.shape}"
        )
    names = [str(value) for value in camera_names]
    requested = names if selection == "dual" else [f"{selection}_camera"]
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"camera features not found: {missing}; available={names}")
    indexes = [names.index(name) for name in requested]
    selected = raw_features[:, indexes, :].reshape(len(raw_features), -1)
    return selected, requested


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    l2: float,
) -> tuple[dict[str, torch.Tensor], float]:
    model = LinearProbe(train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for _ in range(1000):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(validation_x), validation_y).item())
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 100:
            break
    if best_state is None:
        raise RuntimeError("linear probe training did not produce a checkpoint")
    return best_state, best_loss


def predict(
    state: dict[str, torch.Tensor], features: np.ndarray, device: torch.device
) -> np.ndarray:
    model = LinearProbe(features.shape[1]).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cameras",
        choices=("main", "wrist", "dual"),
        default="dual",
        help="camera features used by the linear probe",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.features)
    raw_features = data["features"].astype(np.float32)
    features, selected_cameras = select_camera_features(
        raw_features, data["camera"], args.cameras
    )
    target = data["failure"].astype(np.int64)
    splits = data["split"]
    checkpoints = data["checkpoint_step"].astype(np.int64)
    device = torch.device(args.device)
    train_prevalence = float(target[splits == "train"].mean())
    probabilities = np.empty(len(target), dtype=np.float64)
    models = {}
    model_metadata = {}

    for checkpoint in sorted(np.unique(checkpoints)):
        bucket = checkpoints == checkpoint
        train = bucket & (splits == "train")
        validation = bucket & (splits == "validation")
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
            state, validation_bce = fit_probe(
                train_x, train_y, validation_x, validation_y, l2
            )
            validation_probability = predict(state, normalized[validation], device)
            candidates.append(
                {
                    "l2": l2,
                    "validation_bce": validation_bce,
                    "validation_metrics": binary_metrics(
                        target[validation], validation_probability
                    ),
                    "state": state,
                }
            )
        chosen = max(
            candidates,
            key=lambda item: (
                item["validation_metrics"]["auprc"],
                -item["validation_metrics"]["brier"],
            ),
        )
        probabilities[bucket] = predict(chosen["state"], normalized[bucket], device)
        models[str(int(checkpoint))] = chosen["state"]
        model_metadata[str(int(checkpoint))] = {
            "chosen_l2": chosen["l2"],
            "selection_rule": "max validation AUPRC; tie-break min validation Brier",
            "candidates": [
                {
                    "l2": candidate["l2"],
                    "validation_bce": candidate["validation_bce"],
                    **candidate["validation_metrics"],
                }
                for candidate in candidates
            ],
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    initial_probability = {
        str(data["episode_id"][index]): float(probabilities[index])
        for index in np.flatnonzero(checkpoints == 0)
    }
    report = {
        "features": str(args.features),
        "model": "checkpoint-specific frozen ResNet-50 linear probes",
        "camera_selection": args.cameras,
        "cameras": selected_cameras,
        "feature_dim": int(features.shape[1]),
        "encoder_trained": False,
        "probe_selection_uses_test": False,
        "model_metadata": model_metadata,
        "results": {},
        "predictions": [],
    }
    for split_name in ("train", "validation", "test"):
        split_result = {}
        for checkpoint in sorted(np.unique(checkpoints)):
            selected = (splits == split_name) & (checkpoints == checkpoint)
            labels = target[selected]
            episode_ids = data["episode_id"][selected]
            initial = np.asarray([initial_probability[str(value)] for value in episode_ids])
            split_result[str(int(checkpoint))] = {
                "episodes": int(selected.sum()),
                "failure_rate": float(labels.mean()),
                "prevalence": binary_metrics(
                    labels, np.full(labels.shape, train_prevalence)
                ),
                "initial_frame_vision": binary_metrics(labels, initial),
                "checkpoint_vision": binary_metrics(labels, probabilities[selected]),
            }
        report["results"][split_name] = split_result
    for index in range(len(target)):
        episode_id = str(data["episode_id"][index])
        report["predictions"].append(
            {
                "episode_id": episode_id,
                "split": str(splits[index]),
                "checkpoint_step": int(checkpoints[index]),
                "failure": int(target[index]),
                "prevalence_probability": train_prevalence,
                "initial_frame_vision_probability": initial_probability[episode_id],
                "checkpoint_vision_probability": float(probabilities[index]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_path = args.output.with_suffix(".pt")
    torch.save({"models": models, "metadata": model_metadata}, model_path)
    print(f"FROZEN VISION BASELINE {args.output}")
    print(f"MODEL {model_path}")
    print("device:", device)
    for checkpoint, result in report["results"]["test"].items():
        print("test step", checkpoint, result)


if __name__ == "__main__":
    main()
