"""Train the preregistered state/action temporal MLP outcome baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vlasafe.outcome_metrics import binary_metrics


TRAINING_SEEDS = (0, 1, 2, 3, 4)


class TemporalMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _normalize_and_flatten(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict]:
    train = data["split"] == "train"
    mask = data["mask"].astype(np.float32)
    channels = np.concatenate(
        [data["state"], data["action"], data["timing"]], axis=-1
    ).astype(np.float32)
    valid_train = channels[train][mask[train].astype(bool)]
    mean = valid_train.mean(axis=0)
    std = valid_train.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (channels - mean) / std
    normalized *= mask[..., None]
    progress = (data["checkpoint_step"].astype(np.float32) / 280.0)[:, None]
    features = np.concatenate(
        [normalized.reshape(len(normalized), -1), mask, progress], axis=1
    )
    stats = {
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "feature_order": ["proprio_25", "executed_action_7", "timing_2"],
        "feature_dim": int(features.shape[1]),
        "progress_denominator": 280,
    }
    return features.astype(np.float32), stats


def _fit_one(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TemporalMLP(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(features[train]).to(device)
    y_train = torch.from_numpy(target[train].astype(np.float32)).to(device)
    x_validation = torch.from_numpy(features[validation]).to(device)
    y_validation = torch.from_numpy(target[validation].astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, 501):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 32):
            indexes = order[start : start + 32].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[indexes]), y_train[indexes])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(x_validation), y_validation).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 50:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, best_epoch, best_loss


def _ensemble_predict(
    states: list[dict[str, torch.Tensor]], features: np.ndarray, device: torch.device
) -> np.ndarray:
    predictions = []
    inputs = torch.from_numpy(features).to(device)
    for state in states:
        model = TemporalMLP(features.shape[1]).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            predictions.append(torch.sigmoid(model(inputs)).cpu().numpy())
    return np.mean(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.dataset)
    features, normalization = _normalize_and_flatten(data)
    target = data["failure"].astype(np.int64)
    train = data["split"] == "train"
    validation = data["split"] == "validation"
    test = data["split"] == "test"
    device = torch.device(args.device)

    states = []
    runs = []
    for seed in TRAINING_SEEDS:
        state, epoch, validation_loss = _fit_one(
            features, target, train, validation, seed, device
        )
        states.append(state)
        runs.append(
            {"seed": seed, "best_epoch": epoch, "validation_bce": validation_loss}
        )

    probability = _ensemble_predict(states, features, device)
    train_prevalence = float(target[train].mean())
    report = {
        "dataset": str(args.dataset),
        "model": "temporal_mlp_128_64_dropout_0.1_ensemble_5",
        "selection": "per-seed early stopping on validation BCE; fixed 5-seed ensemble",
        "training_seeds": list(TRAINING_SEEDS),
        "runs": runs,
        "normalization": normalization,
        "results": {},
    }
    for split_name, selected in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        split_result = {}
        for checkpoint in sorted(np.unique(data["checkpoint_step"])):
            bucket = selected & (data["checkpoint_step"] == checkpoint)
            labels = target[bucket]
            split_result[str(int(checkpoint))] = {
                "episodes": int(bucket.sum()),
                "failure_rate": float(labels.mean()),
                "prevalence": binary_metrics(
                    labels, np.full(labels.shape, train_prevalence)
                ),
                "temporal_mlp": binary_metrics(labels, probability[bucket]),
            }
        report["results"][split_name] = split_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_path = args.output.with_suffix(".pt")
    torch.save(
        {
            "model_states": states,
            "normalization": normalization,
            "training_seeds": TRAINING_SEEDS,
        },
        model_path,
    )
    print(f"TEMPORAL MLP {args.output}")
    print(f"MODEL {model_path}")
    print(f"device: {device}")
    print(f"runs: {runs}")
    for split_name in ("validation", "test"):
        print(split_name)
        for checkpoint, result in report["results"][split_name].items():
            print(" step", checkpoint, result)


if __name__ == "__main__":
    main()
