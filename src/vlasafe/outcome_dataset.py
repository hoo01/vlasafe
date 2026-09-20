"""Leakage-safe feature construction for episode outcome prediction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def normalize_and_flatten_temporal(
    data: Any, *, include_checkpoint_progress: bool = True
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize temporal channels on train rows and optionally append checkpoint progress."""

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
    feature_parts = [normalized.reshape(len(normalized), -1), mask]
    if include_checkpoint_progress:
        progress = (data["checkpoint_step"].astype(np.float32) / 280.0)[:, None]
        feature_parts.append(progress)
    features = np.concatenate(feature_parts, axis=1)
    stats = {
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "feature_order": ["proprio_25", "executed_action_7", "timing_2"],
        "feature_dim": int(features.shape[1]),
        "checkpoint_progress_feature": include_checkpoint_progress,
        "progress_denominator": 280 if include_checkpoint_progress else None,
    }
    return features.astype(np.float32), stats

from .predictor_inputs import select_deploy_inputs


PROPRIO_FIELDS: tuple[tuple[str, int], ...] = (
    ("eef.pos", 3),
    ("eef.quat", 4),
    ("gripper.qpos", 2),
    ("gripper.qvel", 2),
    ("joints.pos", 7),
    ("joints.vel", 7),
)


def flatten_proprioception(proprioception: dict[str, Any]) -> np.ndarray:
    """Flatten the frozen, non-redundant proprioception schema to 25 floats."""

    parts = []
    for name, expected_size in PROPRIO_FIELDS:
        if name not in proprioception:
            raise KeyError(f"missing proprioception field: {name}")
        values = np.asarray(proprioception[name], dtype=np.float32).reshape(-1)
        if values.size != expected_size:
            raise ValueError(
                f"{name} has {values.size} values, expected {expected_size}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        parts.append(values)
    return np.concatenate(parts)


def build_history_window(
    rows: Sequence[dict[str, Any]], checkpoint_step: int, window_size: int
) -> dict[str, np.ndarray]:
    """Build a causal window ending after checkpoint_step, left-padded at reset."""

    if checkpoint_step < 0 or checkpoint_step >= len(rows):
        raise IndexError(f"checkpoint step {checkpoint_step} outside episode")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    start = max(0, checkpoint_step - window_size + 1)
    selected = list(rows[start : checkpoint_step + 1])
    pad_count = window_size - len(selected)
    padded = [selected[0]] * pad_count + selected
    mask = np.asarray([0.0] * pad_count + [1.0] * len(selected), dtype=np.float32)

    states = []
    actions = []
    timings = []
    frame_ids = []
    for row in padded:
        deploy = select_deploy_inputs(
            row,
            fields=(
                "frame_id",
                "proprioception",
                "executed_action",
                "inference_latency_ms",
                "control_latency_ms",
            ),
        )
        state = flatten_proprioception(deploy["proprioception"])
        action = np.asarray(deploy["executed_action"], dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.isfinite(action).all():
            raise ValueError("executed_action must contain 7 finite values")
        timing = np.asarray(
            [deploy["inference_latency_ms"], deploy["control_latency_ms"]],
            dtype=np.float32,
        )
        if not np.isfinite(timing).all():
            raise ValueError("timing features must be finite")
        states.append(state)
        actions.append(action)
        timings.append(timing)
        frame_ids.append(deploy["frame_id"])

    return {
        "state": np.stack(states),
        "action": np.stack(actions),
        "timing": np.stack(timings),
        "mask": mask,
        "frame_id": np.asarray(frame_ids, dtype=np.int64),
    }
