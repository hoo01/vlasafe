"""Command-effect consistency features using intended translation and observed EEF motion."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def consistency_errors(
    rows: Sequence[dict[str, Any]],
    command_epsilon: float = 1e-3,
    effect_epsilon: float = 1e-5,
) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for index in range(len(rows) - 1):
        deploy = rows[index].get("deploy_metadata", {})
        intended = np.asarray(deploy.get("intended_action", []), dtype=np.float64)
        if intended.shape != (7,) or not np.isfinite(intended).all():
            continue
        command = intended[:3]
        command_norm = float(np.linalg.norm(command))
        if command_norm < command_epsilon:
            continue
        current = np.asarray(rows[index]["proprioception"]["eef.pos"], dtype=np.float64)
        following = np.asarray(rows[index + 1]["proprioception"]["eef.pos"], dtype=np.float64)
        effect = following - current
        effect_norm = float(np.linalg.norm(effect))
        if effect_norm < effect_epsilon:
            cosine = 0.0
            error = 1.0
        else:
            cosine = float(np.dot(command, effect) / (command_norm * effect_norm))
            cosine = float(np.clip(cosine, -1.0, 1.0))
            error = 1.0 - cosine
        step_id = int(rows[index].get("step_id", index))
        results.append(
            {
                "step": step_id,
                "command_norm": command_norm,
                "effect_norm": effect_norm,
                "cosine": cosine,
                "error": error,
            }
        )
    return results


def window_scores(
    errors: Sequence[dict[str, float | int]], consecutive: int = 3
) -> list[dict[str, float | int]]:
    if consecutive <= 0:
        raise ValueError("consecutive must be positive")
    result: list[dict[str, float | int]] = []
    for end in range(consecutive - 1, len(errors)):
        window = errors[end - consecutive + 1 : end + 1]
        steps = [int(row["step"]) for row in window]
        if any(right != left + 1 for left, right in zip(steps, steps[1:])):
            continue
        result.append({
            "step": steps[-1] + 1,
            "score": float(np.mean([float(row["error"]) for row in window])),
        })
    return result


def episode_consistency_score(
    errors: Sequence[dict[str, float | int]], consecutive: int = 3
) -> float:
    windows = window_scores(errors, consecutive)
    if not windows:
        return float("nan")
    return max(float(row["score"]) for row in windows)


def first_alarm_step(
    errors: Sequence[dict[str, float | int]], threshold: float, consecutive: int = 3
) -> int | None:
    for row in window_scores(errors, consecutive):
        if float(row["score"]) > threshold:
            return int(row["step"])
    return None
