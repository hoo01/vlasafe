"""Small, explicit deployment-fault injectors for paired stress tests."""

from __future__ import annotations

from typing import Any

import numpy as np


FAULT_MODES = ("none", "action_swap_xy", "camera_swap")


def _validate_mode(mode: str) -> None:
    if mode not in FAULT_MODES:
        raise ValueError(f"unknown fault mode: {mode}")


def fault_observation(observation: dict[str, Any], mode: str) -> dict[str, Any]:
    _validate_mode(mode)
    if mode != "camera_swap":
        return observation
    pixels = dict(observation["pixels"])
    pixels["image"], pixels["image2"] = pixels["image2"], pixels["image"]
    return {**observation, "pixels": pixels}


def fault_action(action: np.ndarray, mode: str) -> np.ndarray:
    _validate_mode(mode)
    result = np.asarray(action).copy()
    if mode == "action_swap_xy":
        if result.shape[-1] != 7:
            raise ValueError(f"action_swap_xy expects 7 action dimensions, got {result.shape}")
        result[..., [0, 1]] = result[..., [1, 0]]
    return result
