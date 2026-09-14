"""Deterministic action checks applied immediately before environment execution."""

from __future__ import annotations

from typing import Any

import numpy as np


class ActionProtocolError(ValueError):
    pass


class TimestampProtocolError(ValueError):
    pass


def validate_observation_timestamp(current_ns: Any, previous_ns: int | None) -> int:
    if not isinstance(current_ns, (int, np.integer)):
        raise TimestampProtocolError("observation timestamp must be an integer")
    current = int(current_ns)
    if current < 0:
        raise TimestampProtocolError("observation timestamp must be non-negative")
    if previous_ns is not None and current < previous_ns:
        raise TimestampProtocolError("observation timestamp regressed")
    return current


def validate_action_for_execution(
    action: Any, low: Any, high: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(action)
    lower = np.asarray(low, dtype=np.float32).reshape(-1)
    upper = np.asarray(high, dtype=np.float32).reshape(-1)
    expected_shape = (1, len(lower))
    if array.shape != expected_shape:
        raise ActionProtocolError(
            f"action shape {array.shape} does not match expected {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ActionProtocolError(f"action dtype {array.dtype} is not numeric")
    numeric = array.astype(np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ActionProtocolError("action contains NaN or Inf")
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("invalid action-space bounds")
    below = numeric[0] < lower
    above = numeric[0] > upper
    range_violation = below | above
    executed = np.clip(numeric, lower, upper).astype(np.float32, copy=False)
    clip_linf = float(np.max(np.abs(executed - numeric)))
    return executed, {
        "schema_valid": True,
        "range_violation": bool(range_violation.any()),
        "range_violation_indices": np.flatnonzero(range_violation).astype(int).tolist(),
        "action_clipped": clip_linf > 0.0,
        "action_clip_linf": clip_linf,
    }
