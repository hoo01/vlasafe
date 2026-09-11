"""Deployment-input allowlist for outcome predictors.

Simulator-only labels remain in rollout artifacts for supervision and analysis, but
this module is the sole boundary used to construct predictor inputs from JSONL.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


INDEX_FIELDS = frozenset({"step_id", "frame_id"})
MODEL_INPUT_FIELDS = frozenset(
    {
        "proprioception",
        "predicted_action_chunk",
        "executed_action",
        "inference_latency_ms",
        "control_latency_ms",
    }
)
ALLOWED_FIELDS = INDEX_FIELDS | MODEL_INPUT_FIELDS
FORBIDDEN_FIELDS = frozenset(
    {
        "label_only",
        "reward",
        "success",
        "terminated",
        "truncated",
    }
)


def select_deploy_inputs(
    step: dict[str, Any],
    fields: Iterable[str] = ALLOWED_FIELDS,
) -> dict[str, Any]:
    """Copy only explicitly allowlisted deploy-time fields from one rollout step."""

    requested = frozenset(fields)
    forbidden = {
        field
        for field in requested
        if field.split(".", maxsplit=1)[0] in FORBIDDEN_FIELDS
    }
    if forbidden:
        raise ValueError(f"privileged or target fields requested: {sorted(forbidden)}")
    unknown = requested - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"fields are not in deploy-input allowlist: {sorted(unknown)}")
    missing = requested - step.keys()
    if missing:
        raise KeyError(f"rollout step is missing required deploy fields: {sorted(missing)}")
    return {field: deepcopy(step[field]) for field in sorted(requested)}
