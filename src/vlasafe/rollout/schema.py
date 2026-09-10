"""Public, JSON-serializable schema for synchronized rollout sidecars."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    task: str
    task_id: int
    seed: int
    initial_state_id: str | int | None
    policy_id: str
    policy_revision: str
    lerobot_revision: str
    resolved_config: dict[str, Any]
    started_at_utc: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    step_id: int
    frame_id: int
    observation_timestamp_ns: int
    inference_started_ns: int
    inference_finished_ns: int
    action_executed_ns: int
    inference_latency_ms: float
    control_latency_ms: float
    predicted_action_chunk: list[list[float]]
    executed_action: list[float]
    proprioception: dict[str, list[float]]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    deploy_metadata: dict[str, Any] = field(default_factory=dict)
    # Privileged simulator information is label-only. Predictor loaders must reject it.
    label_only: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_id < 0 or self.frame_id < 0:
            raise ValueError("step_id and frame_id must be non-negative")
        if self.frame_id != self.step_id:
            raise ValueError("MVP requires frame_id == step_id")
        if self.inference_finished_ns < self.inference_started_ns:
            raise ValueError("inference timestamps are not monotonic")
        if self.action_executed_ns < self.observation_timestamp_ns:
            raise ValueError("action precedes its observation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

