"""Validation for finalized rollout sidecar artifacts."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schema import SCHEMA_VERSION


class ArtifactValidationError(ValueError):
    """Raised when an episode artifact violates the recording contract."""


@dataclass(frozen=True)
class ValidationSummary:
    episode_id: str
    num_steps: int
    video_frames: int
    success: bool
    clipped_steps: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read valid JSON from {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path.name} must contain a JSON object")
    return value


def _probe_video_frames(path: Path) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ArtifactValidationError("ffprobe is required to validate video alignment") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise ArtifactValidationError(f"cannot probe {path.name}: {detail}") from exc
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise ArtifactValidationError(f"ffprobe returned no frame count for {path.name}") from exc


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return False


def validate_episode(
    episode_dir: str | Path,
    *,
    video_probe: Callable[[Path], int] = _probe_video_frames,
) -> ValidationSummary:
    """Validate one finalized episode and return its compact summary."""
    root = Path(episode_dir)
    errors: list[str] = []
    if root.name.endswith(".incomplete"):
        errors.append("episode directory is marked incomplete")

    required = ["COMPLETE", "metadata.json", "result.json", "steps.jsonl", "main_camera.mp4"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing required file: {name}")
    if errors:
        raise ArtifactValidationError("; ".join(errors))

    metadata = _read_json(root / "metadata.json")
    result = _read_json(root / "result.json")
    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        errors.append("metadata.episode_id must be a non-empty string")
        episode_id = "<invalid>"
    elif episode_id != root.name:
        errors.append(f"directory name {root.name!r} != metadata episode_id {episode_id!r}")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {metadata.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    for key in ("task", "task_id", "seed", "policy_id", "policy_revision", "lerobot_revision", "resolved_config", "started_at_utc"):
        if key not in metadata:
            errors.append(f"metadata missing field: {key}")

    rows: list[dict[str, Any]] = []
    try:
        with (root / "steps.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    errors.append(f"blank JSONL row at line {line_number}")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid JSONL at line {line_number}: {exc.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"JSONL line {line_number} is not an object")
                    continue
                rows.append(row)
    except OSError as exc:
        errors.append(f"cannot read steps.jsonl: {exc}")

    required_step_fields = (
        "episode_id", "step_id", "frame_id", "observation_timestamp_ns",
        "inference_started_ns", "inference_finished_ns", "action_executed_ns",
        "inference_latency_ms", "control_latency_ms", "predicted_action_chunk",
        "executed_action", "proprioception", "reward", "terminated", "truncated",
        "success", "deploy_metadata", "label_only",
    )
    clipped_steps = 0
    previous_observation_ns: int | None = None
    for index, row in enumerate(rows):
        missing = [key for key in required_step_fields if key not in row]
        if missing:
            errors.append(f"step {index} missing fields: {', '.join(missing)}")
            continue
        if row["episode_id"] != episode_id:
            errors.append(f"step {index} episode_id mismatch")
        if row["step_id"] != index or row["frame_id"] != index:
            errors.append(f"step/frame IDs are not contiguous at row {index}")
        obs_ns = row["observation_timestamp_ns"]
        if not all(isinstance(row[key], int) for key in (
            "observation_timestamp_ns", "inference_started_ns", "inference_finished_ns", "action_executed_ns"
        )):
            errors.append(f"step {index} timestamps must be integers")
        else:
            if not (obs_ns <= row["inference_started_ns"] <= row["inference_finished_ns"] <= row["action_executed_ns"]):
                errors.append(f"step {index} timestamps are not monotonic")
            if previous_observation_ns is not None and obs_ns < previous_observation_ns:
                errors.append(f"observation timestamp regressed at step {index}")
            previous_observation_ns = obs_ns
        for key in ("inference_latency_ms", "control_latency_ms", "predicted_action_chunk", "executed_action", "proprioception", "reward"):
            if not _finite_numbers(row[key]):
                errors.append(f"step {index} field {key} contains non-finite or invalid values")
        if not isinstance(row["predicted_action_chunk"], list) or not row["predicted_action_chunk"]:
            errors.append(f"step {index} predicted_action_chunk must be non-empty")
        if not isinstance(row["executed_action"], list) or not row["executed_action"]:
            errors.append(f"step {index} executed_action must be non-empty")
        deploy = row["deploy_metadata"]
        if not isinstance(deploy, dict):
            errors.append(f"step {index} deploy_metadata must be an object")
        elif deploy.get("action_clipped") is True:
            clipped_steps += 1

    expected_steps = result.get("num_steps")
    if not isinstance(expected_steps, int) or expected_steps < 0:
        errors.append("result.num_steps must be a non-negative integer")
    elif expected_steps != len(rows):
        errors.append(f"result.num_steps={expected_steps} but JSONL has {len(rows)} rows")
    if not isinstance(result.get("success"), bool):
        errors.append("result.success must be boolean")
    if not isinstance(result.get("termination_reason"), str) or not result.get("termination_reason"):
        errors.append("result.termination_reason must be a non-empty string")

    video_frames = -1
    for video_path in sorted(root.glob("*_camera.mp4")):
        try:
            current_frames = video_probe(video_path)
        except ArtifactValidationError as exc:
            errors.append(str(exc))
            current_frames = -1
        if video_path.name == "main_camera.mp4":
            video_frames = current_frames
        if current_frames != len(rows):
            errors.append(
                f"{video_path.name} has {current_frames} frames but JSONL has {len(rows)} rows"
            )

    if errors:
        raise ArtifactValidationError("\n- " + "\n- ".join(errors))
    return ValidationSummary(
        episode_id=episode_id,
        num_steps=len(rows),
        video_frames=video_frames,
        success=result["success"],
        clipped_steps=clipped_steps,
    )
