from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlasafe.rollout.validator import ArtifactValidationError, validate_episode


def write_episode(root: Path, *, frame_count: int = 2) -> Path:
    episode = root / "ep-valid"
    episode.mkdir()
    (episode / "COMPLETE").touch()
    (episode / "main_camera.mp4").touch()
    (episode / "metadata.json").write_text(json.dumps({
        "episode_id": "ep-valid", "task": "libero_spatial", "task_id": 0,
        "seed": 42, "initial_state_id": 0, "policy_id": "policy",
        "policy_revision": "revision", "lerobot_revision": "revision",
        "resolved_config": {"fps": 20}, "started_at_utc": "2026-09-10T00:00:00+00:00",
        "schema_version": "0.1.0",
    }), encoding="utf-8")
    (episode / "result.json").write_text(json.dumps({
        "num_steps": 2, "success": False, "termination_reason": "horizon",
    }), encoding="utf-8")
    rows = []
    for index in range(2):
        rows.append({
            "episode_id": "ep-valid", "step_id": index, "frame_id": index,
            "observation_timestamp_ns": 100 + index * 100,
            "inference_started_ns": 110 + index * 100,
            "inference_finished_ns": 120 + index * 100,
            "action_executed_ns": 130 + index * 100,
            "inference_latency_ms": 0.01, "control_latency_ms": 0.02,
            "predicted_action_chunk": [[0.0] * 7], "executed_action": [0.0] * 7,
            "proprioception": {"eef.pos": [0.0, 0.0, 0.0]}, "reward": 0.0,
            "terminated": False, "truncated": False, "success": False,
            "deploy_metadata": {"action_clipped": index == 1}, "label_only": {},
        })
    (episode / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return episode


class ArtifactValidatorTest(unittest.TestCase):
    def test_accepts_aligned_finalized_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = write_episode(Path(tmp))
            summary = validate_episode(episode, video_probe=lambda _: 2)
            self.assertEqual(summary.num_steps, 2)
            self.assertEqual(summary.video_frames, 2)
            self.assertEqual(summary.clipped_steps, 1)

    def test_rejects_video_step_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = write_episode(Path(tmp))
            with self.assertRaisesRegex(ArtifactValidationError, "video has 1 frames"):
                validate_episode(episode, video_probe=lambda _: 1)

    def test_rejects_noncontiguous_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = write_episode(Path(tmp))
            rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
            rows[1]["step_id"] = 3
            (episode / "steps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ArtifactValidationError, "not contiguous"):
                validate_episode(episode, video_probe=lambda _: 2)

    def test_rejects_incomplete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = write_episode(Path(tmp))
            incomplete = episode.with_name("ep-valid.incomplete")
            episode.rename(incomplete)
            with self.assertRaisesRegex(ArtifactValidationError, "marked incomplete"):
                validate_episode(incomplete, video_probe=lambda _: 2)


if __name__ == "__main__":
    unittest.main()
