from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from vlasafe.rollout import EpisodeMetadata, SidecarRecorder, StepRecord


def metadata(episode_id: str) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_id=episode_id,
        task="libero_spatial",
        task_id=0,
        seed=42,
        initial_state_id=0,
        policy_id="smoke/zero-action",
        policy_revision="none",
        lerobot_revision="archive-unpinned",
        resolved_config={"fps": 20},
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def step(episode_id: str, step_id: int) -> StepRecord:
    return StepRecord(
        episode_id=episode_id,
        step_id=step_id,
        frame_id=step_id,
        observation_timestamp_ns=100,
        inference_started_ns=110,
        inference_finished_ns=120,
        action_executed_ns=130,
        inference_latency_ms=0.00001,
        control_latency_ms=0.00003,
        predicted_action_chunk=[[0.0] * 7],
        executed_action=[0.0] * 7,
        proprioception={"eef_pos": [0.0, 0.0, 0.0]},
        reward=0.0,
        terminated=False,
        truncated=False,
        success=False,
        label_only={"self_collision": False},
    )


class SidecarRecorderTest(unittest.TestCase):
    def test_finalize_is_complete_and_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SidecarRecorder(tmp, metadata("ep-000001"))
            recorder.add_step(step("ep-000001", 0))
            final_dir = recorder.finalize(termination_reason="smoke_test", success=False)
            self.assertTrue((final_dir / "COMPLETE").is_file())
            self.assertFalse((Path(tmp) / "ep-000001.incomplete").exists())
            records = [json.loads(line) for line in (final_dir / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["step_id"], 0)

    def test_abort_remains_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SidecarRecorder(tmp, metadata("ep-000002"))
            partial_dir = recorder.abort()
            self.assertTrue(partial_dir.is_dir())
            self.assertFalse((partial_dir / "COMPLETE").exists())

    def test_rejects_step_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SidecarRecorder(tmp, metadata("ep-000003"))
            with self.assertRaises(ValueError):
                recorder.add_step(step("ep-000003", 1))
            recorder.abort()


if __name__ == "__main__":
    unittest.main()

