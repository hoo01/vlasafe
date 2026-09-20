from __future__ import annotations

import unittest

import numpy as np

from vlasafe.outcome_dataset import (
    build_history_window,
    flatten_proprioception,
    normalize_and_flatten_temporal,
)


def proprio() -> dict[str, list[float]]:
    return {
        "eef.mat": [0.0] * 9,
        "eef.pos": [0.0] * 3,
        "eef.quat": [0.0] * 4,
        "gripper.qpos": [0.0] * 2,
        "gripper.qvel": [0.0] * 2,
        "joints.pos": [0.0] * 7,
        "joints.vel": [0.0] * 7,
    }


def step(index: int) -> dict:
    return {
        "step_id": index,
        "frame_id": index,
        "proprioception": proprio(),
        "executed_action": [float(index)] * 7,
        "inference_latency_ms": 10.0,
        "control_latency_ms": 2.0,
        "success": True,
        "label_only": {"object_pose": [999.0]},
    }


class OutcomeDatasetTest(unittest.TestCase):
    def test_frozen_proprioception_schema_is_25_dimensional(self) -> None:
        result = flatten_proprioception(proprio())
        self.assertEqual(result.shape, (25,))

    def test_history_is_causal_and_left_padded(self) -> None:
        result = build_history_window([step(0), step(1), step(2)], 1, 4)
        self.assertEqual(result["state"].shape, (4, 25))
        self.assertEqual(result["action"].shape, (4, 7))
        np.testing.assert_array_equal(result["mask"], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result["frame_id"], [0, 0, 0, 1])
        self.assertEqual(result["action"][-1, 0], 1.0)

    def test_window_never_copies_target_or_privileged_fields(self) -> None:
        result = build_history_window([step(0)], 0, 1)
        self.assertNotIn("success", result)
        self.assertNotIn("label_only", result)

    def test_rejects_checkpoint_outside_episode(self) -> None:
        with self.assertRaises(IndexError):
            build_history_window([step(0)], 1, 1)

    def test_checkpoint_progress_ablation_changes_only_explicit_feature(self) -> None:
        data = {
            "split": np.asarray(["train", "test"]),
            "mask": np.ones((2, 2), dtype=np.float32),
            "state": np.zeros((2, 2, 25), dtype=np.float32),
            "action": np.zeros((2, 2, 7), dtype=np.float32),
            "timing": np.zeros((2, 2, 2), dtype=np.float32),
            "checkpoint_step": np.asarray([40, 80]),
        }
        with_progress, with_stats = normalize_and_flatten_temporal(
            data, include_checkpoint_progress=True
        )
        without_progress, without_stats = normalize_and_flatten_temporal(
            data, include_checkpoint_progress=False
        )
        self.assertEqual(with_progress.shape[1], without_progress.shape[1] + 1)
        np.testing.assert_array_equal(with_progress[:, :-1], without_progress)
        np.testing.assert_allclose(with_progress[:, -1], [40 / 280, 80 / 280])
        self.assertTrue(with_stats["checkpoint_progress_feature"])
        self.assertFalse(without_stats["checkpoint_progress_feature"])


if __name__ == "__main__":
    unittest.main()
