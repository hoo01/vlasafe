from __future__ import annotations

import unittest

from vlasafe.predictor_inputs import ALLOWED_FIELDS, select_deploy_inputs


def rollout_step() -> dict:
    return {
        "episode_id": "episode-1",
        "step_id": 3,
        "frame_id": 3,
        "proprioception": {"qpos": [0.1, 0.2]},
        "predicted_action_chunk": [[0.0] * 7],
        "executed_action": [0.0] * 7,
        "inference_latency_ms": 10.0,
        "control_latency_ms": 2.0,
        "reward": 1.0,
        "terminated": True,
        "truncated": False,
        "success": True,
        "label_only": {"joint_violation": True, "object_pose": [1.0, 2.0, 3.0]},
    }


class PredictorInputBoundaryTest(unittest.TestCase):
    def test_default_projection_contains_only_allowlisted_fields(self) -> None:
        projected = select_deploy_inputs(rollout_step())
        self.assertEqual(set(projected), set(ALLOWED_FIELDS))
        self.assertFalse(
            {"label_only", "reward", "success", "terminated", "truncated"}
            & projected.keys()
        )

    def test_projection_is_a_copy(self) -> None:
        source = rollout_step()
        projected = select_deploy_inputs(source)
        projected["proprioception"]["qpos"][0] = 99.0
        self.assertEqual(source["proprioception"]["qpos"][0], 0.1)

    def test_rejects_privileged_or_target_field(self) -> None:
        for field in ("label_only", "label_only.object_pose", "success", "reward"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_deploy_inputs(rollout_step(), [field])

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(ValueError):
            select_deploy_inputs(rollout_step(), ["simulator_state"])

    def test_rejects_missing_allowlisted_field(self) -> None:
        source = rollout_step()
        del source["executed_action"]
        with self.assertRaises(KeyError):
            select_deploy_inputs(source)


if __name__ == "__main__":
    unittest.main()
