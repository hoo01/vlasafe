from __future__ import annotations

import unittest

import numpy as np

from vlasafe.faults import fault_action, fault_observation
from vlasafe.monitors import (
    ActionProtocolError,
    TimestampProtocolError,
    validate_action_for_execution,
    validate_observation_timestamp,
)

from scripts.evaluate_a2_consistency import wilson_interval


class FaultInjectionTest(unittest.TestCase):
    def test_action_swap_xy_preserves_valid_shape_and_values(self) -> None:
        action = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
        result = fault_action(action, "action_swap_xy")
        np.testing.assert_array_equal(result, [[2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        np.testing.assert_array_equal(action, [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])

    def test_camera_swap_does_not_mutate_recorded_observation(self) -> None:
        source = {"pixels": {"image": "main", "image2": "wrist"}, "robot_state": {}}
        result = fault_observation(source, "camera_swap")
        self.assertEqual(result["pixels"], {"image": "wrist", "image2": "main"})
        self.assertEqual(source["pixels"], {"image": "main", "image2": "wrist"})

    def test_unknown_fault_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fault mode"):
            fault_action(np.zeros((1, 7)), "typo")


class RuntimeActionMonitorTest(unittest.TestCase):
    def test_rejects_wrong_shape_and_nonfinite_action(self) -> None:
        bounds = np.full(7, -1.0), np.full(7, 1.0)
        with self.assertRaisesRegex(ActionProtocolError, "shape"):
            validate_action_for_execution(np.zeros((1, 6)), *bounds)
        action = np.zeros((1, 7))
        action[0, 2] = np.nan
        with self.assertRaisesRegex(ActionProtocolError, "NaN or Inf"):
            validate_action_for_execution(action, *bounds)

    def test_clips_and_records_explicit_range_violation(self) -> None:
        action = np.asarray([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0]])
        executed, result = validate_action_for_execution(
            action, np.full(7, -1.0), np.full(7, 1.0)
        )
        np.testing.assert_array_equal(executed, [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
        self.assertTrue(result["range_violation"])
        self.assertEqual(result["range_violation_indices"], [1, 6])
        self.assertEqual(result["action_clip_linf"], 2.0)

    def test_valid_action_passes_without_clipping(self) -> None:
        action = np.zeros((1, 7), dtype=np.float32)
        executed, result = validate_action_for_execution(
            action, np.full(7, -1.0), np.full(7, 1.0)
        )
        np.testing.assert_array_equal(executed, action)
        self.assertFalse(result["range_violation"])
        self.assertFalse(result["action_clipped"])

    def test_rejects_invalid_or_regressed_timestamp(self) -> None:
        self.assertEqual(validate_observation_timestamp(100, None), 100)
        self.assertEqual(validate_observation_timestamp(100, 100), 100)
        with self.assertRaisesRegex(TimestampProtocolError, "integer"):
            validate_observation_timestamp(100.0, None)
        with self.assertRaisesRegex(TimestampProtocolError, "regressed"):
            validate_observation_timestamp(99, 100)


class IntervalTest(unittest.TestCase):
    def test_wilson_interval_handles_boundary_counts(self) -> None:
        self.assertEqual(wilson_interval(0, 10)[0], 0.0)
        self.assertEqual(wilson_interval(10, 10)[1], 1.0)
        low, high = wilson_interval(1, 10)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.1)


if __name__ == "__main__":
    unittest.main()
