import unittest

import numpy as np

from scripts.audit_failure_event_feasibility import (
    detect_drop,
    detect_failed_placement,
    detect_grasp_loss,
    detect_pickup_stall,
    detect_placement_timeout,
    detect_transport_stall,
    first_persistent,
)


class FailureEventFeasibilityTest(unittest.TestCase):
    def test_first_persistent_returns_start_of_confirmed_run(self) -> None:
        condition = np.asarray([False, True, True, False, True, True, True])
        self.assertEqual(first_persistent(condition, 3), 4)

    def test_drop_requires_prior_lift_and_outside_plate(self) -> None:
        data = {
            "lifted": np.asarray([False, True, True, True, True]),
            "running_max_height": np.asarray([0.0, 0.1, 0.1, 0.1, 0.1]),
            "bowl": np.asarray([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.04], [0, 0, 0.03], [0, 0, 0.02]]),
            "plate_xy_distance": np.asarray([1.0, 1.0, 1.0, 1.0, 1.0]),
        }
        self.assertEqual(detect_drop(data, fall=0.04, persistence=3), 2)

    def test_grasp_loss_requires_prior_association(self) -> None:
        data = {
            "lifted": np.asarray([False, True, True, True, True]),
            "bowl_eef_distance": np.asarray([0.3, 0.05, 0.2, 0.2, 0.2]),
            "plate_xy_distance": np.ones(5),
        }
        self.assertEqual(
            detect_grasp_loss(data, close=0.1, separation=0.15, persistence=3), 2
        )

    def test_failed_placement_requires_enter_then_exit(self) -> None:
        data = {"plate_xy_distance": np.asarray([0.3, 0.08, 0.13, 0.14, 0.15])}
        self.assertEqual(
            detect_failed_placement(data, entered=0.1, exited=0.12, persistence=3),
            2,
        )

    def test_pickup_stall_waits_for_absence_of_target_movement(self) -> None:
        data = {
            "bowl_eef_distance": np.asarray([0.2, 0.05, 0.05, 0.05, 0.05]),
            "bowl": np.zeros((5, 3)),
        }
        self.assertEqual(
            detect_pickup_stall(data, approach=0.1, wait=3, movement=0.04), 4
        )

    def test_pickup_stall_accepts_horizontal_target_motion(self) -> None:
        data = {
            "bowl_eef_distance": np.asarray([0.2, 0.09, 0.08, 0.07, 0.06]),
            "bowl": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.0, 0.0],
                 [0.05, 0.0, 0.0], [0.06, 0.0, 0.0]]
            ),
        }
        self.assertIsNone(
            detect_pickup_stall(data, approach=0.1, wait=3, movement=0.04)
        )

    def test_pickup_stall_does_not_search_again_after_valid_first_approach(self) -> None:
        data = {
            "bowl_eef_distance": np.asarray([0.2, 0.09, 0.08, 0.07, 0.06, 0.05]),
            "bowl": np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.0],
                 [0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]
            ),
        }
        self.assertIsNone(
            detect_pickup_stall(data, approach=0.1, wait=2, movement=0.04)
        )

    def test_transport_stall_requires_prior_lift(self) -> None:
        data = {
            "lifted": np.asarray([False, True, True, True, True]),
            "plate_xy_distance": np.asarray([0.5, 0.4, 0.4, 0.4, 0.4]),
        }
        self.assertEqual(
            detect_transport_stall(data, window=2, min_improvement=0.01), 3
        )

    def test_placement_timeout_is_suppressed_by_success(self) -> None:
        data = {
            "plate_xy_distance": np.asarray([0.2, 0.08, 0.08, 0.08, 0.08]),
            "success": np.asarray([False, False, False, True, True]),
        }
        self.assertIsNone(detect_placement_timeout(data, entered=0.1, wait=2))


if __name__ == "__main__":
    unittest.main()
