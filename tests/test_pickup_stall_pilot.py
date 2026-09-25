from __future__ import annotations

import unittest

import numpy as np

from scripts.build_pickup_stall_pilot import (
    aligned_sample,
    alignment_exclusion_reason,
    allocate_groups,
    split_support,
)


class PickupStallPilotTest(unittest.TestCase):
    def test_aligned_positive_uses_first_approach(self) -> None:
        data = {
            "bowl_eef_distance": np.r_[np.ones(20), np.full(50, 0.05)],
            "bowl": np.zeros((70, 3)),
        }
        sample = aligned_sample(data, history=16, wait=40, horizon=20, movement=0.04)
        self.assertEqual(sample["first_approach_step"], 20)
        self.assertEqual(sample["checkpoint_step"], 40)
        self.assertEqual(sample["event_step"], 60)
        self.assertEqual(sample["pickup_stall"], 1)

    def test_aligned_negative_moves_in_same_window(self) -> None:
        bowl = np.zeros((70, 3))
        bowl[45:, 0] = 0.05
        data = {
            "bowl_eef_distance": np.r_[np.ones(20), np.full(50, 0.05)],
            "bowl": bowl,
        }
        sample = aligned_sample(data, history=16, wait=40, horizon=20, movement=0.04)
        self.assertEqual(sample["pickup_stall"], 0)
        self.assertIsNone(sample["event_step"])

    def test_group_allocation_has_no_overlap(self) -> None:
        split = allocate_groups(
            list(range(9)), [9, 10], list(range(11, 20)), seed=7
        )
        self.assertEqual(set(split), set(range(20)))
        self.assertEqual(sum(value == "train" for value in split.values()), 12)
        self.assertEqual(sum(value == "validation" for value in split.values()), 4)
        self.assertEqual(sum(value == "test" for value in split.values()), 4)
        self.assertEqual(split[9], "train")
        self.assertEqual(split[10], "train")

    def test_support_counts_only_same_group_negatives(self) -> None:
        rows = [
            {"initial_state_id": 1, "pickup_stall": 1},
            {"initial_state_id": 1, "pickup_stall": 0},
            {"initial_state_id": 2, "pickup_stall": 1},
        ]
        support = split_support(rows)
        self.assertEqual(support["positives_with_same_state_negative"], 1)
        self.assertEqual(support["positive_same_state_negative_fraction"], 0.5)

    def test_reports_incomplete_window(self) -> None:
        data = {
            "bowl_eef_distance": np.r_[np.ones(20), np.full(10, 0.05)],
            "bowl": np.zeros((30, 3)),
        }
        self.assertEqual(
            alignment_exclusion_reason(data, history=16, wait=40, horizon=20),
            "incomplete_approach_window",
        )


if __name__ == "__main__":
    unittest.main()
