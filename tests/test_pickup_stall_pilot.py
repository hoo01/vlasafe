from __future__ import annotations

import unittest

import numpy as np

from scripts.build_pickup_stall_pilot import aligned_sample, allocate_groups


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
        split = allocate_groups(list(range(11)), list(range(11, 20)), seed=7)
        self.assertEqual(set(split), set(range(20)))
        self.assertEqual(sum(value == "train" for value in split.values()), 12)
        self.assertEqual(sum(value == "validation" for value in split.values()), 4)
        self.assertEqual(sum(value == "test" for value in split.values()), 4)


if __name__ == "__main__":
    unittest.main()
