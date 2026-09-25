from __future__ import annotations

import unittest

import numpy as np

from vlasafe.pilot_metrics import cluster_bootstrap, matched_pair_accuracy


class PickupStallTrainingTest(unittest.TestCase):
    def test_same_state_pair_accuracy(self) -> None:
        result = matched_pair_accuracy(
            np.asarray([1, 0, 1, 0]),
            np.asarray([0.9, 0.1, 0.4, 0.6]),
            np.asarray([1, 1, 2, 2]),
        )
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["correct"], 1)

    def test_cluster_bootstrap_keeps_groups_whole(self) -> None:
        target = np.asarray([1, 0, 1, 0])
        probabilities = {
            "prevalence": np.full(4, 0.5),
            "stage_only": np.asarray([0.6, 0.4, 0.6, 0.4]),
            "temporal_mlp": np.asarray([0.9, 0.1, 0.8, 0.2]),
        }
        result = cluster_bootstrap(
            target, probabilities, np.asarray([1, 1, 2, 2]), samples=20, seed=1
        )
        self.assertEqual(result["valid_samples"], 20)
        self.assertIn("temporal_mlp", result["delta_vs_stage_only"])


if __name__ == "__main__":
    unittest.main()
