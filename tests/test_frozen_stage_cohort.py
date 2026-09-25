import unittest

import numpy as np

from vlasafe.cluster_bootstrap import cluster_bootstrap


class FrozenStageCohortTest(unittest.TestCase):
    def test_cluster_bootstrap_reports_point_metrics(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        probability = np.asarray([0.1, 0.9, 0.2, 0.8])
        groups = np.asarray([30, 30, 31, 31])
        result = cluster_bootstrap(labels, probability, groups, samples=100, seed=7)
        self.assertEqual(result["auroc"]["point"], 1.0)
        self.assertEqual(result["auprc"]["point"], 1.0)
        self.assertEqual(result["auroc"]["valid_samples"], 100)

    def test_cluster_bootstrap_keeps_whole_groups(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.8, 0.9])
        groups = np.asarray([30, 30, 31, 31])
        result = cluster_bootstrap(labels, probability, groups, samples=100, seed=9)
        self.assertLess(result["auroc"]["valid_samples"], 100)


if __name__ == "__main__":
    unittest.main()
