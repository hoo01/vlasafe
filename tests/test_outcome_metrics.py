from __future__ import annotations

import unittest

import numpy as np

from vlasafe.outcome_metrics import binary_metrics, fit_logistic_l2, predict_logistic


class OutcomeMetricsTest(unittest.TestCase):
    def test_constant_score_matches_prevalence_and_random_auroc(self) -> None:
        target = np.asarray([0, 0, 1, 1])
        result = binary_metrics(target, np.full(4, 0.5))
        self.assertAlmostEqual(result["auprc"], 0.5)
        self.assertAlmostEqual(result["auroc"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.25)
        self.assertAlmostEqual(result["ece"], 0.0)

    def test_perfect_ranking_has_unit_auc(self) -> None:
        result = binary_metrics(
            np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9])
        )
        self.assertAlmostEqual(result["auprc"], 1.0)
        self.assertAlmostEqual(result["auroc"], 1.0)

    def test_logistic_fit_learns_simple_separation(self) -> None:
        features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        target = np.asarray([0, 0, 1, 1])
        weights, bias = fit_logistic_l2(features, target, l2=0.01, steps=1000)
        probability = predict_logistic(features, weights, bias)
        self.assertGreater(probability[2], probability[1])
        self.assertAlmostEqual(binary_metrics(target, probability)["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
