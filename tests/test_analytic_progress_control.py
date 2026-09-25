import unittest

import numpy as np

from scripts.analyze_analytic_progress_control import (
    FEATURE_NAMES,
    analytic_features,
    fit_linear,
    predict_linear,
)


def row(eef: list[float], gripper: list[float]) -> dict:
    return {"proprioception": {"eef.pos": eef, "gripper.qpos": gripper}}


class AnalyticProgressControlTest(unittest.TestCase):
    def test_features_are_causal_and_have_expected_values(self) -> None:
        rows = [
            row([0.0, 0.0, 1.0], [0.1, 0.3]),
            row([3.0, 0.0, 2.0], [0.2, 0.4]),
            row([3.0, 4.0, 0.0], [0.5, 0.7]),
            row([99.0, 99.0, 99.0], [9.0, 9.0]),
        ]
        values = analytic_features(rows, 2)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        np.testing.assert_allclose(values[:3], [3.0, 4.0, -1.0])
        self.assertAlmostEqual(values[3], np.sqrt(10.0) + np.sqrt(20.0))
        self.assertAlmostEqual(values[4], 2.0)
        self.assertAlmostEqual(values[5], 0.6)
        self.assertAlmostEqual(values[6], 0.4)

    def test_linear_fit_reconstructs_affine_target(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = 2.0 + 3.0 * x[:, 0]
        coefficients = fit_linear(x, y, ridge=0.0)
        np.testing.assert_allclose(predict_linear(x, coefficients), y)


if __name__ == "__main__":
    unittest.main()
