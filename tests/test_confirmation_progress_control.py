import unittest

import numpy as np

from scripts.analyze_confirmation_progress_control import optimal_matches


class ConfirmationProgressControlTest(unittest.TestCase):
    def test_matching_supports_more_successes_than_failures(self) -> None:
        episode_ids = np.asarray(["s0", "s1", "s2", "f0", "f1"])
        outcome = np.asarray([0, 0, 0, 1, 1])
        progress = np.asarray([0.0, 1.0, 8.0, 0.1, 1.2])
        risk = np.asarray([0.1, 0.2, 0.3, 0.8, 0.7])
        matches = optimal_matches(episode_ids, outcome, progress, risk)
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {(row["success_episode"], row["failure_episode"]) for row in matches},
            {("s0", "f0"), ("s1", "f1")},
        )
        self.assertTrue(all(row["risk_orders_pair_correctly"] for row in matches))

    def test_matching_supports_more_failures_than_successes(self) -> None:
        episode_ids = np.asarray(["s0", "s1", "f0", "f1", "f2"])
        outcome = np.asarray([0, 0, 1, 1, 1])
        progress = np.asarray([0.0, 2.0, 0.2, 2.1, 9.0])
        risk = np.asarray([0.1, 0.2, 0.8, 0.7, 0.9])
        matches = optimal_matches(episode_ids, outcome, progress, risk)
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {(row["success_episode"], row["failure_episode"]) for row in matches},
            {("s0", "f0"), ("s1", "f1")},
        )


if __name__ == "__main__":
    unittest.main()
