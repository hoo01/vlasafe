from __future__ import annotations

import unittest

from scripts.render_failure_event_review import choose_review_rows


class FailureEventReviewTest(unittest.TestCase):
    def test_keeps_success_trigger_and_spans_failure_states(self) -> None:
        rows = [
            {"episode_id": "success", "initial_state_id": 9, "failure": 0, "event_step": 80},
            {"episode_id": "a1", "initial_state_id": 1, "failure": 1, "event_step": 70},
            {"episode_id": "a2", "initial_state_id": 1, "failure": 1, "event_step": 90},
            {"episode_id": "b", "initial_state_id": 2, "failure": 1, "event_step": 100},
        ]
        chosen = choose_review_rows(rows, 3)
        self.assertEqual(chosen[0]["episode_id"], "success")
        self.assertEqual({row["initial_state_id"] for row in chosen[1:]}, {1, 2})


if __name__ == "__main__":
    unittest.main()
