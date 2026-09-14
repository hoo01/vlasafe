from __future__ import annotations

import unittest

from vlasafe.monitors.command_effect import (
    consistency_errors,
    episode_consistency_score,
    first_alarm_step,
)


def rows(effects: list[float]) -> list[dict]:
    result = []
    position = 0.0
    for effect in effects:
        result.append({
            "proprioception": {"eef.pos": [position, 0.0, 0.0]},
            "deploy_metadata": {"intended_action": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        })
        position += effect
    result.append({
        "proprioception": {"eef.pos": [position, 0.0, 0.0]},
        "deploy_metadata": {"intended_action": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    })
    return result


class CommandEffectTest(unittest.TestCase):
    def test_aligned_motion_has_zero_error(self) -> None:
        errors = consistency_errors(rows([0.1, 0.1, 0.1]))
        self.assertAlmostEqual(episode_consistency_score(errors), 0.0)
        self.assertIsNone(first_alarm_step(errors, threshold=0.5))

    def test_reversed_motion_triggers_consecutive_alarm(self) -> None:
        errors = consistency_errors(rows([-0.1, -0.1, -0.1]))
        self.assertAlmostEqual(episode_consistency_score(errors), 2.0)
        self.assertEqual(first_alarm_step(errors, threshold=1.0), 0)

    def test_nonconsecutive_active_steps_do_not_form_window(self) -> None:
        errors = consistency_errors(rows([-0.1, -0.1, -0.1, -0.1]))
        errors[1]["step"] = 5
        self.assertIsNone(first_alarm_step(errors, threshold=1.0, consecutive=3))

    def test_small_commands_are_ignored(self) -> None:
        data = rows([0.1])
        data[0]["deploy_metadata"]["intended_action"][0] = 1e-5
        self.assertEqual(consistency_errors(data), [])


if __name__ == "__main__":
    unittest.main()
