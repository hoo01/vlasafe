import unittest

import numpy as np

from scripts.analyze_stage_aligned_signal import (
    cross_fitted_controls,
    paired_accuracy,
    select_target,
    stage_features,
)


def label(bowl1, bowl2, plate=(1.0, 1.0, 0.0), drawer=(0.0, 0.0, 0.0)):
    def pose(value):
        return {"position": list(value), "quaternion": [1.0, 0.0, 0.0, 0.0]}
    return {
        "scene_body_poses": {
            "akita_black_bowl_1_main": pose(bowl1),
            "akita_black_bowl_2_main": pose(bowl2),
            "plate_1_main": pose(plate),
            "wooden_cabinet_1_cabinet_top": pose(drawer),
        },
        "scene_site_positions": {"wooden_cabinet_1_top_region": [0.0, 0.0, 0.0]},
    }


def row(step, scene_label, eef=(0.0, 0.0, 0.0), gripper=(0.0, 0.0)):
    return {
        "step_id": step,
        "label_only": scene_label,
        "proprioception": {"eef.pos": list(eef), "gripper.qpos": list(gripper)},
    }


class StageAlignedSignalTest(unittest.TestCase):
    def test_selects_bowl_closest_to_top_drawer_region(self) -> None:
        self.assertEqual(
            select_target(label((0.01, 0.01, 0.0), (0.5, 0.5, 0.0))),
            "akita_black_bowl_1_main",
        )

    def test_checkpoint_uses_previous_post_action_scene_state(self) -> None:
        rows = [
            row(0, label((0.0, 0.0, 0.0), (2.0, 2.0, 0.0))),
            row(1, label((0.0, 0.0, 0.1), (2.0, 2.0, 0.0))),
            row(2, label((9.0, 9.0, 9.0), (2.0, 2.0, 0.0)), eef=(0.0, 0.0, 0.1)),
        ]
        features, _ = stage_features(rows, 2)
        self.assertAlmostEqual(features[2], 0.1)
        self.assertAlmostEqual(features[3], 0.1)

    def test_cross_fit_leaves_each_group_out(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, 0, 1, 1])
        risk = np.asarray([0.1, 0.2, 0.8, 0.9])
        groups = np.asarray([0, 0, 1, 1])
        stage, residual = cross_fitted_controls(features, labels, risk, groups, 1e-3)
        self.assertEqual(stage.shape, (4,))
        self.assertEqual(residual.shape, (4,))
        self.assertTrue(np.isfinite(stage).all())
        self.assertTrue(np.isfinite(residual).all())

    def test_paired_accuracy_resamples_whole_initial_states(self) -> None:
        pairs = [
            {"initial_state_id": 30, "raw_risk_correct": True},
            {"initial_state_id": 30, "raw_risk_correct": False},
            {"initial_state_id": 31, "raw_risk_correct": True},
        ]
        result = paired_accuracy(pairs, "raw_risk_correct", samples=100, seed=1)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["pairs"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
