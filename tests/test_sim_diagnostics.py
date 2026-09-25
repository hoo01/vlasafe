from __future__ import annotations

import unittest

from vlasafe.rollout.sim_diagnostics import (
    EVENT_SCHEMA_VERSION,
    _gripper_contact_geoms,
    _is_unsafe_self_contact,
    _model_contact_geoms,
    _scene_body_names,
    _scene_site_names,
)


class Model:
    def __init__(self, geoms):
        self.contact_geoms = geoms


class SceneModel:
    body_names = (
        "world", "table", "robot0_link0", "gripper0_eef", "mount0_base",
        "akita_black_bowl_1_main", "plate_1_main", "wooden_cabinet_1_main",
    )
    site_names = (
        "table_top", "main_table_plate_region", "gripper0_grip_site",
        "akita_black_bowl_1_default_site", "plate_1_default_site",
        "wooden_cabinet_1_top_region",
    )


class SimDiagnosticsHelpersTest(unittest.TestCase):
    def test_event_schema_version_tracks_self_collision_semantics(self) -> None:
        self.assertEqual(EVENT_SCHEMA_VERSION, "0.3.0")

    def test_internal_gripper_contact_is_not_self_collision(self) -> None:
        gripper = {"finger1", "finger2"}
        self.assertFalse(_is_unsafe_self_contact("finger1", "finger2", gripper))
        self.assertTrue(_is_unsafe_self_contact("arm", "finger1", gripper))
        self.assertTrue(_is_unsafe_self_contact("arm1", "arm2", gripper))

    def test_collects_model_contact_geoms(self) -> None:
        self.assertEqual(_model_contact_geoms(Model(["a", "b"])), {"a", "b"})

    def test_collects_dict_gripper_geoms(self) -> None:
        gripper = {"left": Model(["a"]), "right": Model(["b", "c"])}
        self.assertEqual(_gripper_contact_geoms(gripper), {"a", "b", "c"})

    def test_missing_gripper_has_no_geoms(self) -> None:
        self.assertEqual(_gripper_contact_geoms(None), set())

    def test_scene_names_exclude_robot_and_table_entries(self) -> None:
        model = SceneModel()
        self.assertEqual(
            _scene_body_names(model),
            ("akita_black_bowl_1_main", "plate_1_main", "wooden_cabinet_1_main"),
        )
        self.assertEqual(
            _scene_site_names(model),
            (
                "akita_black_bowl_1_default_site",
                "plate_1_default_site",
                "wooden_cabinet_1_top_region",
            ),
        )


if __name__ == "__main__":
    unittest.main()
