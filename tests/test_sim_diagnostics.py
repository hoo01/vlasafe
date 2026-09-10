from __future__ import annotations

import unittest

from vlasafe.rollout.sim_diagnostics import _gripper_contact_geoms, _model_contact_geoms


class Model:
    def __init__(self, geoms):
        self.contact_geoms = geoms


class SimDiagnosticsHelpersTest(unittest.TestCase):
    def test_collects_model_contact_geoms(self) -> None:
        self.assertEqual(_model_contact_geoms(Model(["a", "b"])), {"a", "b"})

    def test_collects_dict_gripper_geoms(self) -> None:
        gripper = {"left": Model(["a"]), "right": Model(["b", "c"])}
        self.assertEqual(_gripper_contact_geoms(gripper), {"a", "b", "c"})

    def test_missing_gripper_has_no_geoms(self) -> None:
        self.assertEqual(_gripper_contact_geoms(None), set())


if __name__ == "__main__":
    unittest.main()
