import unittest

from scripts.build_stage_cohort_manifest import expected_grid, validate_declared_grid


class StageCohortManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "sampling": {
                "preset_initial_state_ids": {"start_inclusive": 30, "stop_exclusive": 32},
                "seed_starts": [3000, 4000],
                "total_episodes": 4,
            }
        }

    def test_expected_grid_offsets_seed_by_state(self) -> None:
        self.assertEqual(
            expected_grid(self.protocol),
            {(30, 3000), (31, 3001), (30, 4000), (31, 4001)},
        )

    def test_declared_grid_accepts_complete_collection(self) -> None:
        episodes = [
            {"initial_state_id": state, "seed": seed}
            for state, seed in expected_grid(self.protocol)
        ]
        validate_declared_grid(self.protocol, episodes)

    def test_declared_grid_rejects_duplicate_pair(self) -> None:
        episodes = [
            {"initial_state_id": 30, "seed": 3000},
            {"initial_state_id": 30, "seed": 3000},
            {"initial_state_id": 30, "seed": 4000},
            {"initial_state_id": 31, "seed": 4001},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_declared_grid(self.protocol, episodes)


if __name__ == "__main__":
    unittest.main()
