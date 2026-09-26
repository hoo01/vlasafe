import json
from pathlib import Path

from scripts.build_stage_cohort_manifest import expected_grid


def test_v05_confirmation_grid_is_complete_and_independent_from_pilot_states() -> None:
    protocol = json.loads(
        Path("docs/manifests/v05_task4_pickup_stall_confirmation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    grid = expected_grid(protocol)
    assert len(grid) == 180
    assert {state for state, _ in grid} == set(range(30))
    assert {seed for state, seed in grid if state == 0} == {
        8000,
        9000,
        10000,
        11000,
        12000,
        13000,
    }
    assert all(state < 30 for state, _ in grid)


def test_v05_confirmation_has_no_model_or_threshold_selection() -> None:
    protocol = json.loads(
        Path("docs/manifests/v05_task4_pickup_stall_confirmation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["frozen_before_collection"] is True
    assert protocol["evaluation"]["model_selection"].startswith("none")
    assert protocol["evaluation"]["threshold_selection"].startswith("none")
    assert protocol["formal_gate"]["shortfall_policy"].startswith("If the fixed 180")
