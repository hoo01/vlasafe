from scripts.build_pickup_stall_confirmation import build_samples


def test_build_samples_empty() -> None:
    samples, exclusions = build_samples([])
    assert samples == []
    assert exclusions == []
