import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.evaluate_frozen_pickup_stall_confirmation import temporal_features


def test_temporal_features_apply_frozen_normalization() -> None:
    data = {
        "state": np.ones((2, 2, 25), dtype=np.float32),
        "action": np.ones((2, 2, 7), dtype=np.float32),
        "timing": np.ones((2, 2, 2), dtype=np.float32),
        "mask": np.asarray([[1, 1], [0, 1]], dtype=np.float32),
    }
    normalization = {
        "channel_mean": [1.0] * 34,
        "channel_std": [2.0] * 34,
        "feature_dim": 70,
    }
    features = temporal_features(data, normalization)
    assert features.shape == (2, 70)
    assert np.all(features[:, :68] == 0)
    assert np.array_equal(features[:, 68:], data["mask"])
