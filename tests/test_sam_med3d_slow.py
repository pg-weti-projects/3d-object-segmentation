"""SAM-Med3D with the real weights, on a small synthetic scan. Excluded from the default run:
    python -m pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.volume import Volume
from src.models.foundation.sam_med3d import assets, cube as geometry, network, segment

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def installed():
    if assets.missing():
        pytest.skip(f"SAM-Med3D is not installed: missing {assets.missing()}")


def scan():
    data = np.full((48, 48, 48), -1000.0, dtype=np.float32)
    data[8:40, 8:40, 8:40] = 40.0
    data[18:30, 18:30, 18:30] = 300.0
    return Volume(data, (3.0, 3.0, 3.0), "HU", "synthetic", "0" * 64)


def test_the_real_network_gives_a_cube_of_probabilities():
    net = network.get_network()
    cube_xyz = np.random.default_rng(0).normal(size=(geometry.SIZE,) * 3).astype(np.float32)
    probability = net.predict(net.embed(cube_xyz), (64, 64, 64))
    assert probability.shape == (geometry.SIZE,) * 3
    assert probability.min() >= 0 and probability.max() <= 1


def test_the_same_input_gives_the_same_mask():
    first = segment(scan(), {"cube": "click", "seed_zyx": [24, 24, 24]})
    second = segment(scan(), {"cube": "click", "seed_zyx": [24, 24, 24]})
    assert second.report["encoder_output_reused"] is True
    assert np.array_equal(first, second)


def test_a_box_cube_reuses_the_encoder_for_another_click_and_gives_a_valid_mask():
    box = {"min_mm": [15.0, 15.0, 15.0], "max_mm": [129.0, 129.0, 129.0]}
    first = segment(scan(), {"cube": "box", "box": box, "seed_zyx": [24, 24, 24]})
    second = segment(scan(), {"cube": "box", "box": box, "seed_zyx": [20, 26, 22]})
    assert first.report["encoder_output_reused"] is False and second.report["encoder_output_reused"] is True
    assert np.asarray(first).shape == (48, 48, 48)
