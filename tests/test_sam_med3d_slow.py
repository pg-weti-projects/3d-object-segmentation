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


def box_contents():
    """What the runner would pass for a box of voxels 5-42: a different cube from scan()'s."""
    whole = scan()
    return Volume(np.ascontiguousarray(whole.data[5:43, 5:43, 5:43]), whole.spacing_mm, "HU", "synthetic", "1" * 64)


def test_the_same_input_gives_the_same_mask():
    first = segment(scan(), {"seed_zyx": [24, 24, 24]})
    second = segment(scan(), {"seed_zyx": [24, 24, 24]})
    assert second.report["encoder_output_reused"] is True
    assert np.array_equal(first, second)


def test_another_click_in_the_same_box_reuses_the_encoder_and_gives_a_valid_mask():
    first = segment(box_contents(), {"seed_zyx": [19, 19, 19]})
    second = segment(box_contents(), {"seed_zyx": [15, 21, 17]})
    assert first.report["encoder_output_reused"] is False and second.report["encoder_output_reused"] is True
    assert np.asarray(first).shape == (38, 38, 38)


def test_a_tuned_set_of_the_standard_parts_gives_the_standard_result(tmp_path, monkeypatch):
    """The loading path of tuned weights, end to end: saved, checked, loaded over the standard weights."""
    import json

    import torch

    standard = network.get_network()
    folder = tmp_path / "same-as-standard"
    folder.mkdir()
    torch.save(network.tuned_state(standard.network.model), folder / assets.TUNED_FILE)
    digest = assets.sha256_file(folder / assets.TUNED_FILE)
    manifest = {"base_weights_sha256": assets.WEIGHTS_SHA256, "sha256": digest}
    (folder / assets.TUNED_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assets, "tuned_folder", lambda name: tmp_path / name)
    monkeypatch.setattr(network, "_loaded", dict(network._loaded))  # the tuned copy is dropped afterwards

    tuned = network.get_network(weights="same-as-standard")
    assert tuned is not standard and tuned.weights_sha256 == digest and standard.weights_sha256 == assets.WEIGHTS_SHA256
    cube_xyz = np.random.default_rng(1).normal(size=(geometry.SIZE,) * 3).astype(np.float32)
    assert np.array_equal(standard.predict(standard.embed(cube_xyz), (60, 64, 70)), tuned.predict(tuned.embed(cube_xyz), (60, 64, 70)))
