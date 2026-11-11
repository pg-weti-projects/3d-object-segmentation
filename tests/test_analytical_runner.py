"""The three analytical methods, as the experiment runner sees them.

The methods themselves are covered by test_chan_vese, test_geodesic and
test_growcut; what is checked here is only the contract the runner relies on:
the name resolves to a function, the function takes ``(volume, parameters)``
and gives back a uint8 mask on the grid it was handed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.core.volume import Volume, hash_bytes
from src.models.analytical import chan_vese, geodesic_active_contour, growcut
from src.pipeline.runner import ALGORITHMS, resolve_algorithm

ANALYTICAL = ("chan_vese", "geodesic_active_contour", "growcut")
METHODS = (chan_vese, geodesic_active_contour, growcut)


def volume_with_a_dark_ball() -> tuple[Volume, tuple[int, int, int]]:
    """Bright tissue with a dark ball in the middle: a bladder as simple as it gets."""
    data = np.full((24, 24, 24), 1000.0, dtype=np.float32)
    z, y, x = np.ogrid[:24, :24, :24]
    data[((z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2) <= 36] = -900.0
    return Volume(data, (1.0, 1.0, 1.0), "HU", "test", hash_bytes(b"analytical")), (12, 12, 12)


@pytest.mark.parametrize("name", ANALYTICAL)
def test_the_runner_can_resolve_every_analytical_method(name: str) -> None:
    assert name in ALGORITHMS
    assert callable(resolve_algorithm(name))


@pytest.mark.parametrize("method", METHODS)
def test_a_method_returns_a_uint8_mask_on_the_grid_it_was_given(method) -> None:
    volume, seed = volume_with_a_dark_ball()
    mask = method(volume, {"seed_zyx": seed, "init_from": "contrast"})

    assert mask.dtype == np.uint8
    assert mask.shape == volume.data.shape
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[seed] == 1, "the ball around the seed point is what the method should find"
    assert mask.sum() < volume.data.size / 2, "the mask should not flood the volume"


@pytest.mark.parametrize("method", METHODS)
def test_a_method_reports_what_it_did(method) -> None:
    volume, seed = volume_with_a_dark_ball()
    report = method(volume, {"seed_zyx": seed, "init_from": "contrast"}).report

    assert report["status"] in {"ok", "empty"}
    assert report["seed_zyx"] == list(seed)
    import json

    json.dumps(report)  # the runner writes it to method_report.json


@pytest.mark.parametrize("method", METHODS)
def test_a_method_says_so_when_it_has_no_seed_point(method) -> None:
    volume, _ = volume_with_a_dark_ball()
    with pytest.raises(ValueError, match="seed"):
        method(volume, {})


@pytest.mark.parametrize("method", (geodesic_active_contour, growcut))
def test_the_contour_methods_refuse_an_unknown_parameter(method) -> None:
    volume, seed = volume_with_a_dark_ball()
    with pytest.raises(ValueError):
        method(volume, {"seed_zyx": seed, "not_a_parameter": 1})


def test_chan_vese_only_warns_about_an_unknown_parameter(caplog) -> None:
    """Chan-Vese ignores a key it does not know and logs it, where the other two
    refuse it. Worth knowing when reading a config: a typo there changes nothing
    and the run still reports a number."""
    volume, seed = volume_with_a_dark_ball()
    with caplog.at_level(logging.WARNING):
        chan_vese(volume, {"seed_zyx": seed, "not_a_parameter": 1})
    assert "not_a_parameter" in caplog.text
