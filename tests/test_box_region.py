import numpy as np
import pytest

from src.core.volume import Volume, hash_bytes
from src.models.box_region import resolve_parameters, segment


def _volume(data, spacing=(1.0, 1.0, 1.0)):
    return Volume(np.asarray(data, dtype=np.float32), spacing, "HU", "test", hash_bytes(b"box"))


def _ball(shape, center, radius, spacing=(1.0, 1.0, 1.0)):
    g = np.ogrid[tuple(slice(0, n) for n in shape)]
    return sum(((a - c) * s) ** 2 for a, c, s in zip(g, center, spacing)) <= radius**2


def _dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _fish_box(shape=(30, 40, 40), center=(15, 20, 20), radius=7, gas=-900.0, tissue=40.0, noise=0.0, seed=0):
    """A body of tissue with a dark ball in it and air around the body."""
    rng = np.random.default_rng(seed)
    body = _ball(shape, (15, 20, 20), 17)
    ball = _ball(shape, center, radius)
    data = np.full(shape, -1000.0, dtype=np.float32)
    data[body] = tissue
    data[ball] = gas
    return data + rng.normal(0, noise, shape).astype(np.float32), ball


def test_gas_pocket_is_found():
    data, ball = _fish_box()

    mask = segment(_volume(data), {})

    assert mask.shape == data.shape and mask.dtype == np.uint8
    assert _dice(mask, ball) > 0.9


def test_bladder_with_little_contrast_and_noise_is_found():
    data, ball = _fish_box(gas=18400.0, tissue=19000.0, noise=150.0)
    data[data < 5000] = 3800.0  # air around the body, like in the Gadus scan

    mask = segment(_volume(data), {})

    assert _dice(mask, ball) > 0.8


def test_brighter_bladder_with_bladder_is_dark_false():
    data, ball = _fish_box(gas=300.0, tissue=40.0)

    assert segment(_volume(data), {"bladder_is_dark": False}).sum() > 0.8 * ball.sum()
    assert segment(_volume(data), {"bladder_is_dark": True}).sum() == 0  # not darker than the tissue


def test_seed_selects_one_of_two_pockets():
    data, ball = _fish_box(center=(15, 12, 12), radius=4)
    other = _ball(data.shape, (15, 28, 28), 4)
    data[other] = -900.0

    near_first = segment(_volume(data), {"seed_zyx": (15, 12, 12), "seed_radius_mm": 2})
    near_second = segment(_volume(data), {"seed_zyx": (15, 28, 28), "seed_radius_mm": 2})

    assert _dice(near_first, ball) > 0.85 and not (near_first.astype(bool) & other).any()
    assert _dice(near_second, other) > 0.85 and not (near_second.astype(bool) & ball).any()


def test_anisotropic_spacing_is_handled():
    spacing = (3.0, 1.0, 1.0)
    shape = (14, 40, 40)
    ball = _ball(shape, (7, 20, 20), 8, spacing)
    body = _ball(shape, (7, 20, 20), 17, spacing)
    data = np.full(shape, -1000.0, dtype=np.float32)
    data[body] = 40.0
    data[ball] = -900.0

    mask = segment(_volume(data, spacing), {})

    assert _dice(mask, ball) > 0.85


def test_gas_open_to_the_outside_does_not_follow_the_air():
    """A pocket in a body whose notch opens to the air outside - the hull stops the mask."""
    shape = (30, 40, 40)
    data = np.full(shape, -1000.0, dtype=np.float32)
    data[:, 8:32, 8:32] = 40.0  # a block of tissue
    pocket = _ball(shape, (15, 20, 20), 6)
    data[pocket] = -900.0
    data[10:20, 14:26, 20:32] = -1000.0  # channel from the pocket to the air at x >= 32 (and beyond)
    air_outside = np.zeros(shape, dtype=bool)
    air_outside[:, :, 32:] = True

    with_hull = segment(_volume(data), {"use_hull": True}).astype(bool)
    without_hull = segment(_volume(data), {"use_hull": False}).astype(bool)

    assert not (with_hull & air_outside).any()
    assert (without_hull & air_outside).sum() > 0  # without the hull the mask runs out of the body
    assert (with_hull & pocket).sum() > 0.8 * pocket.sum()


def test_nothing_around_the_seed_gives_an_empty_mask():
    data = np.full((20, 20, 20), 40.0, dtype=np.float32)
    data[:, :, :3] = -1000.0  # only air on one side, the centre is tissue

    mask = segment(_volume(data), {})

    assert mask.shape == data.shape and mask.sum() == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": 1},
        {"threshold_fraction": 0},
        {"threshold_fraction": 1.5},
        {"sigma_mm": -1},
        {"open_mm": "a"},
        {"seed_radius_mm": 0},
        {"seed_zyx": (1, 2)},
        {"seed_zyx": (1.5, 2, 3)},
    ],
)
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        resolve_parameters(parameters)


def test_seed_outside_and_degenerate_volumes_raise():
    data, _ = _fish_box()
    with pytest.raises(ValueError, match="outside"):
        segment(_volume(data), {"seed_zyx": (99, 0, 0)})
    with pytest.raises(ValueError, match="uniform"):
        segment(_volume(np.ones((5, 5, 5))), {})
    bad = data.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        segment(_volume(bad), {})


def test_segment_does_not_modify_the_input():
    data, _ = _fish_box()
    volume = _volume(data)
    before = volume.data.copy()

    segment(volume, {})

    np.testing.assert_array_equal(volume.data, before)
