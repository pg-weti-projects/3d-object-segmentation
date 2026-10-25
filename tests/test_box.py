import numpy as np
import pytest

from src.core.preprocessor import PreprocessingConfig, preprocess
from src.core.box import Box, BoxError, crop_volume, restore_mask
from src.core.volume import Volume, hash_bytes


def _volume(shape=(10, 20, 30), spacing=(2.0, 1.0, 0.5)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return Volume(data, spacing, "HU", "test", hash_bytes(b"box"))


def test_box_from_voxels_selects_exactly_those_voxels():
    volume = _volume()
    box = Box.from_voxels((2, 5, 10), (6, 12, 25), volume.spacing_mm)

    cropped, where = crop_volume(volume, box)

    assert where == (slice(2, 6), slice(5, 12), slice(10, 25))
    assert cropped.shape == (4, 7, 15)
    np.testing.assert_array_equal(cropped.data, volume.data[2:6, 5:12, 10:25])


def test_crop_keeps_spacing_units_and_hash():
    volume = _volume()

    cropped, _ = crop_volume(volume, Box.from_voxels((0, 0, 0), (3, 3, 3), volume.spacing_mm))

    assert cropped.spacing_mm == volume.spacing_mm
    assert cropped.units == "HU" and cropped.source_hash == volume.source_hash


def test_box_in_mm_is_independent_of_the_grid():
    volume = _volume((20, 20, 20), (1.0, 1.0, 1.0))
    box = Box((4.0, 6.0, 8.0), (10.0, 14.0, 16.0))  # mm
    resampled = preprocess(volume, PreprocessingConfig(target_spacing=(0.5, 0.5, 0.5)))

    _, original = crop_volume(volume, box)
    _, finer = crop_volume(resampled, box)

    assert original == (slice(4, 10), slice(6, 14), slice(8, 16))
    assert finer == (slice(8, 20), slice(12, 28), slice(16, 32))  # twice as many voxels


def test_box_that_is_not_aligned_covers_every_overlapped_voxel():
    box = Box((2.5, 0.0, 0.0), (4.5, 1.0, 1.0))

    where = box.slices((10, 10, 10), (1.0, 1.0, 1.0))

    assert where[0] == slice(2, 5)  # 2.5-4.5 mm overlaps voxels 2, 3 and 4


def test_box_is_clipped_to_the_volume():
    where = Box((0.0, 0.0, 0.0), (500.0, 500.0, 500.0)).slices((4, 5, 6), (1.0, 1.0, 1.0))

    assert where == (slice(0, 4), slice(0, 5), slice(0, 6))


def test_box_outside_the_volume_raises():
    with pytest.raises(BoxError, match="outside"):
        Box((100.0, 0.0, 0.0), (110.0, 5.0, 5.0)).slices((10, 10, 10), (1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "low, high",
    [
        ((0, 0, 0), (0, 5, 5)),  # empty on z
        ((5, 5, 5), (2, 9, 9)),  # inverted
        ((-1, 0, 0), (5, 5, 5)),  # negative
        ((0, 0), (5, 5, 5)),  # wrong length
        ((0, 0, "a"), (5, 5, 5)),  # not a number
        ((0, 0, float("nan")), (5, 5, 5)),
    ],
)
def test_invalid_boxes_raise(low, high):
    with pytest.raises(BoxError):
        Box(low, high)


def test_mapping_roundtrip_and_validation():
    box = Box((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    assert Box.from_mapping(box.to_mapping()) == box
    for bad in [None, {}, {"min_mm": [0, 0, 0]}, {"min_mm": [0, 0, 0], "max_mm": [1, 1, 1], "x": 1}]:
        with pytest.raises(BoxError):
            Box.from_mapping(bad)


def test_restore_mask_puts_the_result_back_on_the_full_grid():
    volume = _volume((8, 8, 8), (1.0, 1.0, 1.0))
    box = Box.from_voxels((2, 3, 4), (5, 7, 8), volume.spacing_mm)
    cropped, where = crop_volume(volume, box)
    local_mask = np.zeros(cropped.shape, dtype=bool)
    local_mask[1, 1, 1] = True

    full = restore_mask(local_mask, volume.shape, where)

    assert full.shape == volume.shape and full.dtype == np.uint8
    assert full.sum() == 1 and full[3, 4, 5] == 1
    assert not full[where][~local_mask].any()


def test_restore_mask_rejects_a_mask_of_the_wrong_shape():
    with pytest.raises(BoxError, match="shape"):
        restore_mask(np.zeros((2, 2, 2)), (8, 8, 8), (slice(0, 3), slice(0, 3), slice(0, 3)))


def test_algorithm_only_sees_the_box_and_output_matches_the_full_grid():
    """The whole semi-automatic flow with a trivial 'algorithm'."""
    data = np.zeros((20, 30, 30), dtype=np.float32)
    data[8:12, 10:20, 12:22] = 100.0  # the object
    data[0:3, 0:3, 0:3] = 100.0  # a distractor far from the box
    volume = Volume(data, (1.0, 1.0, 1.0), "HU", "test", hash_bytes(b"flow"))
    box = Box.from_voxels((6, 8, 10), (14, 22, 24), volume.spacing_mm)

    cropped, where = crop_volume(volume, box)
    local_mask = cropped.data > 50  # stands in for segment(cropped, parameters)
    mask = restore_mask(local_mask, volume.shape, where)

    assert cropped.shape == (8, 14, 14)
    assert mask.shape == volume.shape
    assert mask[8:12, 10:20, 12:22].all()  # the object is found
    assert not mask[0:3, 0:3, 0:3].any()  # the distractor is excluded by the box
