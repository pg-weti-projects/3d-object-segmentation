"""Fine-tuning SAM-Med3D: samples, synthetic bladders, the cache and the loss, without the real network.
The training loop with the real weights is in test_sam_med3d_slow.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from scripts import finetune_sam_med3d as script
from src.core.volume import Volume
from src.models.foundation.sam_med3d import cube as geometry, finetune

SIZE = geometry.SIZE


def fish(shape=(40, 48, 44), spacing=(2.0, 1.5, 1.5)) -> finetune.Specimen:
    """A dark bladder in tissue in air, HU-like, with a little texture."""
    z, y, x = (np.mgrid[: shape[0], : shape[1], : shape[2]] + 0.5) * np.asarray(spacing)[:, None, None, None]
    centre = np.asarray(shape) * np.asarray(spacing) / 2
    body = ((z - centre[0]) / 36) ** 2 + ((y - centre[1]) / 30) ** 2 + ((x - centre[2]) / 28) ** 2 <= 1
    bladder = ((z - centre[0] - 6) / 14) ** 2 + ((y - centre[1]) / 7) ** 2 + ((x - centre[2]) / 6) ** 2 <= 1
    data = np.where(body, 100.0, -1000.0) + np.random.default_rng(0).normal(0, 5, shape)
    data[bladder] = -500.0
    return finetune.Specimen("test-fish", Volume(data.astype(np.float32), spacing, "HU", "synthetic", "0" * 64), bladder)


def still(**changes) -> finetune.Augmentation:
    return finetune.Augmentation(**changes)


# --- sampling -------------------------------------------------------------------------------------


def test_without_rotation_or_scale_the_cube_is_sampled_as_the_method_samples_it():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 34, 38)).astype(np.float32)
    spacing = (2.0, 1.0, 1.5)
    cube = geometry.Cube((-20.0, 5.0, 3.0), 0.6)  # partly outside the scan
    expected, expected_inside = geometry.sample(data, spacing, cube)
    values, inside = finetune.sample_points(data, (0, 0, 0), spacing, finetune.cube_coordinates_mm(cube), 1, data.shape)
    assert np.array_equal(inside, expected_inside) and inside.any() and not inside.all()
    assert np.allclose(values[inside], expected[inside], atol=1e-5)


def test_a_region_of_the_scan_samples_as_the_whole_scan():
    data = np.random.default_rng(2).normal(size=(30, 34, 38)).astype(np.float32)
    spacing = (2.0, 1.0, 1.5)
    points = finetune.cube_coordinates_mm(geometry.Cube((10.0, 8.0, 9.0), 0.15), angle_deg=7, scale=1.05)
    region = finetune.region_for(points, data.shape, spacing, (0, 0, 0))
    whole, _ = finetune.sample_points(data, (0, 0, 0), spacing, points, 1, data.shape)
    part, _ = finetune.sample_points(data[region], [w.start for w in region], spacing, points, 1, data.shape)
    assert np.allclose(whole, part, atol=1e-6)


def test_a_quarter_turn_turns_scan_and_target_together():
    specimen = fish()
    box = ([30.0, 18.0, 18.0], [60.0, 54.0, 48.0])
    plain = finetune.make_sample(specimen, np.random.default_rng(0), augmentation=still(), box_mm=box)
    turned = finetune.make_sample(specimen, np.random.default_rng(0), augmentation=still(quarter_turns=1), box_mm=box)
    assert np.allclose(turned["cube"], np.rot90(plain["cube"], 1, axes=(1, 2)), atol=1e-5)
    assert np.array_equal(turned["target"], np.rot90(plain["target"], 1, axes=(1, 2)))


def test_the_target_is_the_dark_bladder_and_only_inside_the_box():
    specimen = fish()
    whole = finetune.make_sample(specimen, np.random.default_rng(0), augmentation=still(), box_mm=([20.0, 10.0, 10.0], [70.0, 62.0, 56.0]))
    cube, target = whole["cube"], whole["target"]
    assert target.any() and cube[target].mean() < cube[~target].mean() - 0.5  # dark, as the scan
    half = finetune.make_sample(specimen, np.random.default_rng(0), augmentation=still(), box_mm=([20.0, 10.0, 10.0], [46.0, 62.0, 56.0]))
    assert 0 < half["meta"]["target_voxels"] < whole["meta"]["target_voxels"]


def test_jittered_boxes_follow_the_recipe():
    specimen = fish()
    spacing, shape = specimen.volume.spacing_mm, specimen.volume.shape
    low0, high0 = (np.asarray(v) * spacing for v in finetune.bounding_box(specimen.mask))
    long_boxes = 0
    for seed in range(200):
        low, high, meta = finetune.jittered_box_mm(specimen.mask, spacing, shape, np.random.default_rng(seed))
        assert (low >= 0).all() and (high <= np.asarray(shape) * spacing + 1e-9).all() and (high > low).all()
        long_boxes += meta["long_box"]
        assert all(0.10 <= g <= 1.00 for g in meta["growth"])
        assert np.all(np.abs(meta["shift_mm"]) <= 0.10 * (high0 - low0) * (1 + np.asarray(meta["growth"])) + 1e-9)
    assert 25 <= long_boxes <= 75  # a quarter of 200, give or take


def test_a_sample_is_the_same_whenever_it_is_made():
    specimen = fish()
    first = finetune.make_sample(specimen, finetune.sample_seed(finetune.RECIPE, "test-fish", "real", 3))
    again = finetune.make_sample(specimen, finetune.sample_seed(finetune.RECIPE, "test-fish", "real", 3))
    other = finetune.make_sample(specimen, finetune.sample_seed(finetune.RECIPE, "test-fish", "real", 4))
    assert np.array_equal(first["cube"], again["cube"]) and not np.array_equal(first["cube"], other["cube"])


def test_a_click_is_a_target_voxel_in_the_networks_axes():
    target = np.zeros((SIZE,) * 3, bool)
    target[10, 20, 30] = True
    assert finetune.click_from_target(target, np.random.default_rng(0)) == (30, 20, 10)
    with pytest.raises(ValueError, match="no target voxel"):
        finetune.click_from_target(np.zeros_like(target), np.random.default_rng(0))


# --- synthetic bladders ---------------------------------------------------------------------------


def test_a_synthetic_bladder_lies_in_tissue_apart_from_the_real_one_with_its_values():
    specimen = fish(shape=(60, 64, 60))
    made = [finetune.synthetic_specimen(specimen, np.random.default_rng(seed)) for seed in range(6)]
    made = [m for m in made if m is not None]
    assert made, "no synthetic bladder could be placed"
    clearance = finetune.RECIPE["synthetic_clearance_voxels"]
    for synthetic in made:
        new = synthetic.mask
        assert new.any() and not (new & ndi.binary_dilation(specimen.mask, iterations=clearance)).any()
        assert specimen.tissue(clearance)[new].all()
        assert abs(float(np.median(synthetic.volume.data[ndi.binary_erosion(new)])) + 500) < 60  # the bladder's values
        assert np.array_equal(synthetic.volume.data[specimen.mask], specimen.volume.data[specimen.mask])  # the real one untouched
    assert not np.array_equal(made[0].volume.data, specimen.volume.data)


def test_a_deformed_shape_keeps_roughly_the_bladders_size_and_has_no_holes():
    specimen = fish()
    shape, meta = finetune.deformed_shape(specimen.mask, specimen.volume.spacing_mm, np.random.default_rng(3))
    ratio = shape.sum() / specimen.mask.sum()
    assert 0.7**3 * 0.8 < ratio < 1.3**3 * 1.2
    assert np.array_equal(ndi.binary_fill_holes(shape), shape)


def test_a_synthetic_sample_clicks_in_the_synthetic_bladder():
    specimen = fish(shape=(60, 64, 60))
    sample = next(s for s in (finetune.make_synthetic_sample(specimen, np.random.default_rng(seed)) for seed in range(10)) if s is not None)
    assert sample["click_target"].any() and not (sample["click_target"] & ~sample["target"]).any()
    assert "synthetic" in sample["meta"]


# --- the cache --------------------------------------------------------------------------------------


class CountingNetwork:
    def __init__(self):
        self.calls = 0

    def embed(self, cube_xyz):
        assert cube_xyz.shape == (SIZE,) * 3
        self.calls += 1
        return np.full((1, 384, 8, 8, 8), self.calls, np.float32)


def test_encoding_saves_each_cube_and_goes_on_where_it_stopped(tmp_path):
    specimen, net = fish(), CountingNetwork()
    files = finetune.encode(specimen, "real", 3, net, tmp_path)
    assert len(files) == 3 and net.calls == 3
    files[1].unlink()
    again = finetune.encode(specimen, "real", 3, net, tmp_path)
    assert again == files and net.calls == 4
    sample = finetune.load_sample(files[0])
    assert sample["embedding"].shape == (1, 384, 8, 8, 8) and sample["target"].shape == (SIZE,) * 3
    assert np.array_equal(sample["target"], sample["click_target"]) and sample["meta"]["specimen"] == "test-fish"
    assert finetune.specimen_files(tmp_path, ["test-fish"], ["real"]) == files


def test_another_recipe_never_reuses_the_cache(tmp_path):
    other = {**finetune.RECIPE, "steps": 1}
    assert finetune.cache_folder(tmp_path, "a", "real") != finetune.cache_folder(tmp_path, "a", "real", other)


# --- loss and verdict -------------------------------------------------------------------------------


def test_the_loss_is_dice_plus_cross_entropy():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 4, 4, 4)
    target[..., :2] = 1
    assert float(finetune.dice_ce_loss(target * 40 - 20, target)) < 1e-3
    zero = torch.zeros_like(target)
    expected_dice = 1 - (2 * 0.5 * 32 + 1e-5) / (0.25 * 64 + 32 + 1e-5)
    assert float(finetune.dice_ce_loss(zero, target)) == pytest.approx(expected_dice + np.log(2), rel=1e-5)


def test_the_verdict_is_the_protocols_section_10():
    usable = script.verdict([0.8] * 7 + [0.3, 0.2], [5.0] * 9, 7.2)
    assert usable["usable"] and usable["clicks_at_least_0_5"] == 7
    fragile = script.verdict([0.8] * 6 + [0.3] * 3, [5.0] * 9, 7.2)
    assert fragile["overlap_ok"] and not fragile["robustness_ok"] and not fragile["usable"]
    empty = script.verdict([0.9] * 9, [float("inf")] * 9, 7.2)
    assert not empty["boundary_ok"]


def cached(root, name, kind, count):
    folder = finetune.cache_folder(root, name, kind)
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (folder / f"{index:04d}.npz").write_bytes(b"")
    return folder


def test_arm_b_takes_a_third_synthetic_split_over_the_fish_that_have_them(tmp_path):
    cached(tmp_path, "a", "real", 4)
    cached(tmp_path, "b", "real", 4)
    cached(tmp_path, "b", "synthetic", 6)
    real_only, counts = finetune.training_files(tmp_path, ["a", "b"], synthetic=False)
    assert len(real_only) == 8 and counts == {"a/real": 4, "b/real": 4}
    files, counts = finetune.training_files(tmp_path, ["a", "b"], synthetic=True)
    assert len(files) == 12 and counts["b/synthetic"] == 4 and "a/synthetic" not in counts  # a has no room for one
    cached(tmp_path, "a", "synthetic", 3)
    files, counts = finetune.training_files(tmp_path, ["a", "b"], synthetic=True)
    assert counts["a/synthetic"] == 2 and counts["b/synthetic"] == 2 and len(files) == 12
    cached(tmp_path, "a", "real", 20)  # now 12 synthetic are wanted, 6 from each, and a has 3
    with pytest.raises(ValueError, match="encode more"):
        finetune.training_files(tmp_path, ["a", "b"], synthetic=True)


def test_synthetic_tries_without_a_place_are_remembered_and_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(finetune, "make_synthetic_sample", lambda specimen, rng, recipe: calls.append(1))
    files = finetune.encode(fish(), "synthetic", 2, CountingNetwork(), tmp_path)
    assert files == [] and len(calls) == 2 * finetune.SYNTHETIC_ATTEMPTS
    assert finetune.encode(fish(), "synthetic", 2, CountingNetwork(), tmp_path) == [] and len(calls) == 2 * finetune.SYNTHETIC_ATTEMPTS
