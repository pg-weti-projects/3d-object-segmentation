"""SAM-Med3D with a fake network: no torch, no weights, in seconds. Real weights are in test_sam_med3d_slow.py."""

from __future__ import annotations

import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

from scripts import install_sam_med3d as installer
from src.core.volume import Volume
from src.models.foundation.sam_med3d import assets, cube as geometry, network, pipeline
from src.models.foundation.sam_med3d.pipeline import segment
from src.pipeline.runner import main as runner_main, run_experiment

SIZE = geometry.SIZE


class FakeNetwork:
    """Marks a small ball around the click, in the network's (x, y, z), and remembers what it saw."""

    def __init__(self):
        self.cubes = []
        self.clicks = []

    def embed(self, cube_xyz):
        assert cube_xyz.shape == (SIZE, SIZE, SIZE)
        self.cubes.append(np.array(cube_xyz))
        return len(self.cubes)

    def predict(self, embedding, click_xyz):
        self.clicks.append(tuple(click_xyz))
        x, y, z = np.ogrid[:SIZE, :SIZE, :SIZE]
        cx, cy, cz = click_xyz
        return (((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= 9).astype(np.float32)


@pytest.fixture
def fake(monkeypatch):
    net = FakeNetwork()
    cached = network.CachedEncoder(net)
    monkeypatch.setattr(network, "get_network", lambda device="cpu", threads=None: cached)
    return net


def volume(shape=(60, 70, 80), spacing=(2.0, 2.0, 2.0), fill=None):
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 20.0, shape).astype(np.float32) if fill is None else np.full(shape, fill, np.float32)
    return Volume(data, spacing, "HU", "dicom", "0" * 64)


# --- geometry ------------------------------------------------------------------------------------


def test_the_click_falls_on_the_centre_of_voxel_64():
    cube = geometry.cube_on_click((100.0, 50.0, 25.0), 1.5)
    assert geometry.network_click((100.0, 50.0, 25.0), cube) == (64, 64, 64)
    assert cube.centres_mm(0)[64] == pytest.approx(100.0)
    assert cube.side_mm == 192.0


def test_a_box_fills_the_cube_along_its_longest_side():
    cube = geometry.cube_on_box((10.0, 0.0, 0.0), (266.0, 64.0, 32.0))
    assert cube.spacing_mm == 2.0
    low, high = cube.bounds_mm()
    assert (low[0], high[0]) == (10.0, 266.0)
    assert (low[1] + high[1]) / 2 == 32.0  # centred on the box, on the shorter axes too
    assert geometry.cube_on_box((0, 0, 0), (128.0, 10.0, 10.0), margin=0.25).spacing_mm == 1.25


def test_the_axis_conversion_is_explicit_and_its_own_inverse():
    zyx = np.zeros((2, 3, 4))
    zyx[1, 2, 3] = 7
    xyz = geometry.to_network_axes(zyx)
    assert xyz.shape == (4, 3, 2) and xyz[3, 2, 1] == 7
    assert np.array_equal(geometry.to_network_axes(xyz), zyx)


def test_a_click_is_given_to_the_network_as_x_y_z():
    cube = geometry.Cube((0.0, 0.0, 0.0), 1.0)
    assert geometry.network_click((10.5, 20.5, 30.5), cube) == (30, 20, 10)  # from (z, y, x) in mm


def test_a_click_outside_the_cube_is_refused():
    with pytest.raises(ValueError, match="outside the cube"):
        geometry.network_click((500.0, 0.5, 0.5), geometry.Cube((0.0, 0.0, 0.0), 1.0))


def test_sampling_on_the_scans_own_grid_returns_its_voxels_exactly():
    data = np.arange(40**3, dtype=np.float32).reshape(40, 40, 40)
    cube = geometry.cube_on_click(geometry.voxel_centre_mm((20, 20, 20), (1.5,) * 3), 1.5)
    values, inside = geometry.sample(data, (1.5, 1.5, 1.5), cube)
    assert values[64, 64, 64] == data[20, 20, 20]
    assert values[65, 64, 63] == data[21, 20, 19]
    assert inside.sum() == 40**3


def test_sampling_interpolates_a_ramp_exactly_and_in_the_right_axis():
    z = np.arange(30, dtype=np.float32)[:, None, None] * np.ones((30, 20, 10), np.float32)
    cube = geometry.Cube((10.0, 0.0, 0.0), 0.25)  # z from 10 mm, spacing of the scan is 1 mm
    values, inside = geometry.sample(z, (1.0, 1.0, 1.0), cube)
    assert inside[0, 0, 0]
    # the centre of cube voxel k is at 10 + (k + 0.5) * 0.25 mm, which is index that - 0.5 of the scan
    assert values[0, 0, 0] == pytest.approx(10 + 0.125 - 0.5)
    assert values[10, 3, 3] == pytest.approx(10 + 10.5 * 0.25 - 0.5)


def test_the_cubes_edges_never_blend_into_padding():
    data = np.full((10, 10, 10), 100.0, np.float32)
    # voxels straddle the end of the scan at 10 mm: the last centre inside is past the last voxel centre
    cube = geometry.Cube((0.3, 0.3, 0.3), 0.3)
    values, inside = geometry.sample(data, (1.0, 1.0, 1.0), cube)
    assert np.all(values[inside] == 100.0)  # no value between the scan and anything else
    assert np.all(values[~inside] == 0.0)
    prepared, _ = pipeline.prepare(values + inside * np.arange(SIZE**3).reshape((SIZE,) * 3) % 7, inside, _params(padding=-50.0))
    assert len(np.unique(prepared[~inside])) == 1


def test_values_past_the_last_voxel_centre_hold_the_edge_value():
    data = np.zeros((4, 4, 4), np.float32)
    data[3] = 80.0
    cube = geometry.Cube((3.6, 0.0, 0.0), 0.05)  # z = 3.625 ... inside voxel 3, past its centre at 3.5
    values, inside = geometry.sample(data, (1.0, 1.0, 1.0), cube)
    edge = inside[:, 0, 0]
    assert edge.any() and not edge.all()
    assert np.all(values[edge, 0, 0] == 80.0)
    assert np.all(values[~edge, 0, 0] == 0.0)


def test_paste_back_puts_a_cube_voxel_where_it_came_from_and_nothing_outside():
    shape, spacing = (30, 40, 50), (1.5, 1.5, 1.5)
    seed = (7, 21, 33)
    cube = geometry.cube_on_click(geometry.voxel_centre_mm(seed, spacing), 1.5)
    cube_mask = np.zeros((SIZE, SIZE, SIZE), bool)
    cube_mask[64, 64, 64] = True
    pasted = geometry.paste_back(cube_mask, cube, shape, spacing)
    assert pasted.shape == shape and pasted.sum() == 1 and pasted[seed] == 1


def test_paste_back_resamples_to_the_scans_grid():
    spacing = (3.0, 0.75, 0.75)
    seed = (5, 40, 40)
    cube = geometry.cube_on_click(geometry.voxel_centre_mm(seed, spacing), 1.5)
    cube_mask = np.ones((SIZE, SIZE, SIZE), bool)
    pasted = geometry.paste_back(cube_mask, cube, (20, 100, 100), spacing)
    # everything within 96 mm is in the cube: all 20 slices (60 mm), and all of the 75 mm in-plane
    assert pasted.all()
    far = geometry.paste_back(cube_mask, cube, (20, 400, 400), spacing)
    assert far[:, :, 0].any() and not far[:, :, 399].any()


def test_coverage_of_a_small_scan_and_a_long_one():
    small = geometry.coverage(geometry.cube_on_click((10.0, 10.0, 10.0), 1.5), (20, 20, 20), (1.0, 1.0, 1.0))
    assert small["scan_share_in_cube"] == 1.0
    assert small["cube_share_in_scan"] == pytest.approx((20 / 192) ** 3)
    assert small["padding_share_of_cube"] == pytest.approx(1 - (20 / 192) ** 3)
    assert small["scan_axes_longer_than_cube"] == []
    long = geometry.coverage(geometry.cube_on_click((500.0, 50.0, 50.0), 1.5), (1000, 100, 100), (1.0, 1.0, 1.0))
    assert long["scan_share_in_cube"] == pytest.approx(0.192)
    assert long["scan_axes_longer_than_cube"] == ["z"]


def test_a_box_cut_by_the_cube_is_measured():
    report = geometry.coverage(geometry.Cube((0.0, 0.0, 0.0), 1.0), (500, 500, 500), (1.0, 1.0, 1.0), ((0, 0, 0), (256, 64, 64)))
    assert report["box_share_in_cube"] == pytest.approx(0.5)
    assert report["box_share_in_scan"] == 1.0


def test_only_faces_with_scan_beyond_them_count_as_cut():
    cube = geometry.cube_on_click((50.0, 50.0, 50.0), 1.5)
    mask = np.zeros((SIZE, SIZE, SIZE), bool)
    mask[SIZE - 1, 60, 60] = True
    assert geometry.faces_touched_inside_scan(mask, cube, (1000, 100, 100), (1.0, 1.0, 1.0)) == ["high z"]
    assert geometry.faces_touched_inside_scan(mask, cube, (100, 100, 100), (1.0, 1.0, 1.0)) == []


# --- normalisation -------------------------------------------------------------------------------


def _params(**overrides):
    return {"clamp": None, "intensity_shift": 0.0, "padding": 0.0, **overrides}


def test_the_z_score_uses_voxels_above_0_with_an_unbiased_std():
    values = np.zeros((SIZE, SIZE, SIZE), np.float32)
    values[:10] = -1000.0
    values[10:20] = 50.0
    values[20:30] = 150.0
    prepared, report = pipeline.prepare(values, np.ones(values.shape, bool), _params())
    above = values[values > 0]
    assert report["mean"] == pytest.approx(above.mean())
    assert report["std"] == pytest.approx(above.std(ddof=1))
    assert prepared[15, 0, 0] == pytest.approx((50 - above.mean()) / above.std(ddof=1), rel=1e-5)
    assert report["air_is_above_0"] is False


def test_a_scan_above_0_everywhere_is_reported_not_changed():
    values = np.full((SIZE, SIZE, SIZE), 3815.0, np.float32)
    values[20:40] = 18000.0
    _, report = pipeline.prepare(values, np.ones(values.shape, bool), _params())
    assert report["air_is_above_0"] is True
    assert report["share_of_scan_voxels_above_0"] == 1.0
    assert report["mean"] > 3815


def test_conversions_are_explicit_and_applied_in_order():
    values = np.full((SIZE, SIZE, SIZE), 3815.0, np.float32)
    values[20:40] = 18000.0  # clamped to 10000, then shifted to 6185
    values[50:60] = 5815.0  # shifted to 2000
    inside = np.ones(values.shape, bool)
    inside[:5] = False
    prepared, report = pipeline.prepare(values, inside, _params(clamp=[0.0, 10000.0], intensity_shift=-3815.0))
    assert report["scan_minimum_in_cube"] == 0.0
    assert report["air_is_above_0"] is False
    assert report["padding_value"] == 0.0
    assert report["mean"] == pytest.approx((20 * 6185.0 + 10 * 2000.0) / 30)


def test_padding_can_be_the_lowest_scan_value():
    values = np.full((SIZE, SIZE, SIZE), 500.0, np.float32)
    values[30:40] = 900.0
    inside = np.zeros(values.shape, bool)
    inside[20:60] = True
    _, report = pipeline.prepare(values, inside, _params(padding="minimum"))
    assert report["padding_value"] == 500.0


@pytest.mark.parametrize("fill", [0.0, -5.0, 7.0])
def test_a_flat_or_empty_cube_is_a_clear_error(fill):
    with pytest.raises(ValueError, match="above 0|one value"):
        pipeline.prepare(np.full((SIZE, SIZE, SIZE), fill, np.float32), np.ones((SIZE, SIZE, SIZE), bool), _params())


# --- segment -------------------------------------------------------------------------------------


def test_track_a_returns_a_mask_on_the_volumes_grid_around_the_click(fake):
    scan = volume()
    mask = segment(scan, {"cube": "click", "seed_zyx": [30, 35, 40]})
    assert np.asarray(mask).shape == scan.shape
    assert mask[30, 35, 40] == 1
    assert fake.clicks == [(64, 64, 64)]
    assert mask.report["spacing_mm"] == 1.5 and mask.report["side_mm"] == 192.0
    assert mask.report["post_processing"] is None


def test_axis_order_end_to_end(fake):
    """A bright voxel at a known (z, y, x) reaches the network as (x, y, z), and the click comes back to it."""
    data = np.full((64, 80, 96), 10.0, np.float32)
    data[20, 30, 40] = 500.0
    scan = Volume(data, (1.5, 1.5, 1.5), "HU", "dicom", "0" * 64)
    mask = segment(scan, {"cube": "click", "seed_zyx": [20, 30, 40]})
    assert fake.cubes[0].argmax() == np.ravel_multi_index((64, 64, 64), (SIZE, SIZE, SIZE))
    assert mask[20, 30, 40] == 1
    data2 = data.copy()
    data2[20, 30, 42] = 900.0  # two voxels along x from the click: axis 0 of the network's cube
    mask = segment(Volume(data2, (1.5, 1.5, 1.5), "HU", "dicom", "0" * 64), {"cube": "click", "seed_zyx": [20, 30, 40]})
    assert np.unravel_index(fake.cubes[1].argmax(), (SIZE, SIZE, SIZE)) == (66, 64, 64)


def test_track_b_scales_to_the_box_and_the_whole_scan_stays_visible(fake):
    scan = volume(shape=(60, 70, 80), spacing=(2.0, 2.0, 2.0))
    box = {"min_mm": [40.0, 60.0, 70.0], "max_mm": [104.0, 80.0, 80.0]}  # 64 mm longest, 10 mm shortest
    mask = segment(scan, {"cube": "box", "box": box, "seed_zyx": [25, 35, 37]})
    assert mask.report["spacing_mm"] == 0.5
    assert mask.report["box_share_in_cube"] == 1.0
    # not cropped to the box: the cube's other axes reach real scan outside the box, not padding
    assert mask.report["padding_share_of_cube"] == 0.0
    assert not np.isclose(fake.cubes[0], fake.cubes[0][0, 0, 0]).all()
    assert mask[25, 35, 37] == 1


def test_a_box_near_the_scans_edge_is_padded_beyond_the_scan_only(fake):
    box = {"min_mm": [20.0, 20.0, 20.0], "max_mm": [84.0, 40.0, 30.0]}
    report = segment(volume(), {"cube": "box", "box": box, "seed_zyx": [25, 15, 12]}).report
    assert 0 < report["padding_share_of_cube"] < 0.2
    assert report["box_share_in_cube"] == 1.0


def test_a_margin_makes_the_cube_coarser(fake):
    box = {"min_mm": [20.0, 20.0, 20.0], "max_mm": [84.0, 40.0, 30.0]}
    mask = segment(volume(), {"cube": "box", "box": box, "box_margin": 0.5, "seed_zyx": [25, 15, 12]})
    assert mask.report["spacing_mm"] == 0.75


def test_another_click_in_the_same_box_cube_does_not_rerun_the_encoder(fake):
    scan = volume()
    box = {"min_mm": [20.0, 20.0, 20.0], "max_mm": [84.0, 40.0, 30.0]}
    first = segment(scan, {"cube": "box", "box": box, "seed_zyx": [25, 15, 12]})
    second = segment(scan, {"cube": "box", "box": box, "seed_zyx": [30, 12, 13]})
    assert len(fake.cubes) == 1
    assert len(fake.clicks) == 2 and fake.clicks[0] != fake.clicks[1]
    assert first.report["encoder_output_reused"] is False and second.report["encoder_output_reused"] is True
    assert not np.array_equal(first, second)


def test_a_different_box_runs_the_encoder_again(fake):
    scan = volume()
    segment(scan, {"cube": "box", "box": {"min_mm": [20.0, 20.0, 20.0], "max_mm": [84.0, 40.0, 30.0]}, "seed_zyx": [25, 15, 12]})
    segment(scan, {"cube": "box", "box": {"min_mm": [10.0, 20.0, 20.0], "max_mm": [84.0, 40.0, 30.0]}, "seed_zyx": [25, 15, 12]})
    assert len(fake.cubes) == 2


def test_the_threshold_is_a_parameter(fake, monkeypatch):
    monkeypatch.setattr(fake, "predict", lambda embedding, click: np.full((SIZE, SIZE, SIZE), 0.6, np.float32))
    assert segment(volume(), {"cube": "click", "seed_zyx": [30, 35, 40]}).any()
    assert not segment(volume(), {"cube": "click", "seed_zyx": [30, 35, 40], "threshold": 0.7}).any()


def test_a_mask_reaching_the_cube_inside_a_long_scan_warns(fake, monkeypatch):
    monkeypatch.setattr(fake, "predict", lambda embedding, click: np.ones((SIZE, SIZE, SIZE), np.float32))
    report = segment(volume(shape=(400, 40, 40), spacing=(2.0, 2.0, 2.0)), {"cube": "click", "seed_zyx": [200, 20, 20]}).report
    assert report["faces_reached_inside_scan"] == ["low z", "high z"]
    assert any("cut off" in warning for warning in report["warnings"])
    assert report["scan_axes_longer_than_cube"] == ["z"]
    assert report["scan_share_in_cube"] == pytest.approx(192 / 800)


def test_a_scan_above_0_warns(fake):
    report = segment(volume(), {"cube": "click", "seed_zyx": [30, 35, 40]}).report
    assert report["normalisation"]["air_is_above_0"] is True
    assert any("also takes in the air" in warning for warning in report["warnings"])


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"seed_zyx": [1, 1, 1]}, "needs the parameter cube"),
        ({"cube": "auto", "seed_zyx": [1, 1, 1]}, "needs the parameter cube"),
        ({"cube": "box", "seed_zyx": [1, 1, 1]}, "needs a box"),
        ({"cube": "click", "box": {"min_mm": [0, 0, 0], "max_mm": [9, 9, 9]}, "seed_zyx": [1, 1, 1]}, "takes no box"),
        ({"cube": "box", "box": {"min_mm": [0, 0, 0], "max_mm": [9, 9, 9]}, "spacing_mm": 1.5, "seed_zyx": [1, 1, 1]}, "remove spacing_mm"),
        ({"cube": "click", "box_margin": 0.2, "seed_zyx": [1, 1, 1]}, "belongs to cube: box"),
        ({"cube": "click"}, "needs a click"),
        ({"cube": "click", "seed_zyx": [999, 1, 1]}, "outside the volume"),
        ({"cube": "click", "seed_zyx": [1, 1, 1], "threshold": 2}, "between 0 and 1"),
        ({"cube": "click", "seed_zyx": [1, 1, 1], "padding": "mean"}, "padding"),
        ({"cube": "click", "seed_zyx": [1, 1, 1], "clamp": [5, 1]}, "low < high"),
        ({"cube": "click", "seed_zyx": [1, 1, 1], "sliding_window": True}, "unknown sam_med3d parameters"),
    ],
)
def test_wrong_parameters_are_clear_errors(fake, parameters, message):
    with pytest.raises(ValueError, match=message):
        segment(volume(), parameters)


# --- assets --------------------------------------------------------------------------------------


def test_without_assets_the_error_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setenv(assets.ASSETS_VARIABLE, str(tmp_path))
    with pytest.raises(assets.NotInstalledError) as error:
        assets.require()
    assert assets.INSTALL_COMMAND in str(error.value)
    assert str(tmp_path) in str(error.value)


def test_wrong_weights_and_a_wrong_commit_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(assets.ASSETS_VARIABLE, str(tmp_path))
    source = tmp_path / assets.SOURCE_DIR / "segment_anything"
    source.mkdir(parents=True)
    (source / "build_sam3D.py").write_text("")
    weights = tmp_path / assets.WEIGHTS_FILE
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not the weights")
    monkeypatch.setattr(assets, "missing", lambda: [])
    monkeypatch.setattr(assets, "source_commit", lambda path: "abc")
    with pytest.raises(assets.NotInstalledError, match="expected " + assets.SOURCE_COMMIT):
        assets.require()
    monkeypatch.setattr(assets, "source_commit", lambda path: assets.SOURCE_COMMIT)
    with pytest.raises(assets.NotInstalledError, match="SHA-256"):
        assets.require()


def test_importing_the_method_does_not_import_torch():
    code = "import sys; import src.models.foundation.sam_med3d; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# --- install script ------------------------------------------------------------------------------


def test_correct_weights_are_not_downloaded_again(tmp_path, monkeypatch):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"good")
    monkeypatch.setattr(assets, "WEIGHTS_SHA256", assets.sha256_file(weights))
    called = []
    installer.ensure_weights(weights, fetch=lambda url, target: called.append(url))
    assert called == []


def test_a_bad_download_leaves_nothing_behind(tmp_path, monkeypatch):
    weights = tmp_path / "sub" / "w.pth"
    monkeypatch.setattr(assets, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="expected"):
        installer.ensure_weights(weights, fetch=lambda url, target: target.write_bytes(b"bad"))
    assert list(weights.parent.iterdir()) == []


def test_a_good_download_lands_under_its_final_name(tmp_path, monkeypatch):
    import hashlib

    weights = tmp_path / "w.pth"
    monkeypatch.setattr(assets, "WEIGHTS_SHA256", hashlib.sha256(b"good").hexdigest())
    installer.ensure_weights(weights, fetch=lambda url, target: target.write_bytes(b"good"))
    assert weights.read_bytes() == b"good"
    assert [path.name for path in tmp_path.iterdir()] == ["w.pth"]


def test_the_pinned_source_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "source_commit", lambda path: assets.SOURCE_COMMIT)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no git call expected"))
    installer.ensure_source(tmp_path / "src")


# --- through the runner --------------------------------------------------------------------------


def _config(tmp_path, *, cube, box=None, seed=(30, 35, 40), shape=(60, 70, 80), target=(slice(25, 35), slice(30, 40), slice(35, 45))):
    data = volume(shape=shape).data
    (tmp_path / "scan.json").write_text(json.dumps({"data": np.round(data, 1).tolist(), "spacing_mm": [2.0, 2.0, 2.0], "units": "HU"}))
    reference = np.zeros(data.shape, np.uint8)
    reference[target] = 1
    nib.save(nib.Nifti1Image(reference.transpose(2, 1, 0), np.diag([2.0, 2.0, 2.0, 1.0])), str(tmp_path / "ref.nii.gz"))
    config = {
        "experiment": {"id": "exp", "repetitions": 1},
        "data": {"input_path": str(tmp_path / "scan.json"), "reference_mask_path": str(tmp_path / "ref.nii.gz"), "seed_zyx": list(seed)},
        "algorithm": {"name": "sam_med3d", "parameters": {"cube": cube}},
        "mesh": {"smoothing_iterations": 0, "output_format": "glb"},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    if box:
        config["data"]["box"] = box
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


BOX = {"min_mm": [40.0, 40.0, 40.0], "max_mm": [100.0, 100.0, 100.0]}


def test_the_runner_runs_track_a_on_the_whole_scan(tmp_path, fake):
    result = run_experiment(_config(tmp_path, cube="click"))
    directory = tmp_path / "results" / "exp"
    for name in ("config.yaml", "mask.nii.gz", "mesh.glb", "metrics.json", "run.log", "method_report.json"):
        assert (directory / name).is_file(), name
    saved = json.loads((directory / "metrics.json").read_text())
    assert saved["method_report"]["cube"] == "click"
    assert result.mask.shape == volume().shape
    assert result.mask[30, 35, 40] == 1
    assert result.box_prompted is False


def test_the_runner_runs_track_b_without_cropping_to_the_box(tmp_path, fake):
    result = run_experiment(_config(tmp_path, cube="box", box=BOX))
    assert result.box_prompted is True
    assert result.metrics["method_report"]["spacing_mm"] == pytest.approx(60 / 128)
    assert result.mask[30, 35, 40] == 1  # the seed is in the whole scan's voxels, not the box's
    assert result.mask.shape == volume().shape


def test_the_runner_refuses_both_mismatches(tmp_path, fake, capsys):
    assert runner_main([str(_config(tmp_path, cube="click", box=BOX))]) == 1
    assert "takes no box" in capsys.readouterr().err
    assert runner_main([str(_config(tmp_path, cube="box"))]) == 1
    assert "needs a box" in capsys.readouterr().err


def test_the_runner_reports_a_reference_that_does_not_fit_in_the_cube(tmp_path, fake):
    # the reference is 400 mm long, against a cube of 192 mm
    path = _config(
        tmp_path, cube="click", seed=(100, 20, 20), shape=(200, 40, 40), target=(slice(0, 200), slice(15, 25), slice(15, 25))
    )
    result = run_experiment(path)
    assert result.metrics["reference_share_in_view"] == pytest.approx(96 / 200)
    assert any("of the reference mask lies outside" in warning for warning in result.warnings)


def test_a_method_that_is_not_installed_fails_with_a_readable_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(network, "_loaded", {})
    monkeypatch.setattr(assets, "missing", lambda: ["the Python package torch"])
    assert runner_main([str(_config(tmp_path, cube="click"))]) == 1
    error = capsys.readouterr().err
    assert "SAM-Med3D is not installed" in error and assets.INSTALL_COMMAND in error
    assert "Traceback" not in error


def test_methods_without_a_report_still_crop_to_the_box(tmp_path):
    from src.pipeline.runner import register_algorithm

    seen = {}

    def plain(volume_, parameters):
        seen["shape"] = volume_.shape
        seen["parameters"] = dict(parameters)
        return np.ones(volume_.shape, np.uint8)

    register_algorithm("plain", plain)
    path = _config(tmp_path, cube="click", box=BOX)
    config = yaml.safe_load(path.read_text())
    config["algorithm"] = {"name": "plain", "parameters": {}}
    path.write_text(yaml.safe_dump(config))
    result = run_experiment(path)
    assert seen["shape"] == (30, 30, 30)
    assert "box" not in seen["parameters"]
    assert "method_report" not in result.metrics
    assert ndi.label(result.mask)[1] == 1
