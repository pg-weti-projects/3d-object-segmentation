import json

import nibabel as nib
import numpy as np
import pytest
import yaml

from src.pipeline.runner import (
    RunnerError,
    match_grid,
    register_algorithm,
    run_experiment,
)


def _scan(shape=(12, 30, 30)):
    """A bright ball (the 'bladder') and a bright distractor far away from it."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    data = np.zeros(shape, dtype=np.float32)
    ball = (zz - 6) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 16
    data[ball] = 100.0
    data[0:3, 0:4, 0:4] = 100.0  # distractor
    return data, ball


def _setup(tmp_path, *, box=None, reference=True, target_spacing=None, name="threshold"):
    data, ball = _scan()
    (tmp_path / "scan.json").write_text(
        json.dumps({"data": data.tolist(), "spacing_mm": [2.0, 1.0, 1.0], "units": "HU"})
    )
    config = {
        "experiment": {"id": "exp", "repetitions": 2},
        "data": {"input_path": str(tmp_path / "scan.json"), "format": "voxels-json"},
        "preprocessing": {"target_spacing": target_spacing},
        "algorithm": {"name": name, "parameters": {"level": 50}},
        "mesh": {"smoothing_iterations": 0, "output_format": "glb"},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    if box:
        config["data"]["box"] = box
    if reference:
        path = tmp_path / "ref.nii.gz"
        nib.save(
            nib.Nifti1Image(ball.astype(np.uint8).transpose(2, 1, 0), np.diag([1.0, 1.0, 2.0, 1.0])),
            str(path),
        )
        config["data"]["reference_mask_path"] = str(path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(autouse=True)
def threshold_algorithm():
    seen = {}

    def threshold(volume, parameters):
        seen["shape"] = volume.shape
        return (volume.data > parameters["level"]).astype(np.uint8)

    register_algorithm("threshold", threshold)
    return seen


def test_full_scan_run_writes_all_outputs(tmp_path):
    result = run_experiment(_setup(tmp_path))

    directory = tmp_path / "results" / "exp"
    for name in ("config.yaml", "mask.nii.gz", "mesh.glb", "metrics.json", "run.log"):
        assert (directory / name).is_file(), name
    assert result.box_prompted is False
    assert len(result.runtimes_s) == 2 and result.deterministic
    # the distractor is segmented too, so the score is below perfect
    assert 0.5 < result.metrics["dice"] < 1.0
    assert json.loads((directory / "metrics.json").read_text())["box_prompted"] is False


def test_box_hides_the_rest_of_the_scan_from_the_algorithm(tmp_path, threshold_algorithm):
    box = {"min_mm": [0, 8, 8], "max_mm": [24, 23, 23]}  # z in mm (spacing 2), y and x in mm (spacing 1)

    result = run_experiment(_setup(tmp_path, box=box))

    assert threshold_algorithm["shape"] == (12, 15, 15)  # cropped
    assert result.mask.shape == (12, 30, 30)  # restored on the full grid
    assert not result.mask[0:3, 0:4, 0:4].any()  # distractor excluded
    assert result.box_prompted and result.metrics["box_prompted"] is True
    assert result.metrics["dice"] > 0.95


def test_reference_is_matched_to_a_resampled_grid(tmp_path):
    result = run_experiment(_setup(tmp_path, target_spacing=[1.0, 1.0, 1.0]))

    assert result.mask.shape == (24, 30, 30)  # z resampled from 2 mm to 1 mm
    assert result.metrics["volume_ref_mm3"] > 0
    assert 0.5 < result.metrics["dice"] < 1.0


def test_run_without_reference_has_no_metrics(tmp_path):
    result = run_experiment(_setup(tmp_path, reference=False))

    assert result.metrics is None
    assert not (tmp_path / "results" / "exp" / "metrics.json").exists()


def test_empty_mask_gives_a_warning_but_no_mesh(tmp_path):
    register_algorithm("nothing", lambda volume, parameters: np.zeros(volume.shape, dtype=np.uint8))

    result = run_experiment(_setup(tmp_path, reference=False, name="nothing"))

    assert result.mesh_path is None and "empty" in result.warnings[0]


def test_unknown_algorithm_and_wrong_mask_shape_are_reported(tmp_path):
    with pytest.raises(RunnerError, match="Unknown algorithm"):
        run_experiment(_setup(tmp_path, name="does_not_exist"))

    register_algorithm("wrong", lambda volume, parameters: np.zeros((2, 2, 2)))
    with pytest.raises(RunnerError, match="shape"):
        run_experiment(_setup(tmp_path, name="wrong"))
    assert "failed" in (tmp_path / "results" / "exp" / "run.log").read_text()


def test_non_deterministic_algorithm_is_flagged(tmp_path):
    state = {"calls": 0}

    def flaky(volume, parameters):
        state["calls"] += 1
        mask = np.zeros(volume.shape, dtype=np.uint8)
        mask[5:8, 10:20, 10:20] = 1
        mask[0, 0, state["calls"]] = 1
        return mask

    register_algorithm("flaky", flaky)

    result = run_experiment(_setup(tmp_path, reference=False, name="flaky"))

    assert result.deterministic is False and "different" in result.warnings[0]


def test_match_grid_nearest_neighbour_and_outside_is_zero():
    reference = np.zeros((4, 4, 4), dtype=bool)
    reference[:2] = True  # first half in z

    same = match_grid(reference, (1, 1, 1), (4, 4, 4), (1, 1, 1))
    finer = match_grid(reference, (2, 2, 2), (8, 8, 8), (1, 1, 1))  # 1 mm grid over 8 mm
    longer = match_grid(reference, (1, 1, 1), (6, 4, 4), (1, 1, 1))

    assert same is not None and same.sum() == reference.sum()
    assert finer.shape == (8, 8, 8) and finer[:4].all() and not finer[4:].any()
    assert longer[:2].all() and not longer[4:].any()  # beyond the reference -> 0


def test_seed_from_the_config_is_given_to_the_algorithm_relative_to_the_box(tmp_path):
    seen = {}

    def spy(volume, parameters):
        seen["seed"] = parameters.get("seed_zyx")
        return np.zeros(volume.shape, dtype=np.uint8)

    register_algorithm("spy", spy)
    path = _setup(tmp_path, box={"min_mm": [0, 8, 8], "max_mm": [24, 23, 23]}, reference=False, name="spy")
    config = yaml.safe_load(path.read_text())
    config["data"]["seed_zyx"] = [6, 15, 15]  # full-volume voxel
    path.write_text(yaml.safe_dump(config))

    run_experiment(path)

    assert seen["seed"] == (6, 7, 7)  # box starts at y=8, x=8


def test_seed_outside_the_volume_is_reported(tmp_path):
    path = _setup(tmp_path, box={"min_mm": [0, 8, 8], "max_mm": [24, 23, 23]}, reference=False)
    config = yaml.safe_load(path.read_text())
    config["data"]["seed_zyx"] = [6, 0, 0]
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(RunnerError, match="seed_zyx"):
        run_experiment(path)
