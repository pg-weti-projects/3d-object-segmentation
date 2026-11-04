"""Scan and reference mask must agree on orientation, on small synthetic volumes."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import yaml

from src.pipeline.alignment import MISMATCH, OK, UNVERIFIED, _stride, check_alignment
from src.pipeline.runner import register_algorithm, run_experiment

SPACING = (1.0, 1.0, 1.0)


def fish(*, bladder_value: float = -900.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A body with an off-centre egg-shaped bladder and an off-centre dense bone, so that no flip maps
    the scan onto itself; `bladder_value` makes the bladder dark (gas) or bright (contrast agent)."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[:40, :48, :52].astype(float)
    data = np.full(zz.shape, -1000.0)
    data[((zz - 20) / 18) ** 2 + ((yy - 24) / 21) ** 2 + ((xx - 26) / 23) ** 2 <= 1] = 40.0
    bladder = ((zz - 13) / 7) ** 2 + ((yy - 17) / 6) ** 2 + ((xx - 18) / 8) ** 2 <= 1
    data[((zz - 28) / 3) ** 2 + ((yy - 33) / 3) ** 2 + ((xx - 36) / 6) ** 2 <= 1] = 900.0
    data[bladder] = bladder_value
    return data + rng.normal(0, 5, data.shape), bladder


@pytest.mark.parametrize("bladder_value", [-900.0, 400.0], ids=["dark", "bright"])
def test_a_mask_on_its_scan_is_accepted(bladder_value):
    data, mask = fish(bladder_value=bladder_value)
    report = check_alignment(data, mask, SPACING)
    assert report["status"] == OK and report["message"] is None
    assert report["edge"] > 1.1


@pytest.mark.parametrize("bladder_value", [-900.0, 400.0], ids=["dark", "bright"])
@pytest.mark.parametrize("axis, name", [(0, "z"), (1, "y"), (2, "x")])
def test_a_mirrored_mask_is_caught_and_the_axis_named(axis, name, bladder_value):
    data, mask = fish(bladder_value=bladder_value)
    report = check_alignment(data, np.flip(mask, axis), SPACING)
    assert report["status"] == MISMATCH
    assert f"mirrored along {name}" in report["message"]


def test_a_uniform_scan_cannot_confirm_anything():
    _, mask = fish()
    report = check_alignment(np.full(mask.shape, 7.0), mask, SPACING)
    assert report["status"] == UNVERIFIED
    assert "cannot be confirmed" in report["message"]


def test_a_bladder_too_faint_to_see_is_unverified_rather_than_passed():
    data, mask = fish(bladder_value=40.0)  # exactly the tissue around it: there is no boundary to see
    assert check_alignment(data, mask, SPACING)["status"] == UNVERIFIED


def test_an_empty_mask_cannot_be_checked():
    data, mask = fish()
    assert check_alignment(data, np.zeros_like(mask), SPACING)["status"] == UNVERIFIED


def test_a_mask_on_another_grid_is_refused():
    data, _ = fish()
    with pytest.raises(ValueError, match="same grid"):
        check_alignment(data, np.zeros((3, 3, 3), bool), SPACING)


def test_a_large_scan_is_reduced_along_its_finest_axis_first():
    assert _stride((10, 10, 10), (1.0, 1.0, 1.0)) == (1, 1, 1)
    step = _stride((263, 512, 512), (3.5, 0.839, 0.839))
    assert step[0] == 1 and step[1] > 1 and step[2] > 1
    assert np.prod([n // k for n, k in zip((263, 512, 512), step)]) <= 4_000_000


def test_the_result_does_not_depend_on_the_intensity_unit():
    data, mask = fish()
    first = check_alignment(data, mask, SPACING)
    second = check_alignment(data * 0.25 + 5000.0, mask, SPACING)
    assert first["edge"] == pytest.approx(second["edge"], rel=1e-3)
    assert first["separation"] == pytest.approx(second["separation"], rel=1e-3)


# --- through the runner ---------------------------------------------------------------------------


def _run(tmp_path, mirror: bool):
    data, mask = fish()
    (tmp_path / "scan.json").write_text(json.dumps({"data": data.tolist(), "spacing_mm": [1.0, 1.0, 1.0], "units": "HU"}))
    reference = np.flip(mask, 0) if mirror else mask
    nib.save(nib.Nifti1Image(reference.astype(np.uint8).transpose(2, 1, 0), np.eye(4)), str(tmp_path / "ref.nii.gz"))
    register_algorithm("known_mask", lambda volume, parameters: mask.astype(np.uint8))
    config = {
        "experiment": {"id": "exp", "repetitions": 1},
        "data": {"input_path": str(tmp_path / "scan.json"), "reference_mask_path": str(tmp_path / "ref.nii.gz")},
        "algorithm": {"name": "known_mask", "parameters": {}},
        "mesh": {"smoothing_iterations": 0, "output_format": "glb"},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return run_experiment(path), tmp_path / "results" / "exp" / "metrics.json"


def test_the_runner_records_an_agreeing_mask_without_a_warning(tmp_path):
    result, saved = _run(tmp_path, mirror=False)
    assert result.warnings == []
    assert result.metrics["dice"] == 1.0
    assert json.loads(saved.read_text())["alignment"]["status"] == OK


def test_the_runner_warns_about_a_mirrored_mask_and_still_reports_metrics(tmp_path):
    result, saved = _run(tmp_path, mirror=True)
    assert any("mirrored along z" in warning for warning in result.warnings)
    written = json.loads(saved.read_text())
    assert written["alignment"]["status"] == MISMATCH
    assert written["dice"] == result.metrics["dice"]
    assert "mirrored along z" in (saved.parent / "run.log").read_text()
