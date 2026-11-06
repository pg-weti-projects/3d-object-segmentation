"""SAM-Med3D on /segment, with the fake network of test_sam_med3d.py: no torch, no weights."""

from __future__ import annotations

import io
import json

import nibabel as nib
import numpy as np
import pytest
import yaml
from markupsafe import escape

from src.api import segmentation
from src.api.app import create_app
from src.api.processing import preprocessing_config
from src.core.box import Box
from src.models.foundation.sam_med3d import assets, network
from src.pipeline.runner import run_experiment
from tests.test_sam_med3d import FakeNetwork, volume
from tests.test_segmentation_page import volume_id

AS_CLI = {"target_spacing": "none", "hu_range": "none"}  # and no "denoise": unticked
BOX = {"box_x_min": "15", "box_x_max": "65", "box_y_min": "10", "box_y_max": "60", "box_z_from": "10", "box_z_to": "49"}
SEED = (30, 35, 40)


@pytest.fixture
def fake(monkeypatch):
    net = FakeNetwork()
    cached = network.CachedEncoder(net)
    monkeypatch.setattr(network, "get_network", lambda device="cpu", threads=None: cached)
    return net


@pytest.fixture
def scan_file(tmp_path):
    scan = volume()
    path = tmp_path / "scan.json"
    path.write_text(json.dumps({"voxels": np.round(scan.data, 2).tolist(), "spacing_mm": list(scan.spacing_mm), "units": "HU"}))
    return path


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config["SEGMENTATION_STORE_DIR"] = str(tmp_path / "store")
    return app.test_client()


def load(client, scan_file, **form):
    upload = (io.BytesIO(scan_file.read_bytes()), "scan.json")
    response = client.post("/segment", data={"volume": [upload], **form}, content_type="multipart/form-data")
    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    return response.get_data(as_text=True)


def run(client, page, **extra):
    data = {"volume_id": volume_id(page), "seed_zyx": ",".join(map(str, SEED)), "algorithm": "sam_med3d", **extra}
    return client.post("/segment/run", data=data)


@pytest.mark.parametrize("fields", [{}, {"sam_med3d__context": "scan", "sam_med3d__spacing": "trained", "sam_med3d__intensity_shift": "auto"}])
def test_the_web_gives_the_cli_mask_with_the_same_box_click_and_preprocessing(client, scan_file, fake, tmp_path, fields):
    page = load(client, scan_file, **AS_CLI)
    store = tmp_path / "store"
    form = {"volume_id": volume_id(page), "seed_zyx": ",".join(map(str, SEED)), "algorithm": "sam_med3d", **BOX, **fields}
    web = segmentation.run_segmentation(form, store)
    assert web["preview_step"] == [1, 1, 1]
    web_mask = np.array(web["result"]["mask"], bool)

    scan = volume()
    box = Box.from_voxels((10, 10, 15), (50, 60, 65), scan.spacing_mm)  # the same voxels as BOX
    reference = np.zeros(scan.shape, np.uint8)
    reference[25:35, 30:40, 35:45] = 1
    nib.save(nib.Nifti1Image(reference.transpose(2, 1, 0), np.diag([2.0, 2.0, 2.0, 1.0])), str(tmp_path / "ref.nii.gz"))
    config = {
        "experiment": {"id": "cli", "repetitions": 1},
        "data": {"input_path": str(scan_file), "reference_mask_path": str(tmp_path / "ref.nii.gz"), "seed_zyx": list(SEED), "box": box.to_mapping()},
        "algorithm": {"name": "sam_med3d", "parameters": {key.split("__")[1]: value for key, value in fields.items()}},
        "mesh": {"smoothing_iterations": 0, "output_format": "glb"},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    (tmp_path / "cli.yaml").write_text(yaml.safe_dump(config))
    cli_mask = run_experiment(tmp_path / "cli.yaml").mask != 0

    assert web_mask.any() and np.array_equal(web_mask, cli_mask)
    # the encoder ran once: the CLI's cube had the same content (the cache is keyed by its hash), and the same click
    assert len(fake.cubes) == 1 and len(fake.clicks) == 2 and fake.clicks[0] == fake.clicks[1]
    assert not any("resampled, clipped or denoised" in warning for warning in web["result"]["warnings"])


def test_the_method_is_listed_with_its_note_and_help(client, scan_file):
    page = load(client, scan_file, **AS_CLI)
    assert 'value="sam_med3d"' in page and "SAM-Med3D" in page
    assert "The first run loads the weights" in page
    for key in ("threshold", "padding", "intensity_shift", "context", "spacing"):
        assert f'name="sam_med3d__{key}"' in page and str(escape(segmentation.FIELD_HELP["sam_med3d"][key])) in page
    assert 'name="box_z_from"' in page and 'name="box_z_to"' in page


def test_without_a_box_it_asks_for_one(client, scan_file, fake):
    response = run(client, load(client, scan_file, **AS_CLI))
    assert response.status_code == 400 and "Draw a box around the bladder" in response.get_data(as_text=True)


def test_a_box_over_every_slice_is_refused_with_how_to_fix_it(client, scan_file, fake):
    whole_z = {key: value for key, value in BOX.items() if not key.startswith("box_z")}
    response = run(client, load(client, scan_file, **AS_CLI), **whole_z)
    text = response.get_data(as_text=True)
    assert response.status_code == 400 and "needs the box limited in z" in text and fake.cubes == []


def test_a_preprocessed_volume_is_flagged(client, scan_file, fake):
    page = load(client, scan_file, target_spacing="2,2,2", hu_range="-1000,200", denoise="on")
    text = run(client, page, **BOX).get_data(as_text=True)
    assert "resampled, clipped or denoised" in text and "Cube voxel" in text


def test_not_installed_is_labelled_and_reported_without_a_traceback(client, scan_file, monkeypatch):
    monkeypatch.setattr(assets, "missing", lambda: ["the Python package torch"])
    monkeypatch.setattr(network, "_loaded", {})
    page = load(client, scan_file, **AS_CLI)
    assert "SAM-Med3D (foundation model; one click in a box with a slice range) - not installed" in page
    response = run(client, page, **BOX)
    text = response.get_data(as_text=True)
    assert response.status_code == 400 and "SAM-Med3D is not installed" in text and "Traceback" not in text


def test_none_turns_resampling_and_clipping_off():
    config = preprocessing_config({"target_spacing": "none", "hu_range": " "})
    assert config.target_spacing is None and config.hu_range is None and config.denoise is False
    assert preprocessing_config({"target_spacing": "1,1,1", "hu_range": "-1000,200"}).hu_range == (-1000.0, 200.0)


def test_the_slice_range_is_optional_for_other_methods():
    scan = volume()
    whole = segmentation.parse_box({"box_x_min": "1", "box_x_max": "5", "box_y_min": "1", "box_y_max": "5"}, scan)
    assert whole.slices(scan.shape, scan.spacing_mm)[0] == slice(0, scan.shape[0])
    limited = segmentation.parse_box({"box_x_min": "1", "box_x_max": "5", "box_y_min": "1", "box_y_max": "5", "box_z_from": "40", "box_z_to": "12"}, scan)
    assert limited.slices(scan.shape, scan.spacing_mm)[0] == slice(12, 41)  # either order, both included
    with pytest.raises(ValueError, match="does not fit"):
        segmentation.parse_box({"box_x_min": "1", "box_x_max": "5", "box_y_min": "1", "box_y_max": "5", "box_z_from": "0", "box_z_to": "60"}, scan)
