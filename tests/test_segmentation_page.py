from __future__ import annotations

import io
import json
import re

import numpy as np
import pytest

from src.api import segmentation
from src.api.app import create_app
from src.api.segmentation import algorithm_config, mask_preview, parse_seed, preview_steps
from tests.test_segment import make_fish_hu


def fish_upload() -> tuple[tuple[io.BytesIO, str], np.ndarray, tuple[int, int, int]]:
    volume, bladder, seed = make_fish_hu()
    payload = {"voxels": volume.data.tolist(), "spacing_mm": list(volume.spacing_mm), "units": "HU"}
    return (io.BytesIO(json.dumps(payload).encode()), "fish.json"), bladder, seed


def load_form(upload) -> dict:
    return {"volume": [upload], "target_spacing": "0.5,0.5,0.5", "hu_range": "-1000,200"}


def volume_id(page: str) -> str:
    return re.search(r'name="volume_id" value="([0-9a-f]+)"', page).group(1)


def script_value(page: str, name: str):
    start = page.index(f"window.{name} = ") + len(f"window.{name} = ")
    return json.loads(page[start : page.index(";\n", start)])


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config["SEGMENTATION_STORE_DIR"] = str(tmp_path / "store")
    return app.test_client()


@pytest.fixture
def loaded(client):
    upload, bladder, seed = fish_upload()
    response = client.post("/segment", data=load_form(upload), content_type="multipart/form-data")
    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    return client, response.get_data(as_text=True), bladder, seed


def run(client, page: str, seed, algorithm="growcut", **extra):
    data = {
        "volume_id": volume_id(page),
        "seed_zyx": ",".join(str(v) for v in seed),
        "algorithm": algorithm,
        **extra,
    }
    return client.post("/segment/run", data=data)


def test_load_form(client):
    page = client.get("/segment").get_data(as_text=True)
    assert 'name="volume"' in page and "Load volume" in page


def test_view_after_loading(loaded):
    _, page, bladder, _ = loaded
    for name in ("chan_vese", "geodesic_active_contour", "growcut"):
        assert f'value="{name}"' in page
    assert 'name="seed_zyx"' in page
    assert script_value(page, "volumePreviewPickSeed") == "seed_zyx"
    assert script_value(page, "volumePreviewMask") is None
    assert np.shape(script_value(page, "volumePreviewData")) == bladder.shape
    assert script_value(page, "volumePreviewStep") == [1, 1, 1]


@pytest.mark.parametrize(
    ("algorithm", "extra"),
    [
        (
            "chan_vese",
            {
                "chan_vese__data_term_sign": "chan_vese",
                "chan_vese__init_anchor": "center",
                "chan_vese__remove_outside_air": "on",
            },
        ),
        ("geodesic_active_contour", {"geodesic_active_contour__domain_size_mm": "none"}),
        ("growcut", {}),
    ],
)
def test_segmentation_on_stored_volume(loaded, algorithm, extra):
    client, page, bladder, seed = loaded
    response = run(client, page, seed, algorithm, **extra)
    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    result = response.get_data(as_text=True)
    assert "Result" in result
    assert script_value(result, "volumePreviewSeed") == list(seed)
    mask = np.array(script_value(result, "volumePreviewMask"), dtype=bool)
    assert mask.shape == bladder.shape
    assert 2 * (mask & bladder).sum() / (mask.sum() + bladder.sum()) > 0.8
    assert volume_id(result) == volume_id(page)
    assert run(client, result, seed, algorithm, **extra).status_code == 200


def test_only_last_volume_is_kept(loaded, tmp_path):
    client, first, _, seed = loaded
    upload, _, _ = fish_upload()
    second = client.post("/segment", data=load_form(upload), content_type="multipart/form-data")
    assert second.status_code == 200
    files = sorted(p.name for p in (tmp_path / "store").iterdir())
    assert files == ["volume.json", "volume.npy"]
    response = run(client, first, seed)
    assert response.status_code == 400
    assert "Load the scan again" in response.get_data(as_text=True)
    assert run(client, second.get_data(as_text=True), seed).status_code == 200


def test_invalid_seed_keeps_view_with_error(loaded):
    client, page, _, _ = loaded
    response = run(client, page, (999, 0, 0))
    assert response.status_code == 400
    text = response.get_data(as_text=True)
    assert "Could not segment the volume" in text
    assert volume_id(text) == volume_id(page)


def test_parameter_typo_shows_error(loaded):
    client, page, _, seed = loaded
    response = run(client, page, seed, advanced_parameters='{"fg_radius": 1}')
    assert response.status_code == 400
    assert "literówka" in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "extra",
    [{"chan_vese__max_iterations": "none"}, {"advanced_parameters": '{"max_iterations": null}'}],
)
def test_none_for_required_parameter_shows_error(loaded, extra):
    client, page, _, seed = loaded
    assert run(client, page, seed, "chan_vese", **extra).status_code == 400


def test_chan_vese_seed_outside_object_shows_error(loaded):
    client, page, _, _ = loaded
    extra = {"chan_vese__remove_outside_air": "on", "chan_vese__roi_radius_mm": "0.5"}
    response = run(client, page, (14, 20, 1), "chan_vese", **extra)
    assert response.status_code == 400
    assert "Could not segment the volume" in response.get_data(as_text=True)


def test_missing_data_file_shows_error(loaded, tmp_path):
    client, page, _, seed = loaded
    (tmp_path / "store" / "volume.npy").unlink()
    response = run(client, page, seed)
    assert response.status_code == 400
    assert "Load the scan again" in response.get_data(as_text=True)


def test_interrupted_save_rejects_old_id(loaded, tmp_path, monkeypatch):
    from src.api import segmentation

    client, page, _, seed = loaded

    def broken_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(segmentation.np, "save", broken_save)
    upload, _, _ = fish_upload()
    assert (
        client.post("/segment", data=load_form(upload), content_type="multipart/form-data").status_code == 400
    )
    monkeypatch.undo()
    assert run(client, page, seed).status_code == 400
    assert not list((tmp_path / "store").glob("*.tmp"))


def test_seed_with_spaces_and_kept_text_on_error(loaded):
    client, page, _, seed = loaded
    data = {"volume_id": volume_id(page), "seed_zyx": " ".join(map(str, seed)), "algorithm": "growcut"}
    assert client.post("/segment/run", data=data).status_code == 200
    data["seed_zyx"] = "1,2"
    text = client.post("/segment/run", data=data).get_data(as_text=True)
    assert 'value="1,2"' in text


def test_run_without_loaded_volume(client):
    response = client.post(
        "/segment/run", data={"volume_id": "abc", "seed_zyx": "1,1,1", "algorithm": "growcut"}
    )
    assert response.status_code == 400
    assert "Load the scan again" in response.get_data(as_text=True)


def test_many_dicom_files_are_accepted_by_the_form(client):
    files = [(io.BytesIO(b"not dicom"), f"slice_{i:04d}.dcm") for i in range(2500)]
    data = {"volume": files, "target_spacing": "1,1,1", "hu_range": "-1000,200"}
    response = client.post("/segment", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert create_app().config["MAX_FORM_PARTS"] == 5000


def test_large_volume_preview_is_downsampled(client, monkeypatch):
    monkeypatch.setattr(segmentation, "PREVIEW_MAX_SIDE", 8)
    upload, bladder, seed = fish_upload()
    page = client.post("/segment", data=load_form(upload), content_type="multipart/form-data")
    assert page.status_code == 200
    page = page.get_data(as_text=True)
    step = script_value(page, "volumePreviewStep")
    expected = [-(-n // k) for n, k in zip(bladder.shape, step)]
    assert step != [1, 1, 1]
    assert list(np.shape(script_value(page, "volumePreviewData"))) == expected
    assert script_value(page, "volumePreviewShape") == list(bladder.shape)
    response = run(client, page, seed)
    assert response.status_code == 200
    result = response.get_data(as_text=True)
    assert script_value(result, "volumePreviewSeed") == list(seed)
    assert list(np.shape(script_value(result, "volumePreviewMask"))) == expected
    assert "every" in result


class TestPreview:
    def test_preview_steps_fit_limits(self):
        shape = (2622, 766, 780)
        steps = preview_steps(shape)
        sizes = [-(-n // k) for n, k in zip(shape, steps)]
        assert max(sizes) <= segmentation.PREVIEW_MAX_SIDE
        assert np.prod(sizes) <= segmentation.PREVIEW_MAX_VOXELS
        assert preview_steps((60, 80, 90)) == (1, 1, 1)

    def test_mask_preview_keeps_small_objects(self):
        mask = np.zeros((10, 9, 7), dtype=bool)
        mask[5, 4, 3] = True
        reduced = np.array(mask_preview(mask, (2, 3, 4)))
        assert reduced.shape == (5, 3, 2)
        assert reduced.sum() == 1 and reduced[2, 1, 0] == 1


class TestFormParsing:
    def test_parse_seed(self):
        assert parse_seed("3, 4,5") == (3, 4, 5)
        assert parse_seed("3;4;5") == (3, 4, 5)
        for bad in ("", "1,2", "a,b,c", "1.5,2,3"):
            with pytest.raises(ValueError):
                parse_seed(bad)

    def test_algorithm_config(self):
        cfg = algorithm_config(
            {
                "algorithm": "growcut",
                "growcut__domain_size_mm": "none",
                "growcut__bg_radius_mm": "12",
                "growcut__fg_radius_mm": "",
                "advanced_parameters": '{"neighborhood": "moore"}',
            }
        )
        assert cfg.name == "growcut"
        assert cfg.parameters == {"domain_size_mm": None, "bg_radius_mm": 12.0, "neighborhood": "moore"}
        cv = algorithm_config({"algorithm": "chan_vese", "chan_vese__max_iterations": "20"})
        assert cv.parameters == {"max_iterations": 20, "remove_outside_air": False}
        for bad in (
            {"algorithm": "unet"},
            {"algorithm": "growcut", "growcut__bg_radius_mm": "abc"},
            {"algorithm": "growcut", "advanced_parameters": "[1]"},
            {"algorithm": "chan_vese", "chan_vese__init_anchor": "middle"},
            {"algorithm": "chan_vese", "chan_vese__max_iterations": "none"},
        ):
            with pytest.raises(ValueError):
                algorithm_config(bad)


def test_parameter_fields_have_help(loaded):
    from src.api.segmentation import ALGORITHM_FIELDS, FIELD_HELP

    _, page, _, _ = loaded
    for name, fields in ALGORITHM_FIELDS.items():
        assert set(FIELD_HELP[name]) == {key for key, _label, _kind in fields}
    assert page.count('class="param-help"') == sum(len(fields) for fields in ALGORITHM_FIELDS.values())
    assert 'title="Distance from the seed where background is forced' in page
