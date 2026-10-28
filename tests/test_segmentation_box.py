from __future__ import annotations

import numpy as np
import pytest

from src.api.segmentation import METHODS, algorithm_config, form_context, parse_box
from src.core.volume import Volume, hash_bytes
from tests.test_segmentation_page import loaded, client, run, script_value, volume_id  # noqa: F401 (fixtures)


def box_fields(bladder: np.ndarray, margin: int = 2) -> dict[str, str]:
    _, ys, xs = np.nonzero(bladder)
    _, rows, columns = bladder.shape
    return {
        "box_x_min": str(max(xs.min() - margin, 0)),
        "box_x_max": str(min(xs.max() + 1 + margin, columns)),
        "box_y_min": str(max(ys.min() - margin, 0)),
        "box_y_max": str(min(ys.max() + 1 + margin, rows)),
    }


def dice(mask: np.ndarray, reference: np.ndarray) -> float:
    return 2 * (mask & reference).sum() / (mask.sum() + reference.sum())


TICKED = {"box_region__bladder_is_dark": "on", "box_region__use_hull": "on"}  # the defaults of the form


def mask_of(page: str) -> np.ndarray:
    return np.array(script_value(page, "volumePreviewMask"), dtype=bool)


def test_box_region_is_offered_with_its_fields():
    context = form_context()

    assert "box_region" in [name for name, _ in context["algorithms"]] and "box_region" in METHODS
    assert {key for key, _, _ in context["fields"]["box_region"]} <= set(context["defaults"]["box_region"])
    assert set(context["help"]["box_region"]) == {key for key, _, _ in context["fields"]["box_region"]}


def test_page_has_the_box_controls(loaded):
    _, page, _, _ = loaded

    for name in ("box_x_min", "box_x_max", "box_y_min", "box_y_max"):
        assert f'name="{name}"' in page
    assert 'id="box-overlay"' in page and 'id="box-clear"' in page and "box-select.js" in page
    assert 'value="box_region"' in page


def test_box_region_with_a_box_finds_the_bladder(loaded):
    client, page, bladder, seed = loaded

    response = run(client, page, seed, "box_region", **TICKED, **box_fields(bladder))

    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    result = response.get_data(as_text=True)
    mask = mask_of(result)
    assert mask.shape == bladder.shape
    assert dice(mask, bladder) > 0.8
    assert "x " in result and "(all slices)" in result  # box shown in the result
    assert 'name="box_x_min" value="' + box_fields(bladder)["box_x_min"] + '"' in result  # survives a re-run


def test_box_region_without_a_box_asks_for_one_and_keeps_the_view(loaded):
    client, page, _, seed = loaded

    response = run(client, page, seed, "box_region")

    assert response.status_code == 400
    assert "Draw a box" in response.get_data(as_text=True)
    assert 'name="seed_zyx"' in response.get_data(as_text=True)


def test_any_method_can_use_the_box(loaded):
    client, page, bladder, seed = loaded
    fields = box_fields(bladder)

    response = run(client, page, seed, "growcut", **fields)

    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    mask = mask_of(response.get_data(as_text=True))
    x0, x1, y0, y1 = (int(fields[k]) for k in ("box_x_min", "box_x_max", "box_y_min", "box_y_max"))
    assert mask.sum() > 0
    outside = mask.copy()
    outside[:, y0:y1, x0:x1] = False
    assert not outside.any()  # nothing outside the box


def test_seed_outside_the_box_is_reported(loaded):
    client, page, bladder, seed = loaded
    fields = box_fields(bladder, margin=0)
    far_seed = (seed[0], 0, 0)

    response = run(client, page, far_seed, "box_region", **TICKED, **fields)

    assert response.status_code == 400
    assert "outside the box" in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"box_x_min": "a", "box_x_max": "5", "box_y_min": "1", "box_y_max": "6"}, "four whole numbers"),
        ({"box_x_min": "5", "box_x_max": "5", "box_y_min": "1", "box_y_max": "6"}, "does not fit"),
        ({"box_x_min": "0", "box_x_max": "9999", "box_y_min": "0", "box_y_max": "5"}, "does not fit"),
        ({"box_x_min": "1", "box_x_max": "5", "box_y_min": "1"}, "four whole numbers"),
    ],
)
def test_invalid_box_is_reported(loaded, fields, message):
    client, page, _, seed = loaded

    response = run(client, page, seed, "box_region", **fields)

    assert response.status_code == 400
    assert message in response.get_data(as_text=True)


def test_parse_box_covers_the_whole_z_range_in_mm():
    volume = Volume(np.zeros((10, 20, 30), np.float32), (2.0, 0.5, 0.5), "HU", "test", hash_bytes(b"b"))

    assert parse_box({}, volume) is None
    assert parse_box({"box_x_min": "", "box_x_max": "", "box_y_min": "", "box_y_max": ""}, volume) is None
    box = parse_box({"box_x_min": "4", "box_x_max": "10", "box_y_min": "2", "box_y_max": "8"}, volume)

    assert box.min_mm == (0.0, 1.0, 2.0) and box.max_mm == (20.0, 4.0, 5.0)  # z: all 10 slices x 2 mm


def test_box_region_parameters_from_the_form():
    config = algorithm_config(
        {"algorithm": "box_region", "box_region__threshold_fraction": "0.5", "box_region__open_mm": "0",
         "box_region__bladder_is_dark": "on"}
    )

    assert config.parameters == {
        "threshold_fraction": 0.5, "open_mm": 0.0, "bladder_is_dark": True, "use_hull": False,
    }
