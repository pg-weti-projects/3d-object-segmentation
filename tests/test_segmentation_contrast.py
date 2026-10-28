"""Tests for the contrast-driven start on the segmentation page."""

from __future__ import annotations

import numpy as np

from src.api.segmentation import algorithm_config, form_context, measured_contrast
from src.models.params import ALGORITHMS, resolve_algorithm_params
from tests.test_segmentation_box import box_fields, dice, mask_of
from tests.test_segmentation_page import client, loaded, run, script_value, volume_id  # noqa: F401

CONTRAST = {"init_from": "contrast"}


def field(algorithm: str, key: str, value: str) -> dict[str, str]:
    return {f"{algorithm}__{key}": value}


def test_every_analytical_method_offers_the_start_field():
    context = form_context()

    for name in ALGORITHMS:
        keys = [key for key, _label, _kind in context["fields"][name]]
        assert keys[0] == "init_from"
        assert "contrast_fraction" in keys
        assert set(context["help"][name]) == set(keys)
        assert context["defaults"][name]["init_from"] == "seed_ball"
    assert context["choices"]["init_from"] == ("seed_ball", "contrast")


def test_the_start_reaches_the_parameters():
    config = algorithm_config(
        {"algorithm": "growcut", **field("growcut", "init_from", "contrast"),
         **field("growcut", "contrast_fraction", "0.3")}
    )

    assert config.parameters["init_from"] == "contrast"
    assert config.parameters["contrast_fraction"] == 0.3


def test_an_empty_fraction_is_left_to_the_default():
    # An empty field is not sent at all, so the method keeps its own default,
    # which for contrast_fraction means "choose it from the measured regime".
    config = algorithm_config(
        {"algorithm": "growcut", **field("growcut", "init_from", "contrast"),
         **field("growcut", "contrast_fraction", "")}
    )

    assert "contrast_fraction" not in config.parameters
    assert resolve_algorithm_params("growcut", config.parameters)["contrast_fraction"] is None


def test_an_unknown_start_is_refused(loaded):
    client, page, _, seed = loaded

    response = run(client, page, seed, "growcut", **field("growcut", "init_from", "magic"))

    assert response.status_code == 400
    assert "init_from" in response.get_data(as_text=True)


def test_contrast_start_finds_the_bladder_and_reports_the_contrast(loaded):
    client, page, bladder, seed = loaded

    response = run(
        client,
        page,
        seed,
        "growcut",
        **field("growcut", "init_from", "contrast"),
        **box_fields(bladder),
    )

    assert response.status_code == 200, response.get_data(as_text=True)[:2000]
    result = response.get_data(as_text=True)
    assert dice(mask_of(result), bladder) > 0.7
    assert "Contrast" in result and "high" in result
    assert "of the dynamic range" in result


def test_the_seed_ball_start_does_not_report_a_contrast(loaded):
    client, page, bladder, seed = loaded

    response = run(client, page, seed, "growcut", **box_fields(bladder))

    assert response.status_code == 200
    assert "of the dynamic range" not in response.get_data(as_text=True)


def test_measured_contrast_reads_both_shapes():
    measured = {"regime": "low", "separability": 0.1, "cnr": 4.0, "notes": ["low contrast"]}

    assert measured_contrast({"contrast": measured})["regime"] == "low"
    assert measured_contrast({"roi_meta": {"contrast": measured}})["cnr"] == 4.0
    assert measured_contrast({}) is None
    assert measured_contrast({"contrast": None}) is None
    assert measured_contrast({"contrast": {"separability": 0.5}}) is None
