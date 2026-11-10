"""The experiments page: what the runner left in results/, as a table and a chart."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from src.api.app import create_app
from src.api.experiments import (
    CHARTABLE,
    bar_chart,
    comparison,
    format_value,
    list_experiments,
    page_context,
)

METRICS = {
    "dice": 0.9348,
    "iou": 0.8776,
    "vs": 0.9914,
    "hd95_mm": 4.2426,
    "assd_mm": 0.5544,
    "volume_pred_mm3": 31752.0,
    "volume_ref_mm3": 31212.0,
    "box_prompted": True,
    "runtime_s": 0.0124,
}


def write_run(results, name, *, metrics=METRICS, algorithm="box_region",
              scan="Oreo_fish", mesh=True, repetitions=3, age_minutes=0):
    directory = results / name
    directory.mkdir(parents=True)
    if algorithm is not None:
        (directory / "config.yaml").write_text(
            "experiment:\n"
            f"  id: \"{name}\"\n"
            f"  repetitions: {repetitions}\n"
            "data:\n"
            f"  input_path: \"data/raw/{scan}/density\"\n"
            "algorithm:\n"
            f"  name: \"{algorithm}\"\n",
            encoding="utf-8",
        )
    if metrics is not None:
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "mask.nii.gz").write_bytes(b"")
    if mesh:
        (directory / "mesh.glb").write_bytes(b"")
    if age_minutes:
        when = (datetime.now() - timedelta(minutes=age_minutes)).timestamp()
        for item in directory.iterdir():
            import os
            os.utime(item, (when, when))
    return directory


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def client(results):
    results.mkdir(parents=True, exist_ok=True)
    app = create_app()
    app.config["RESULTS_DIR"] = str(results)
    return app.test_client()


class TestReading:
    def test_a_run_is_read_from_its_config_and_metrics(self, results):
        write_run(results, "check_oreo_box")

        run, = list_experiments(results)

        assert run["id"] == "check_oreo_box"
        assert run["dataset"] == "Oreo_fish"  # not "density"
        assert run["algorithm"] == "box_region"
        assert run["repetitions"] == 3
        assert run["metrics"]["dice"] == pytest.approx(0.9348)
        assert run["box_prompted"] is True
        assert run["meshes"] == ["glb"]

    def test_newest_first(self, results):
        write_run(results, "older", age_minutes=120)
        write_run(results, "newer")

        assert [run["id"] for run in list_experiments(results)] == ["newer", "older"]

    def test_a_run_without_a_reference_mask_has_no_numbers(self, results):
        write_run(results, "no_metrics", metrics=None)

        run, = list_experiments(results)

        assert run["has_metrics"] is False
        assert run["metrics"]["dice"] is None
        assert run["algorithm"] == "box_region"

    def test_unreadable_files_do_not_break_the_listing(self, results):
        directory = write_run(results, "broken")
        (directory / "metrics.json").write_text("{not json", encoding="utf-8")
        (directory / "config.yaml").write_text("algorithm: [unbalanced", encoding="utf-8")

        run, = list_experiments(results)

        assert run["id"] == "broken"
        assert run["dataset"] == "-" and run["algorithm"] == "-"
        assert run["metrics"]["dice"] is None

    def test_files_and_odd_names_are_skipped(self, results):
        write_run(results, "good")
        results.joinpath("notes.txt").write_text("x", encoding="utf-8")
        results.joinpath(".hidden").mkdir()

        assert [run["id"] for run in list_experiments(results)] == ["good"]

    def test_a_missing_results_directory_is_empty(self, tmp_path):
        assert list_experiments(tmp_path / "nothing") == []


class TestComparison:
    def test_higher_is_better_for_dice(self, results):
        write_run(results, "worse", metrics={**METRICS, "dice": 0.1})
        write_run(results, "better", metrics={**METRICS, "dice": 0.9})

        rows = comparison(list_experiments(results), "dice")

        assert [row["id"] for row in rows] == ["better", "worse"]

    def test_lower_is_better_for_hd95(self, results):
        write_run(results, "far", metrics={**METRICS, "hd95_mm": 40.0})
        write_run(results, "close", metrics={**METRICS, "hd95_mm": 4.0})

        rows = comparison(list_experiments(results), "hd95_mm")

        assert [row["id"] for row in rows] == ["close", "far"]

    def test_runs_without_the_metric_are_left_out(self, results):
        write_run(results, "with")
        write_run(results, "without", metrics=None)

        assert [row["id"] for row in comparison(list_experiments(results), "dice")] == ["with"]


class TestFormatting:
    @pytest.mark.parametrize(
        "value, kind, expected",
        [
            (0.93481, "ratio", "0.935"),
            (4.2426, "mm", "4.24 mm"),
            (31752.0, "mm3", "31.8 mL"),
            (0.0124, "seconds", "0.01 s"),
            (125.0, "seconds", "2.1 min"),
            (None, "ratio", "-"),
            ("odd", "ratio", "odd"),
        ],
    )
    def test_values_are_readable(self, value, kind, expected):
        assert format_value(value, kind) == expected


class TestChart:
    def test_bars_are_drawn_and_text_is_escaped(self, results):
        write_run(results, "one", scan="A<B&C")

        svg = bar_chart(comparison(list_experiments(results), "dice"), "dice")

        assert svg.startswith("<svg") and "</svg>" in svg
        assert svg.count("<rect") == 1
        assert "A&lt;B&amp;C" in svg and "A<B&C" not in svg

    def test_no_runs_means_no_chart(self):
        assert bar_chart([], "dice") == ""


class TestPage:
    def test_the_table_shows_the_numbers(self, client, results):
        write_run(results, "check_oreo_box")

        page = client.get("/experiments").get_data(as_text=True)

        assert "check_oreo_box" in page
        assert "Oreo_fish" in page and "box_region" in page
        assert "0.935" in page and "4.24 mm" in page and "31.8 mL" in page
        assert "3 repetitions" in page
        assert "/viewer?id=check_oreo_box" in page  # the mesh opens in the viewer

    def test_an_empty_results_directory_says_so(self, client):
        page = client.get("/experiments").get_data(as_text=True)

        assert "No runs yet" in page
        assert "src.pipeline.runner" in page

    def test_the_metric_can_be_switched(self, client, results):
        write_run(results, "one")

        page = client.get("/experiments?metric=hd95_mm").get_data(as_text=True)

        assert "lower is better" in page
        assert "<svg" in page

    def test_an_unknown_metric_falls_back_to_the_first_one(self, client, results):
        write_run(results, "one")

        assert client.get("/experiments?metric=nonsense").status_code == 200
        assert page_context(results, "nonsense")["metric"] == CHARTABLE[0]

    def test_a_run_without_a_mesh_has_no_viewer_link(self, client, results):
        write_run(results, "maskless", mesh=False)

        page = client.get("/experiments").get_data(as_text=True)

        assert "maskless" in page
        assert "/viewer?id=maskless" not in page

    def test_the_page_is_linked_from_the_navigation(self, client):
        assert ">Experiments<" in client.get("/viewer").get_data(as_text=True)
