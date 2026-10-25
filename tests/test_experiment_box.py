import pytest

from src.core.box import Box
from src.pipeline.experiment import ExperimentConfigError, load_experiment_config


def _write(tmp_path, data_extra=""):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "data:\n  input_path: scan.dcm\n" + data_extra + "algorithm:\n  name: medsam\n",
        encoding="utf-8",
    )
    return path


def test_config_without_box_is_unchanged(tmp_path):
    assert load_experiment_config(_write(tmp_path)).data.box is None


def test_box_is_read_from_the_config(tmp_path):
    path = _write(tmp_path, "  box:\n    min_mm: [1, 2, 3]\n    max_mm: [4, 5, 6]\n")

    config = load_experiment_config(path)

    assert config.data.box == Box((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


@pytest.mark.parametrize(
    "block",
    [
        "  box:\n    min_mm: [1, 2, 3]\n",  # max_mm missing
        "  box:\n    min_mm: [5, 5, 5]\n    max_mm: [1, 1, 1]\n",  # inverted
        "  box: [1, 2, 3]\n",  # not a mapping
        "  box:\n    min_mm: [1, 2, 3]\n    max_mm: [4, 5, 6]\n    margin: 2\n",  # unknown field
    ],
)
def test_invalid_box_is_a_config_error(tmp_path, block):
    with pytest.raises(ExperimentConfigError, match="data.box"):
        load_experiment_config(_write(tmp_path, block))


def test_seed_is_read_and_validated(tmp_path):
    assert load_experiment_config(_write(tmp_path, "  seed_zyx: [3, 4, 5]\n")).data.seed_zyx == (3, 4, 5)
    assert load_experiment_config(_write(tmp_path)).data.seed_zyx is None
    for bad in ("  seed_zyx: [1, 2]\n", "  seed_zyx: [1, 2, -3]\n", "  seed_zyx: [1.5, 2, 3]\n", "  seed_zyx: 7\n"):
        with pytest.raises(ExperimentConfigError, match="seed_zyx"):
            load_experiment_config(_write(tmp_path, bad))
