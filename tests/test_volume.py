import json

import numpy as np

from src.core.loaders import load_volume
from src.core.preprocessor import PreprocessingConfig, preprocess


def test_json_volume_is_loaded_with_metadata(tmp_path):
    path = tmp_path / "scan.json"
    path.write_text(
        json.dumps(
            {
                "voxels": [[[1, 2], [3, 4]]],
                "spacing_mm": [2, 1, 0.5],
                "units": "HU",
            }
        ),
        encoding="utf-8",
    )

    volume = load_volume(path)

    assert volume.shape == (1, 2, 2)
    assert volume.spacing_mm == (2.0, 1.0, 0.5)
    assert volume.source_format == "voxels-json"
    assert len(volume.source_hash) == 64
    np.testing.assert_array_equal(volume.data, [[[1, 2], [3, 4]]])


def test_preprocessing_clips_inverts_denoises_and_resamples(tmp_path):
    path = tmp_path / "scan.json"
    path.write_text(
        json.dumps({"data": np.arange(8).reshape(2, 2, 2).tolist()}),
        encoding="utf-8",
    )
    volume = load_volume(path)

    result = preprocess(
        volume,
        PreprocessingConfig(
            target_spacing=(0.5, 0.5, 0.5),
            hu_range=(1, 5),
            invert_hu=True,
            denoise=True,
        ),
    )

    assert result.shape == (4, 4, 4)
    assert result.spacing_mm == (0.5, 0.5, 0.5)
    assert result.units == "HU"
    assert result.data.min() >= 1
    assert result.data.max() <= 5


def test_unknown_suffix_is_rejected(tmp_path):
    path = tmp_path / "scan.bmp"
    path.write_bytes(b"not a volume")

    try:
        load_volume(path)
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("unsupported input should raise ValueError")
