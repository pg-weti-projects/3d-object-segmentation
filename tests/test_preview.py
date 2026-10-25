import io
import json

import numpy as np

from src.api.app import create_app
from src.api.processing import slices_for_preview


def test_preview_is_globally_normalized_nested_zyx_floats():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 10 - 100

    slices = slices_for_preview(data)

    assert np.shape(slices) == (2, 3, 4)
    assert isinstance(slices[0][0][0], float)
    assert slices[0][0][0] == 0.0
    assert slices[-1][-1][-1] == 1.0
    assert slices[1][0][0] > slices[0][2][3]
    assert np.all(np.diff(np.ravel(slices)) >= 0)


def test_constant_volume_preview_is_black():
    slices = slices_for_preview(np.full((2, 2, 2), 7.0))

    assert not np.any(slices)


def test_result_page_shows_three_planes():
    volume = {"voxels": np.arange(60).reshape(3, 4, 5).tolist(), "spacing_mm": [1, 1, 1]}
    upload = (io.BytesIO(json.dumps(volume).encode()), "scan.json")
    data = {
        "target_spacing": "1,1,1",
        "hu_range": "-1000,1000",
        "denoise": "",
        "volume": [upload],
    }

    response = create_app().test_client().post(
        "/process", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    for plane, last in (("axial", 2), ("coronal", 3), ("sagittal", 4)):
        assert f'id="{plane}-canvas"' in page
        assert f'id="{plane}-slider" data-plane="{plane}"\n                   min="0" max="{last}"' in page
    assert "data-volume-slices='[[[" in page
    assert "data-volume-spacing='[1.0, 1.0, 1.0]'" in page
