import io
import os
import time

import nibabel as nib
import numpy as np
import pytest
import trimesh

from src.api.app import create_app
from src.pipeline.mesh import export_mesh, reconstruct_mesh


def _ellipsoid_mask(shape=(30, 40, 40), center=(15, 20, 20), radii=(8, 12, 12)):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0


def _nifti_bytes(mask, zooms=(1.0, 1.0, 1.0)):
    """`mask` is (z, y, x); `zooms` are the NIfTI voxel sizes in (x, y, z) order."""
    affine = np.diag([*zooms, 1.0])
    image = nib.Nifti1Image(mask.astype(np.uint8).transpose(2, 1, 0), affine)  # (z,y,x) -> (x,y,z)
    return image.to_bytes()


@pytest.fixture
def results(tmp_path):
    """A results/ directory with one experiment that has a .glb mesh."""
    mesh = reconstruct_mesh(_ellipsoid_mask(), (1.0, 1.0, 1.0), smoothing_iterations=2)
    export_mesh(mesh, tmp_path / "exp_a" / "mesh.glb")
    return tmp_path


@pytest.fixture
def client(results):
    app = create_app()
    app.config["RESULTS_DIR"] = results
    return app.test_client()


def test_viewer_without_results_explains_what_to_do(tmp_path):
    app = create_app()
    app.config["RESULTS_DIR"] = tmp_path / "does-not-exist-yet"

    response = app.test_client().get("/viewer")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "No meshes" in page
    assert "meshViewerConfig" not in page  # no viewer script without a mesh


def test_viewer_lists_results_newest_first(results, client):
    export_mesh(
        reconstruct_mesh(_ellipsoid_mask(), (1.0, 1.0, 1.0), smoothing_iterations=0),
        results / "exp_b" / "mesh.stl",
    )
    os.utime(results / "exp_a" / "mesh.glb", (time.time() - 100, time.time() - 100))

    page = client.get("/viewer").get_data(as_text=True)

    assert page.index("exp_b") < page.index("exp_a")
    assert "glb" in page and "stl" in page


def test_selected_result_gets_viewer_downloads_and_info(client):
    response = client.get("/viewer?id=exp_a")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'data-mesh-url="/meshes/exp_a/mesh.glb"' in page
    assert 'type="module"' in page and "webgl-viewer.js" in page
    for fmt in ("glb", "stl", "obj", "ply"):
        assert f'href="/meshes/exp_a/mesh.{fmt}?download=1"' in page
    assert "Watertight" in page and "Triangles" in page


def test_unknown_result_is_a_404_page(client):
    response = client.get("/viewer?id=nope")

    assert response.status_code == 404
    assert "No mesh found" in response.get_data(as_text=True)


def test_glb_is_served_as_stored_and_only_downloaded_on_request(client):
    inline = client.get("/meshes/exp_a/mesh.glb")
    download = client.get("/meshes/exp_a/mesh.glb?download=1")

    assert inline.status_code == 200
    assert inline.data[:4] == b"glTF"
    assert inline.mimetype == "model/gltf-binary"
    assert "Content-Disposition" not in inline.headers
    assert download.headers["Content-Disposition"] == 'attachment; filename="exp_a.glb"'


@pytest.mark.parametrize("fmt", ["stl", "obj", "ply"])
def test_other_formats_are_converted_from_the_stored_mesh(client, fmt):
    original = trimesh.load(io.BytesIO(client.get("/meshes/exp_a/mesh.glb").data), file_type="glb", force="mesh")

    response = client.get(f"/meshes/exp_a/mesh.{fmt}")

    assert response.status_code == 200
    converted = trimesh.load(io.BytesIO(response.data), file_type=fmt, force="mesh")
    assert len(converted.faces) == len(original.faces)
    np.testing.assert_allclose(converted.extents, original.extents, rtol=1e-4)


def test_glb_is_converted_when_only_another_format_is_stored(results, client):
    export_mesh(
        reconstruct_mesh(_ellipsoid_mask(), (1.0, 1.0, 1.0)), results / "exp_stl_only" / "mesh.stl"
    )

    response = client.get("/meshes/exp_stl_only/mesh.glb")

    assert response.status_code == 200 and response.data[:4] == b"glTF"


@pytest.mark.parametrize(
    "path",
    [
        "/meshes/missing/mesh.glb",
        "/meshes/exp_a/mesh.xyz",  # unsupported format
        "/meshes/..%2F..%2Fetc/mesh.glb",  # path traversal
        "/meshes/%2E%2E/mesh.glb",
        "/meshes/.hidden/mesh.glb",
    ],
)
def test_bad_mesh_requests_are_404(client, path):
    assert client.get(path).status_code == 404


def test_building_a_mesh_from_a_nifti_mask(results, client):
    data = {
        "mask": (io.BytesIO(_nifti_bytes(_ellipsoid_mask(), (1.0, 1.0, 3.0))), "mask.nii"),
        "smoothing_iterations": "5",
        "decimation_factor": "0.3",
    }

    response = client.post("/viewer/build", data=data, content_type="multipart/form-data")

    assert response.status_code == 302
    location = response.headers["Location"]
    assert location.startswith("/viewer?id=mask_")
    mesh_id = location.split("id=")[1]
    assert (results / mesh_id / "mesh.glb").is_file()
    page = client.get(location).get_data(as_text=True)
    assert f'data-mesh-url="/meshes/{mesh_id}/mesh.glb"' in page
    # radii (z, y, x) = (8, 12, 12) voxels with 3 mm slices: size (x, y, z) = (24, 24, 48) mm
    mesh = trimesh.load(results / mesh_id / "mesh.glb", force="mesh")
    np.testing.assert_allclose(mesh.extents, [24, 24, 48], atol=3)


@pytest.mark.parametrize(
    "files, form, message",
    [
        ({}, {}, "Choose a mask"),
        ({"mask": (io.BytesIO(b"abc"), "mask.txt")}, {}, "NIfTI"),
        ({"mask": (io.BytesIO(b"not a nifti file"), "mask.nii")}, {}, "readable NIfTI"),
        ({"mask": (io.BytesIO(b"not gzip"), "mask.nii.gz")}, {}, "readable NIfTI"),
        (
            {"mask": (io.BytesIO(_nifti_bytes(np.zeros((6, 6, 6)))), "empty.nii")},
            {},
            "empty",
        ),
        (
            {"mask": (io.BytesIO(_nifti_bytes(_ellipsoid_mask())), "mask.nii")},
            {"smoothing_iterations": "abc"},
            "must be a number",
        ),
        (
            {"mask": (io.BytesIO(_nifti_bytes(_ellipsoid_mask())), "mask.nii")},
            {"decimation_factor": "1.5"},
            "decimation_factor",
        ),
    ],
)
def test_invalid_mask_uploads_show_an_error(results, client, files, form, message):
    before = sorted(path.name for path in results.iterdir())

    response = client.post("/viewer/build", data={**files, **form}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert message in response.get_data(as_text=True)
    assert sorted(path.name for path in results.iterdir()) == before  # nothing was stored


def test_decimal_comma_from_the_browser_is_accepted(results, client):
    data = {
        "mask": (io.BytesIO(_nifti_bytes(_ellipsoid_mask())), "Segmentation.nii"),
        "smoothing_iterations": "10",
        "decimation_factor": "0,2",
    }

    response = client.post("/viewer/build", data=data, content_type="multipart/form-data")

    assert response.status_code == 302


def test_rejected_upload_is_logged_with_the_reason(client, caplog):
    data = {"mask": (io.BytesIO(b"abc"), "scan.tif")}

    with caplog.at_level("WARNING"):
        client.post("/viewer/build", data=data, content_type="multipart/form-data")

    assert "scan.tif" in caplog.text and "NIfTI" in caplog.text
