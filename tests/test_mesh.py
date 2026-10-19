from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.mesh import (
    export_mesh,
    reconstruct_mesh,
    reconstruct_mesh_from_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MASKS = sorted((REPO_ROOT / "data" / "reference_masks").glob("*/Segmentation.nii.gz"))


def _ellipsoid(shape, center, radii):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0


def test_sphere_is_watertight_with_correct_volume():
    mask = _ellipsoid((40, 40, 40), (20, 20, 20), (12, 12, 12))

    mesh = reconstruct_mesh(mask, smoothing_iterations=10)

    assert mesh.is_watertight
    assert mesh.volume > 0  # positive volume = outward-facing triangle winding
    expected = 4 / 3 * np.pi * 12**3
    assert mesh.volume == pytest.approx(expected, rel=0.05)


def test_spacing_scales_mesh_and_axes_are_xyz():
    # (z, y, x) mask, deliberately different extents per axis and anisotropic voxels
    mask = _ellipsoid((30, 40, 50), (15, 20, 25), (8, 12, 20))
    spacing = (2.0, 1.0, 0.5)  # (sz, sy, sx) in mm

    mesh = reconstruct_mesh(mask, spacing, smoothing_iterations=0)

    extents = mesh.bounds[1] - mesh.bounds[0]  # (x, y, z)
    expected_xyz = np.array([2 * 20 * 0.5, 2 * 12 * 1.0, 2 * 8 * 2.0])
    voxel_xyz = np.array([0.5, 1.0, 2.0])  # tolerance: one voxel per axis
    assert np.all(np.abs(extents - expected_xyz) <= voxel_xyz + 1e-6)
    # centre in physical coordinates (x, y, z)
    centre = mesh.bounds.mean(axis=0)
    np.testing.assert_allclose(centre, [25 * 0.5, 20 * 1.0, 15 * 2.0], atol=1.0)


def test_object_touching_volume_border_is_still_closed():
    mask = np.ones((10, 10, 10), dtype=np.uint8)

    mesh = reconstruct_mesh(mask, smoothing_iterations=0)

    assert mesh.is_watertight
    # surface sits half a voxel outside the outermost voxel centres -> ~10^3
    assert mesh.volume == pytest.approx(10**3, rel=0.03)


def test_only_largest_component_is_kept():
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[5:25, 5:25, 5:25] = 1  # big cube
    mask[32:35, 32:35, 32:35] = 1  # stray fragment

    cleaned = reconstruct_mesh(mask, smoothing_iterations=0)
    raw = reconstruct_mesh(mask, clean=False, smoothing_iterations=0)

    assert len(cleaned.split(only_watertight=False)) == 1
    assert len(raw.split(only_watertight=False)) == 2
    assert cleaned.bounds[1].max() < 30  # fragment's corner (>= 32) is gone


def test_decimation_reduces_triangle_count():
    mask = _ellipsoid((50, 50, 50), (25, 25, 25), (20, 20, 20))

    full = reconstruct_mesh(mask, smoothing_iterations=0, decimation_factor=0.0)
    reduced = reconstruct_mesh(mask, smoothing_iterations=0, decimation_factor=0.5)

    assert len(reduced.faces) == pytest.approx(len(full.faces) * 0.5, rel=0.1)
    assert reduced.volume == pytest.approx(full.volume, rel=0.05)


def test_settings_object_drives_reconstruction():
    mask = _ellipsoid((40, 40, 40), (20, 20, 20), (12, 12, 12))
    settings = SimpleNamespace(smoothing_iterations=5, decimation_factor=0.3)

    mesh = reconstruct_mesh_from_settings(mask, (1.0, 1.0, 1.0), settings)

    assert len(mesh.faces) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mask": np.zeros((5, 5, 5))},  # empty
        {"mask": np.ones((5, 5))},  # not 3D
        {"mask": np.ones((5, 5, 5)), "spacing": (1.0, 0.0, 1.0)},
        {"mask": np.ones((5, 5, 5)), "decimation_factor": 1.0},
        {"mask": np.ones((5, 5, 5)), "smoothing_iterations": -1},
    ],
)
def test_invalid_input_raises(kwargs):
    with pytest.raises(ValueError):
        reconstruct_mesh(**kwargs)


@pytest.mark.parametrize("fmt", ["glb", "stl", "obj", "ply"])
def test_export_writes_every_supported_format(tmp_path, fmt):
    mesh = reconstruct_mesh(_ellipsoid((20, 20, 20), (10, 10, 10), (6, 6, 6)))

    path = export_mesh(mesh, tmp_path / "nested" / f"mesh.{fmt}")

    assert path.exists() and path.stat().st_size > 0


def test_export_rejects_unknown_format(tmp_path):
    mesh = reconstruct_mesh(_ellipsoid((20, 20, 20), (10, 10, 10), (6, 6, 6)))

    with pytest.raises(ValueError):
        export_mesh(mesh, tmp_path / "mesh.xyz")


@pytest.mark.parametrize("mask_path", REFERENCE_MASKS, ids=lambda p: p.parent.name)
def test_real_reference_masks_give_valid_meshes(mask_path):
    nib = pytest.importorskip("nibabel")
    image = nib.load(mask_path)
    # NIfTI is (x, y, z); the pipeline uses (z, y, x) like core.Volume
    mask = np.asarray(image.dataobj).transpose(2, 1, 0)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])[::-1]

    mesh = reconstruct_mesh(mask, spacing, smoothing_iterations=10, decimation_factor=0.2)

    assert len(mesh.faces) > 100
    assert mesh.is_watertight
    assert mesh.volume > 0
    voxel_volume = float(np.prod(spacing))
    assert mesh.volume == pytest.approx(int((mask > 0).sum()) * voxel_volume, rel=0.25)
