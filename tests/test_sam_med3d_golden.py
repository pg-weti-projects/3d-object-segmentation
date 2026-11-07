"""Our cube and our network against upstream's, from tests/data/sam_med3d_golden.npz. That file holds
upstream's own output (torchio's cube, medim's network) on the synthetic input below, recorded once
with upstream's code. The cube needs no weights, so it is a fast test. The network needs them, so it
is slow."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.models.foundation.sam_med3d import assets, cube as geometry
from src.models.foundation.sam_med3d.network import SamMed3D
from src.models.foundation.sam_med3d.pipeline import prepare

FIXTURE = np.load(Path(__file__).parent / "data" / "sam_med3d_golden.npz")
STRIDE = 8  # of a 128^3 cube, in the fixture


def synthetic_scan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small scan (x, y, z), its label and its affine, on a grid like upstream's test scan: thick
    slices, and the x axis running from the patient's left, so that canonical order flips it."""
    x, y, z = np.mgrid[:64, :72, :30].astype(np.float64)
    data = np.full(x.shape, -1000.0)
    data[((x - 32) / 28) ** 2 + ((y - 36) / 32) ** 2 + ((z - 15) / 13) ** 2 <= 1] = 40.0
    data += 30 * np.sin(x / 5) * np.cos(y / 7) * np.sin(z / 3 + 1)
    label = ((x - 24) / 10) ** 2 + ((y - 30) / 9) ** 2 + ((z - 12) / 4) ** 2 <= 1
    data[label] = 250.0
    affine = np.array([[-0.8, 0, 0, 120.0], [0, 0.8, 0, -40.0], [0, 0, 4.0, 10.0], [0, 0, 0, 1.0]])
    return data.astype(np.int16), label.astype(np.uint8), affine


def synthetic_cube() -> np.ndarray:
    """A normalised cube for the network part of the fixture."""
    x, y, z = np.mgrid[:128, :128, :128].astype(np.float32)
    field = np.sin(x / 9) * np.cos(y / 13) * np.sin(z / 7)
    field[((x - 70) / 20) ** 2 + ((y - 60) / 25) ** 2 + ((z - 64) / 18) ** 2 <= 1] += 2.0
    return field.astype(np.float32)


def cube_from_affine(roi_affine: np.ndarray, scan_affine: np.ndarray, spacing_zyx) -> geometry.Cube:
    """The placement of upstream's cube, in our mm: voxel i of an axis covers [i * s, (i + 1) * s)."""
    size = float(roi_affine[0, 0])
    corner_xyz = [
        roi_affine[axis, 3] - size / 2 - (scan_affine[axis, 3] - spacing_zyx[2 - axis] / 2)
        for axis in range(3)
    ]
    return geometry.Cube(tuple(corner_xyz[::-1]), size)


def _our_cube(truncate: bool) -> np.ndarray:
    data, _, affine = synthetic_scan()
    image = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    array = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])[::-1]
    cube = cube_from_affine(FIXTURE["roi_affine"], image.affine, spacing)
    values, inside = geometry.sample(array, spacing, cube)
    if truncate:
        values[inside] = np.trunc(np.asarray(values[inside], dtype=np.float64))
    return geometry.to_network_axes(prepare(values, inside)[0])


def test_our_cube_is_upstreams_cube_once_truncated_like_its_resampler():
    ours = _our_cube(truncate=True)[::STRIDE, ::STRIDE, ::STRIDE]
    difference = np.abs(ours - FIXTURE["cube"])
    assert np.mean(difference) < 1e-4
    assert np.mean(difference > 1e-3) < 0.01  # whole-number edge cases of the float to int cast


def test_without_truncation_our_cube_differs_by_less_than_one_unit_of_the_scan():
    """The one known difference: upstream's resampler loses up to 1 HU towards zero, and ours does not."""
    ours = _our_cube(truncate=False)[::STRIDE, ::STRIDE, ::STRIDE]
    difference = np.abs(ours - FIXTURE["cube"])
    assert difference.max() < 0.1  # measured 0.089 on this small scan, whose thin normalisation amplifies 1 unit
    assert np.corrcoef(ours.ravel(), FIXTURE["cube"].ravel())[0, 1] > 0.999


@pytest.mark.slow
def test_our_network_gives_upstreams_probabilities():
    if assets.missing():
        pytest.skip(f"SAM-Med3D is not installed: missing {assets.missing()}")
    located = assets.require()
    network = SamMed3D(located.source, located.weights)
    click = tuple(int(v) for v in FIXTURE["click"])
    probability = network.predict(network.embed(synthetic_cube()), click)[::STRIDE, ::STRIDE, ::STRIDE]
    assert np.abs(probability - FIXTURE["probability"]).max() < 1e-3
