"""Our cube and our network against upstream's, from tests/data/sam_med3d_golden.npz. That file holds
upstream's own output (torchio's cube, medim's network) on input generated in
scripts/sam_med3d_golden.py; to regenerate it, see that script. The cube needs no weights, so it is
a fast test. The network needs them, so it is slow."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.sam_med3d_golden import PREPARE, STRIDE, cube_from_affine, synthetic_cube, synthetic_scan
from src.models.foundation.sam_med3d import assets, cube as geometry
from src.models.foundation.sam_med3d.network import SamMed3D
from src.models.foundation.sam_med3d.pipeline import prepare

FIXTURE = np.load(Path(__file__).parent / "data" / "sam_med3d_golden.npz")


def _our_cube(truncate: bool) -> np.ndarray:
    data, _, affine = synthetic_scan()
    image = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    array = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])[::-1]
    cube = cube_from_affine(FIXTURE["roi_affine"], image.affine, spacing)
    values, inside = geometry.sample(array, spacing, cube)
    if truncate:
        values[inside] = np.trunc(np.asarray(values[inside], dtype=np.float64))
    return geometry.to_network_axes(prepare(values, inside, PREPARE)[0])


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
