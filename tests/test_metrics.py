import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import cdist

from src.pipeline.metrics import (
    assd,
    compute_metrics,
    dice,
    hd95,
    iou,
    volume_similarity,
    write_metrics_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MASKS = sorted((REPO_ROOT / "data" / "reference_masks").glob("*/Segmentation.nii.gz"))


def _ball(shape, center, radius):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    ) <= radius**2


def _brute_force_distances(pred, ref, spacing):
    """Independent reference: all-pairs distances between surface voxel centres."""
    structure = ndimage.generate_binary_structure(3, 1)

    def surface_points(mask):
        surface = mask & ~ndimage.binary_erosion(mask, structure, border_value=0)
        return np.argwhere(surface) * np.array(spacing)

    a, b = surface_points(pred), surface_points(ref)
    d = cdist(a, b)
    d_ab, d_ba = d.min(axis=1), d.min(axis=0)
    return (
        max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)),
        np.concatenate([d_ab, d_ba]).mean(),
    )


def test_overlap_metrics_on_hand_computed_case():
    ref = np.zeros((1, 1, 10), dtype=np.uint8)
    pred = np.zeros((1, 1, 10), dtype=np.uint8)
    ref[0, 0, 0:4] = 1  # 4 voxels
    pred[0, 0, 2:6] = 1  # 4 voxels, 2 overlapping

    assert dice(pred, ref) == pytest.approx(2 * 2 / 8)
    assert iou(pred, ref) == pytest.approx(2 / 6)
    assert volume_similarity(pred, ref) == pytest.approx(1.0)  # equal volumes


def test_volume_similarity_penalises_size_difference():
    ref = np.ones((1, 1, 6), dtype=np.uint8)
    pred = np.zeros((1, 1, 6), dtype=np.uint8)
    pred[0, 0, :2] = 1  # 2 vs 6 voxels

    assert volume_similarity(pred, ref) == pytest.approx(1 - 4 / 8)


def test_identical_masks_are_perfect():
    mask = _ball((30, 30, 30), (15, 15, 15), 8)

    metrics = compute_metrics(mask, mask, (1.0, 1.0, 1.0))

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["vs"] == 1.0
    assert metrics["hd95_mm"] == 0.0
    assert metrics["assd_mm"] == 0.0


def test_disjoint_masks_have_zero_overlap():
    pred = np.zeros((10, 10, 10), dtype=np.uint8)
    ref = np.zeros((10, 10, 10), dtype=np.uint8)
    pred[:3, :3, :3] = 1
    ref[6:, 6:, 6:] = 1

    metrics = compute_metrics(pred, ref)

    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["hd95_mm"] > 0


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (3.0, 0.8, 0.8), (0.5, 2.0, 1.0)])
def test_surface_distances_match_brute_force(spacing):
    rng = np.random.default_rng(0)
    ref = _ball((24, 24, 24), (12, 12, 12), 7)
    pred = _ball((24, 24, 24), (13, 11, 14), 6)
    # a few extra voxels so the surfaces are not trivially smooth
    pred |= rng.random(pred.shape) > 0.995

    expected_hd95, expected_assd = _brute_force_distances(pred, ref, spacing)

    assert hd95(pred, ref, spacing) == pytest.approx(expected_hd95)
    assert assd(pred, ref, spacing) == pytest.approx(expected_assd)


def test_distances_scale_with_spacing():
    ref = _ball((30, 30, 30), (15, 15, 15), 8)
    pred = _ball((30, 30, 30), (15, 15, 18), 8)

    base = compute_metrics(pred, ref, (1.0, 1.0, 1.0))
    scaled = compute_metrics(pred, ref, (2.0, 2.0, 2.0))

    assert scaled["hd95_mm"] == pytest.approx(2 * base["hd95_mm"])
    assert scaled["assd_mm"] == pytest.approx(2 * base["assd_mm"])
    assert scaled["dice"] == base["dice"]  # overlap does not depend on spacing


def test_shifted_ball_hd95_is_close_to_the_shift():
    ref = _ball((40, 40, 40), (20, 20, 20), 10)
    pred = _ball((40, 40, 40), (20, 20, 24), 10)  # shifted by 4 voxels along x

    result = hd95(pred, ref, (1.0, 1.0, 1.0))

    assert 3.0 <= result <= 4.5


def test_empty_prediction_is_a_valid_result_with_undefined_distances():
    ref = _ball((20, 20, 20), (10, 10, 10), 5)
    pred = np.zeros_like(ref)

    metrics = compute_metrics(pred, ref)

    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["vs"] == 0.0
    assert metrics["hd95_mm"] is None
    assert metrics["assd_mm"] is None
    json.dumps(metrics)  # must stay JSON-serialisable (no NaN / inf)


def test_empty_reference_raises():
    pred = _ball((20, 20, 20), (10, 10, 10), 5)

    with pytest.raises(ValueError, match="Reference"):
        compute_metrics(pred, np.zeros_like(pred))


@pytest.mark.parametrize(
    "pred, ref, spacing",
    [
        (np.ones((4, 4, 4)), np.ones((4, 4, 5)), (1, 1, 1)),  # shape mismatch
        (np.ones((4, 4)), np.ones((4, 4)), (1, 1, 1)),  # not 3D
        (np.ones((4, 4, 4)), np.ones((4, 4, 4)), (1, 0, 1)),  # bad spacing
        (np.ones((4, 4, 4)), np.ones((4, 4, 4)), (1, 1)),  # wrong length
    ],
)
def test_invalid_input_raises(pred, ref, spacing):
    with pytest.raises(ValueError):
        compute_metrics(pred, ref, spacing)


def test_volumes_are_reported_in_mm3():
    ref = np.zeros((4, 4, 4), dtype=np.uint8)
    ref[:2, :2, :2] = 1  # 8 voxels

    metrics = compute_metrics(ref, ref, (2.0, 1.0, 0.5))

    assert metrics["volume_ref_mm3"] == pytest.approx(8 * 1.0)
    assert metrics["volume_pred_mm3"] == pytest.approx(8 * 1.0)


def test_write_metrics_json_roundtrip(tmp_path):
    mask = _ball((20, 20, 20), (10, 10, 10), 5)
    metrics = compute_metrics(mask, mask)

    path = write_metrics_json(metrics, tmp_path / "results" / "exp" / "metrics.json")

    assert json.loads(path.read_text(encoding="utf-8")) == metrics


@pytest.mark.parametrize("mask_path", REFERENCE_MASKS, ids=lambda p: p.parent.name)
def test_real_reference_masks(mask_path):
    nib = pytest.importorskip("nibabel")
    image = nib.load(mask_path)
    ref = np.asarray(image.dataobj).transpose(2, 1, 0) > 0  # (x, y, z) -> (z, y, x)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])[::-1]
    box = ndimage.find_objects(ref.astype(np.uint8))[0]
    ref = np.pad(ref[box], 2)  # crop the empty surroundings: keeps the test fast

    same = compute_metrics(ref, ref, spacing)
    assert same["dice"] == 1.0 and same["hd95_mm"] == 0.0

    dilated = ndimage.binary_dilation(ref)  # one voxel larger in every direction
    grown = compute_metrics(dilated, ref, spacing)
    assert 0.5 < grown["dice"] < 1.0
    assert grown["vs"] < 1.0
    # surfaces are about one voxel apart (face connectivity), so HD95 should be
    # roughly one voxel, i.e. between the smallest and the largest voxel edge
    assert 0.5 * min(spacing) <= grown["hd95_mm"] <= 1.5 * max(spacing)
