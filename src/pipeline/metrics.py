"""

Segmentation quality metrics: Dice, IoU, HD95, ASSD and VS, computed
between a predicted mask and a reference mask (data/reference_masks/).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

_FACE_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)


def compute_metrics(
    pred: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """All metrics in one dict, ready to be written to ``metrics.json``:
    ``dice``, ``iou``, ``vs``, ``hd95_mm``, ``assd_mm`` plus both volumes in mm^3."""
    pred_mask, ref_mask, spacing = _validate(pred, reference, spacing)

    voxel_volume = float(np.prod(spacing))
    metrics = {
        "dice": dice(pred_mask, ref_mask),
        "iou": iou(pred_mask, ref_mask),
        "vs": volume_similarity(pred_mask, ref_mask),
        "hd95_mm": None,
        "assd_mm": None,
        "volume_pred_mm3": float(pred_mask.sum()) * voxel_volume,
        "volume_ref_mm3": float(ref_mask.sum()) * voxel_volume,
    }
    if pred_mask.any():
        distances = _surface_distances(pred_mask, ref_mask, spacing)
        metrics["hd95_mm"] = distances["hd95"]
        metrics["assd_mm"] = distances["assd"]
    return metrics


def dice(pred: np.ndarray, reference: np.ndarray) -> float:
    pred_mask, ref_mask = _as_pair(pred, reference)
    total = int(pred_mask.sum()) + int(ref_mask.sum())
    if total == 0:
        raise ValueError("Both masks are empty - Dice is undefined.")
    return 2.0 * int((pred_mask & ref_mask).sum()) / total


def iou(pred: np.ndarray, reference: np.ndarray) -> float:
    pred_mask, ref_mask = _as_pair(pred, reference)
    union = int((pred_mask | ref_mask).sum())
    if union == 0:
        raise ValueError("Both masks are empty - IoU is undefined.")
    return int((pred_mask & ref_mask).sum()) / union


def volume_similarity(pred: np.ndarray, reference: np.ndarray) -> float:
    pred_mask, ref_mask = _as_pair(pred, reference)
    v_pred, v_ref = int(pred_mask.sum()), int(ref_mask.sum())
    if v_pred + v_ref == 0:
        raise ValueError("Both masks are empty - VS is undefined.")
    return 1.0 - abs(v_pred - v_ref) / (v_pred + v_ref)


def hd95(pred: np.ndarray, reference: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance in mm."""
    return _surface_distances_checked(pred, reference, spacing)["hd95"]


def assd(pred: np.ndarray, reference: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """Average symmetric surface distance in mm."""
    return _surface_distances_checked(pred, reference, spacing)["assd"]


def write_metrics_json(metrics: dict, path: str | Path) -> Path:
    """Write metrics to e.g. ``results/<experiment_id>/metrics.json``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return target


def _surface_distances_checked(pred, reference, spacing) -> dict:
    pred_mask, ref_mask, spacing = _validate(pred, reference, spacing)
    if not pred_mask.any():
        raise ValueError("Prediction is empty - surface distances are undefined.")
    return _surface_distances(pred_mask, ref_mask, spacing)


def _surface_distances(pred: np.ndarray, reference: np.ndarray, spacing: tuple) -> dict:
    box = _bounding_box(pred | reference)
    pred, reference = pred[box], reference[box]

    pred_surface = _surface(pred)
    ref_surface = _surface(reference)

    to_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    d_pred_to_ref = to_ref[pred_surface]
    d_ref_to_pred = to_pred[ref_surface]

    pooled = np.concatenate([d_pred_to_ref, d_ref_to_pred])
    return {
        "hd95": float(max(np.percentile(d_pred_to_ref, 95), np.percentile(d_ref_to_pred, 95))),
        "assd": float(pooled.mean()),
    }


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of a non-empty boolean mask, without copying the volume."""
    slices = []
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        indices = np.flatnonzero(mask.any(axis=other_axes))
        slices.append(slice(int(indices[0]), int(indices[-1]) + 1))
    return tuple(slices)


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=_FACE_CONNECTIVITY, border_value=0)
    return mask & ~eroded


def _as_pair(pred: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_mask = _as_bool(pred)
    ref_mask = _as_bool(reference)
    if pred_mask.shape != ref_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs reference {ref_mask.shape}")
    return pred_mask, ref_mask


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    return array if array.dtype == bool else array != 0


def _validate(pred, reference, spacing) -> tuple[np.ndarray, np.ndarray, tuple]:
    pred_mask, ref_mask = _as_pair(pred, reference)
    if pred_mask.ndim != 3:
        raise ValueError(f"Masks must be 3-dimensional, got {pred_mask.ndim}")
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError("spacing must contain three positive finite values")
    if not ref_mask.any():
        raise ValueError("Reference mask is empty - metrics cannot be computed.")
    return pred_mask, ref_mask, spacing
