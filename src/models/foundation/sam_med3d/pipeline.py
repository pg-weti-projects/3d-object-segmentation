"""
segment(volume, parameters): one click in a box, one cube, one mask on the volume's grid.

The cube is centred on the user's 3D box and scaled so that the box's longest side fills it; the
rest of the cube shows the scan around the box, as in training, and the mask is set to 0 outside
the box. Before the scan is shrunk onto a coarser cube it is smoothed, so that fine detail does not
alias. The box is never a model prompt. A cube centred on the click alone was tried and dropped
because its Dice was too low: a median of 0.32 over the 15 organs of upstream's human CT.

Nothing else is post-processed: no component filtering, no sliding window. A target that reaches
the edge of the box may be cut off, and the report says so.

Upstream's normalisation is a z-score over the voxels above 0, which behaves as in training
only when air is below 0, as it is in HU. On scans with (almost) nothing below 0 the default
intensity_shift "auto" moves 0 to where HU has it (`auto_shift`); HU scans are left as they are.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.core.volume import Volume

from . import assets, cube as geometry, network

logger = logging.getLogger(__name__)

TRAINED_SPACING_MM = 1.5  # reported against, not used: the box sets the spacing
PADDING = 0.0  # cube voxels outside the scan, as upstream
AUTO = "auto"
AIR_TO_ZERO = 1000 / 1050  # where 0 lies in HU between air (-1000) and soft tissue (+50)
BELOW_0_TRIGGER = 0.01  # "auto" shifts only a box with less than this share of its voxels below 0

DEFAULT_PARAMETERS: dict[str, Any] = {
    "seed_zyx": None,  # required: the click, as a voxel of the volume it is given
    "box_zyx": None,  # [[start, stop]] x 3, voxels of the box in the volume given; None = all of it
    "threshold": 0.5,  # on the sigmoid, as upstream
    "intensity_shift": AUTO,  # added to the scan before the z-score: a number, or "auto" (`auto_shift`)
    "device": "cpu",
    "threads": None,  # None = torch's default
}


class MaskWithReport(np.ndarray):
    """A mask that also says what the method did. It is an ordinary array to everything else."""

    report: dict[str, Any]

    def __new__(cls, mask: np.ndarray, report: dict[str, Any]) -> "MaskWithReport":
        result = np.asarray(mask).view(cls)
        result.report = report
        return result

    def __array_finalize__(self, source) -> None:
        self.report = getattr(source, "report", {})


def resolve_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    section = dict(parameters or {})
    unknown = sorted(str(key) for key in section if key not in DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown sam_med3d parameters {unknown}; allowed: {sorted(DEFAULT_PARAMETERS)}")
    params = {**DEFAULT_PARAMETERS, **section}

    if params["seed_zyx"] is None:
        raise ValueError("sam_med3d needs a click: set data.seed_zyx to a voxel inside the target.")
    if params["intensity_shift"] != AUTO:
        _number(params["intensity_shift"], f"intensity_shift (a number or '{AUTO}')")
    threshold = _number(params["threshold"], "threshold")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold={threshold!r}: expected a number between 0 and 1")
    return params


def prepare(values: np.ndarray, inside: np.ndarray, intensity_shift: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Shift the scan voxels, pad the rest with 0, then upstream's z-score (over voxels above 0,
    with torch's unbiased std, applied to the whole cube)."""
    prepared = np.array(values, dtype=np.float32)
    prepared[inside] += float(intensity_shift)
    scan_voxels = prepared[inside]
    if scan_voxels.size == 0:
        raise ValueError("The cube does not overlap the scan")
    prepared[~inside] = PADDING

    above = prepared > 0
    count = int(above.sum())
    if count < 2:
        raise ValueError("Nothing in the cube is above 0, so upstream's normalisation cannot be computed")
    mean, std = float(prepared[above].mean()), float(prepared[above].std(ddof=1))
    if std == 0:
        raise ValueError("The voxels above 0 in the cube all have one value, so there is nothing to normalise")
    report = {
        "method": "z-score over voxels above 0, as upstream",
        "intensity_shift": float(intensity_shift),
        "share_of_cube_above_0": count / prepared.size,
        "share_of_scan_voxels_above_0": float((scan_voxels > 0).mean()),
        "air_is_above_0": bool(scan_voxels.min() > 0),
        "scan_minimum_in_cube": float(scan_voxels.min()),
        "mean": mean,
        "std": std,
    }
    return (prepared - mean) / std, report


def auto_shift(values: np.ndarray) -> dict[str, Any]:
    """The `intensity_shift` that puts 0 where HU has it between air and soft tissue, from the box's
    values alone: Otsu's split, air and tissue as the medians of the two classes, and
    shift = -(air + 0.952 (tissue - air))."""
    from skimage.filters import threshold_otsu

    values = np.asarray(values, dtype=np.float64).ravel()
    split = float(threshold_otsu(values))
    air, tissue = float(np.median(values[values <= split])), float(np.median(values[values > split]))
    # Trigger: the model expects HU, where air and fat are below 0. A box with 1% or more of its voxels
    # below 0 already has that (HU: the tuna 39%, human organs 6-49%), and a shift would push its tissue
    # below 0 too (tuna Dice 0.58 -> 0.34). Only a box with (almost) nothing below 0 is shifted:
    # Oreo (mass density) and Gadus 0%, raw unsigned CT (HU + 1024) about 0%.
    share_below_0 = float((values < 0).mean())
    applied = share_below_0 < BELOW_0_TRIGGER
    shift = -(air + AIR_TO_ZERO * (tissue - air)) if applied else 0.0
    return {"otsu": split, "air": air, "tissue": tissue, "share_below_0": share_below_0, "applied": applied, "shift": float(shift)}


def _box(params: Mapping[str, Any], shape) -> tuple[slice, slice, slice]:
    if params["box_zyx"] is None:
        return tuple(slice(0, n) for n in shape)
    try:
        box = tuple(slice(int(a), int(b)) for a, b in params["box_zyx"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"box_zyx={params['box_zyx']!r}: expected [[start, stop]] for z, y and x") from error
    if len(box) != 3 or any(not 0 <= w.start < w.stop <= n for w, n in zip(box, shape)):
        raise ValueError(f"box_zyx={params['box_zyx']!r} does not fit in the volume {tuple(shape)}")
    return box


def segment(volume: Volume, parameters: Mapping[str, Any] | None = None) -> MaskWithReport:
    params = resolve_parameters(parameters)
    spacing = tuple(float(s) for s in volume.spacing_mm)
    shape = tuple(volume.shape)
    box = _box(params, shape)
    seed = tuple(int(v) for v in params["seed_zyx"])
    if any(not w.start <= v < w.stop for v, w in zip(seed, box)):
        raise ValueError(f"seed_zyx={seed} lies outside the volume's box {[[w.start, w.stop] for w in box]} (volume {shape})")
    click_mm = geometry.voxel_centre_mm(seed, spacing)

    box_low = tuple(w.start * s for w, s in zip(box, spacing))
    box_high = tuple(w.stop * s for w, s in zip(box, spacing))
    extent_mm = tuple(b - a for a, b in zip(box_low, box_high))
    cube = geometry.cube_on_box(box_low, box_high)
    click_xyz = geometry.network_click(click_mm, cube)

    # the cube is sampled from the scan around it, not from the box alone
    region = geometry.region_around(cube, shape, spacing)
    offset = tuple(w.start * s for w, s in zip(region, spacing))
    local_cube = geometry.Cube(tuple(c - o for c, o in zip(cube.corner_mm, offset)), cube.spacing_mm)
    data = volume.data[region]

    shift_rule = None
    shift = params["intensity_shift"]
    if shift == AUTO:
        shift_rule = auto_shift(volume.data[box])
        shift = shift_rule["shift"]

    smoothed, sigma = geometry.antialias(data, spacing, cube.spacing_mm)
    values, inside = geometry.sample(smoothed, spacing, local_cube)
    del smoothed
    prepared, normalisation = prepare(values, inside, shift)
    net = network.get_network(str(params["device"]), params["threads"])
    embedding = net.embed(geometry.to_network_axes(prepared))
    probability = geometry.to_network_axes(net.predict(embedding, click_xyz))
    cube_mask = probability > float(params["threshold"])
    mask = np.zeros(shape, dtype=np.uint8)
    mask[region] = geometry.paste_back(cube_mask, local_cube, data.shape, spacing)
    in_box = np.zeros(shape, dtype=bool)
    in_box[box] = True
    mask[~in_box] = 0  # the mask counts only inside the box

    report: dict[str, Any] = {
        "spacing_mm": cube.spacing_mm,
        "spacing_vs_trained": cube.spacing_mm / TRAINED_SPACING_MM,
        "side_mm": cube.side_mm,
        "box_extent_mm_zyx": list(extent_mm),
        "smoothing_sigma_voxels_zyx": list(sigma),
        "click_mm_zyx": list(click_mm),
        "click_in_cube_xyz": list(click_xyz),
        "threshold": float(params["threshold"]),
        "encoder_output_reused": bool(getattr(net, "last_was_cached", False)),
        "normalisation": normalisation,
        "intensity_shift_rule": shift_rule,
        "padding_share_of_cube": 1.0 - float(inside.mean()),
    }
    report["box_faces_reached"] = _faces_reached(mask[box])
    report["warnings"] = _warnings(report)
    return MaskWithReport(mask, report)


segment.needs_box = True  # the runner refuses a run without data.box
segment.takes_box = True  # the runner gives the whole scan and the box (box_zyx) instead of cropping to it


def _faces_reached(mask: np.ndarray) -> list[str]:
    """The faces of the box that the mask reaches: there the target may go on."""
    faces = []
    for axis in range(3):
        if np.take(mask, 0, axis=axis).any():
            faces.append("low " + "zyx"[axis])
        if np.take(mask, mask.shape[axis] - 1, axis=axis).any():
            faces.append("high " + "zyx"[axis])
    return faces


def _warnings(report: Mapping[str, Any]) -> list[str]:
    warnings = []
    if report["box_faces_reached"]:
        warnings.append(
            f"The mask reaches the box's {', '.join(report['box_faces_reached'])} face: the target may go on "
            "past the box and be cut off. Draw the box around the whole target."
        )
    if report["normalisation"]["air_is_above_0"]:
        warnings.append(
            f"Every scan voxel in the cube is above 0 (minimum {report['normalisation']['scan_minimum_in_cube']:.4g}), "
            "so the z-score also takes in the air. It behaves as in training only when air is below 0; "
            "the intensity_shift parameter changes this."
        )
    return warnings


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name}={value!r}: expected a number")
    return float(value)


__all__ = ["segment", "resolve_parameters", "prepare", "MaskWithReport", "assets"]
