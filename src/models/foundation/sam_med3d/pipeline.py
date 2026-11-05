"""
segment(volume, parameters): one click and a box, one cube, one mask on the volume's grid.

The cube is centred on the box and scaled so that the box (plus box_margin) fills it. The box
only sets what the cube covers; it is never a prompt, and the scan is not cropped to it.
A cube centred on the click alone was tried and dropped: on upstream's human CT it failed the
protocol's control (see the README).

Nothing is post-processed: no component filtering, no sliding window. A target larger than
the cube is cut off, and the report says so.

Upstream's normalisation is a z-score over the voxels above 0, which behaves as in training
only when air is below 0, as it is in HU. On scans above 0 everywhere it also takes in the
air; that is reported, and `intensity_shift` is there as an explicit way to change it.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.core.box import Box
from src.core.volume import Volume

from . import assets, cube as geometry, network

logger = logging.getLogger(__name__)

PADDING_MINIMUM = "minimum"
TRAINED_SPACING_MM = 1.5  # reported against, not used: the box sets the spacing

DEFAULT_PARAMETERS: dict[str, Any] = {
    "seed_zyx": None,  # required: the click, as a voxel of the volume
    "box": None,  # required: {"min_mm": [z, y, x], "max_mm": [z, y, x]}; the runner passes data.box here
    "box_margin": 0.0,  # share of the box's longest side added to the cube's side
    "padding": 0.0,  # value outside the scan, as upstream, or "minimum" = lowest scan value in the cube
    "threshold": 0.5,  # on the sigmoid, as upstream
    "clamp": None,  # [low, high] applied to the scan first; None = as upstream's validation
    "intensity_shift": 0.0,  # added to the scan after the clamp and before the z-score; 0 = as upstream
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

    if params["box"] is None:
        raise ValueError("sam_med3d needs a box (data.box) around the target, and none was given.")
    if params["seed_zyx"] is None:
        raise ValueError("sam_med3d needs a click: set data.seed_zyx to a voxel inside the target.")
    _number(params["box_margin"], "box_margin", above=-1)
    _number(params["intensity_shift"], "intensity_shift")
    threshold = _number(params["threshold"], "threshold")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold={threshold!r}: expected a number between 0 and 1")
    if params["padding"] != PADDING_MINIMUM:
        _number(params["padding"], f"padding (a number or '{PADDING_MINIMUM}')")
    clamp = params["clamp"]
    if clamp is not None and (len(clamp) != 2 or not float(clamp[0]) < float(clamp[1])):
        raise ValueError(f"clamp={clamp!r}: expected [low, high] with low < high")
    return params


def prepare(values: np.ndarray, inside: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp and shift the scan voxels, pad the rest, then upstream's z-score (over voxels above 0,
    with torch's unbiased std, applied to the whole cube)."""
    prepared = np.array(values, dtype=np.float32)
    if params["clamp"] is not None:
        prepared[inside] = np.clip(prepared[inside], float(params["clamp"][0]), float(params["clamp"][1]))
    prepared[inside] += float(params["intensity_shift"])
    scan_voxels = prepared[inside]
    if scan_voxels.size == 0:
        raise ValueError("The cube does not overlap the scan")
    padding = float(scan_voxels.min()) if params["padding"] == PADDING_MINIMUM else float(params["padding"])
    prepared[~inside] = padding

    above = prepared > 0
    count = int(above.sum())
    if count < 2:
        raise ValueError("Nothing in the cube is above 0, so upstream's normalisation cannot be computed")
    mean, std = float(prepared[above].mean()), float(prepared[above].std(ddof=1))
    if std == 0:
        raise ValueError("The voxels above 0 in the cube all have one value, so there is nothing to normalise")
    report = {
        "method": "z-score over voxels above 0, as upstream",
        "clamp": params["clamp"],
        "intensity_shift": float(params["intensity_shift"]),
        "padding_value": padding,
        "share_of_cube_above_0": count / prepared.size,
        "share_of_scan_voxels_above_0": float((scan_voxels > 0).mean()),
        "air_is_above_0": bool(scan_voxels.min() > 0),
        "scan_minimum_in_cube": float(scan_voxels.min()),
        "mean": mean,
        "std": std,
    }
    return (prepared - mean) / std, report


def segment(volume: Volume, parameters: Mapping[str, Any] | None = None) -> MaskWithReport:
    params = resolve_parameters(parameters)
    spacing = tuple(float(s) for s in volume.spacing_mm)
    seed = tuple(int(v) for v in params["seed_zyx"])
    if any(not 0 <= v < n for v, n in zip(seed, volume.shape)):
        raise ValueError(f"seed_zyx={seed} lies outside the volume {tuple(volume.shape)}")
    click_mm = geometry.voxel_centre_mm(seed, spacing)

    box = Box.from_mapping(dict(params["box"]))
    box_mm = (box.min_mm, box.max_mm)
    cube = geometry.cube_on_box(box.min_mm, box.max_mm, float(params["box_margin"]))
    click_xyz = geometry.network_click(click_mm, cube)

    values, inside = geometry.sample(volume.data, spacing, cube)
    prepared, normalisation = prepare(values, inside, params)
    net = network.get_network(str(params["device"]), params["threads"])
    embedding = net.embed(geometry.to_network_axes(prepared))
    probability = geometry.to_network_axes(net.predict(embedding, click_xyz))
    cube_mask = probability > float(params["threshold"])
    mask = geometry.paste_back(cube_mask, cube, volume.shape, spacing)

    report: dict[str, Any] = {
        "spacing_mm": cube.spacing_mm,
        "spacing_vs_trained": cube.spacing_mm / TRAINED_SPACING_MM,
        "side_mm": cube.side_mm,
        "click_mm_zyx": list(click_mm),
        "click_in_cube_xyz": list(click_xyz),
        "threshold": float(params["threshold"]),
        "post_processing": None,
        "encoder_output_reused": bool(getattr(net, "last_was_cached", False)),
        "normalisation": normalisation,
        **geometry.coverage(cube, volume.shape, spacing, box_mm),
    }
    report["faces_reached_inside_scan"] = geometry.faces_touched_inside_scan(cube_mask, cube, volume.shape, spacing)
    report["warnings"] = _warnings(report)
    return MaskWithReport(mask, report)


segment.handles_box = True  # the runner passes data.box as a parameter and does not crop to it


def _warnings(report: Mapping[str, Any]) -> list[str]:
    warnings = []
    if report.get("box_share_in_cube", 1.0) < 1.0:
        warnings.append(f"Only {100 * report['box_share_in_cube']:.1f}% of the box fits in the cube.")
    if report["faces_reached_inside_scan"]:
        warnings.append(
            f"The mask reaches the cube's {', '.join(report['faces_reached_inside_scan'])} face inside the scan: "
            f"the target may be cut off at the cube ({report['side_mm']:.0f} mm a side). Draw the box around the whole target."
        )
    if report["normalisation"]["air_is_above_0"]:
        warnings.append(
            f"Every scan voxel in the cube is above 0 (minimum {report['normalisation']['scan_minimum_in_cube']:.4g}), "
            "so the z-score also takes in the air. It behaves as in training only when air is below 0; "
            "the intensity_shift parameter changes this."
        )
    return warnings


def _number(value: Any, name: str, above: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name}={value!r}: expected a number")
    if above is not None and value <= above:
        raise ValueError(f"{name}={value!r}: expected a number above {above}")
    return float(value)


__all__ = ["segment", "resolve_parameters", "prepare", "MaskWithReport", "assets"]
