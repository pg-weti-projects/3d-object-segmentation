"""
segment(volume, parameters): one click on the contents of a box, one cube, one mask on the volume's grid.

The runner crops the scan to the user's 3D box, as for the other box methods, and passes only
the box's contents. The cube is centred on them and scaled so that their longest side fills it;
the shorter sides are padded. Before the scan is shrunk onto a coarser cube it is smoothed, so
that fine detail does not alias. The box sets only what the cube covers and is never a model
prompt. A cube centred on the click alone was tried and dropped: on upstream's human CT it
failed the protocol's control (see the README).

Nothing is post-processed: no component filtering, no sliding window. A target that reaches the
edge of the box may be cut off, and the report says so.

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

from src.core.volume import Volume

from . import assets, cube as geometry, network

logger = logging.getLogger(__name__)

PADDING_MINIMUM = "minimum"
TRAINED_SPACING_MM = 1.5  # reported against, not used: the box sets the spacing

DEFAULT_PARAMETERS: dict[str, Any] = {
    "seed_zyx": None,  # required: the click, as a voxel of the volume it is given (the box's contents)
    "padding": 0.0,  # value outside the box, as upstream, or "minimum" = lowest scan value in the cube
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

    if params["seed_zyx"] is None:
        raise ValueError("sam_med3d needs a click: set data.seed_zyx to a voxel inside the target.")
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

    extent_mm = tuple(n * s for n, s in zip(volume.shape, spacing))
    cube = geometry.cube_on_box((0.0, 0.0, 0.0), extent_mm)
    click_xyz = geometry.network_click(click_mm, cube)

    smoothed, sigma = geometry.antialias(volume.data, spacing, cube.spacing_mm)
    values, inside = geometry.sample(smoothed, spacing, cube)
    del smoothed
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
        "box_extent_mm_zyx": list(extent_mm),
        "smoothing_sigma_voxels_zyx": list(sigma),
        "click_mm_zyx": list(click_mm),
        "click_in_cube_xyz": list(click_xyz),
        "threshold": float(params["threshold"]),
        "post_processing": None,
        "encoder_output_reused": bool(getattr(net, "last_was_cached", False)),
        "normalisation": normalisation,
        "view_mm": {"min_zyx": [0.0, 0.0, 0.0], "max_zyx": list(extent_mm)},  # all of what it was given
        "padding_share_of_cube": 1.0 - float(np.prod([e / cube.side_mm for e in extent_mm])),
    }
    report["box_faces_reached"] = _faces_reached(mask)
    report["warnings"] = _warnings(report)
    return MaskWithReport(mask, report)


segment.needs_box = True  # the runner refuses a run without data.box, then crops to it as for the other box methods


def _faces_reached(mask: np.ndarray) -> list[str]:
    """The faces of the volume it was given (the box) that the mask reaches: there the target may go on."""
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


def _number(value: Any, name: str, above: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name}={value!r}: expected a number")
    if above is not None and value <= above:
        raise ValueError(f"{name}={value!r}: expected a number above {above}")
    return float(value)


__all__ = ["segment", "resolve_parameters", "prepare", "MaskWithReport", "assets"]
