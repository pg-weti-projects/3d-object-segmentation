from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

from src.core.volume import Volume

logger = logging.getLogger(__name__)

DEFAULT_PARAMETERS: dict[str, Any] = {
    "sigma_mm": 0.8,  # Gaussian smoothing; 0 = none
    "seed_zyx": None,  # voxel inside the bladder; None = centre of the volume
    "seed_radius_mm": None,  # ball for the object level; None = 8% of the smallest extent
    "threshold_fraction": 0.4,  # 0..1, share of the way from the object to the tissue level
    "open_mm": 1.5,  # opening radius; 0 = none. Use at least one voxel to have any effect
    "bladder_is_dark": True,
    "use_hull": True,  # limit candidates to the convex hull of the tissue
}


def resolve_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    section = dict(parameters or {})
    unknown = sorted(str(key) for key in section if key not in DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown box_region parameters {unknown}; allowed: {sorted(DEFAULT_PARAMETERS)}")
    params = {**DEFAULT_PARAMETERS, **section}

    for key in ("sigma_mm", "open_mm"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key}={value!r}: expected a number >= 0")
    radius = params["seed_radius_mm"]
    if radius is not None and (
        isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0
    ):
        raise ValueError(f"seed_radius_mm={radius!r}: expected a number > 0 or null")
    fraction = params["threshold_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError(f"threshold_fraction={fraction!r}: expected a number between 0 and 1")
    seed = params["seed_zyx"]
    if seed is not None and (
        len(seed) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in seed)
    ):
        raise ValueError(f"seed_zyx={seed!r}: expected three integer indices (z, y, x) or null")
    params["bladder_is_dark"] = bool(params["bladder_is_dark"])
    params["use_hull"] = bool(params["use_hull"])
    return params


def segment(volume: Volume, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
    params = resolve_parameters(parameters)
    data = np.asarray(volume.data, dtype=np.float32)
    spacing = tuple(float(s) for s in volume.spacing_mm)
    if not np.isfinite(data).all():
        raise ValueError("Volume contains NaN or infinite values")
    if data.max() == data.min():
        raise ValueError("Volume is uniform - nothing to segment")
    dark = params["bladder_is_dark"]

    smooth = data
    if params["sigma_mm"] > 0:
        smooth = ndi.gaussian_filter(data, [params["sigma_mm"] / s for s in spacing])

    seed = _seed(params["seed_zyx"], data.shape)
    extent = min(n * s for n, s in zip(data.shape, spacing))
    radius = params["seed_radius_mm"] or 0.08 * extent
    near_seed = _within_mm(data.shape, seed, spacing, radius)
    object_level = float(np.median(smooth[near_seed]))

    split = threshold_otsu(smooth)
    tissue = smooth > split if dark else smooth < split
    tissue_level = float(np.median(smooth[tissue])) if tissue.any() else float(np.median(smooth))
    contrast = (tissue_level - object_level) if dark else (object_level - tissue_level)
    if contrast <= 0:
        logger.warning(
            "box_region: the seed region (%.4g) is not %s than the tissue (%.4g) - empty mask",
            object_level,
            "darker" if dark else "brighter",
            tissue_level,
        )
        return np.zeros(data.shape, dtype=np.uint8)

    threshold = object_level + params["threshold_fraction"] * (tissue_level - object_level)
    candidates = smooth < threshold if dark else smooth > threshold
    if params["use_hull"]:
        body = _largest_component(tissue)
        if body.any():
            candidates &= _axis_hulls(body)
    if params["open_mm"] > 0:
        candidates = ndi.binary_opening(candidates, structure=_ellipsoid(params["open_mm"], spacing))

    component = _component_at(candidates, seed, spacing)
    if component is None:
        logger.warning("box_region: no region found around the seed - empty mask")
        return np.zeros(data.shape, dtype=np.uint8)
    return ndi.binary_fill_holes(component).astype(np.uint8)


def _seed(seed_zyx, shape) -> tuple[int, int, int]:
    if seed_zyx is None:
        return tuple(n // 2 for n in shape)
    seed = tuple(int(v) for v in seed_zyx)
    if any(not 0 <= v < n for v, n in zip(seed, shape)):
        raise ValueError(f"seed_zyx={seed} lies outside the volume {tuple(shape)}")
    return seed


def _within_mm(shape, center, spacing, radius_mm) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    squared = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, center, spacing))
    inside = squared <= radius_mm**2
    inside[center] = True
    return inside


def _ellipsoid(radius_mm: float, spacing) -> np.ndarray:
    """Structuring element of a given radius in mm (odd shape, centre voxel set)."""
    half = [max(int(np.ceil(radius_mm / s)), 0) for s in spacing]
    grids = np.ogrid[tuple(slice(-n, n + 1) for n in half)]
    return sum((g * s / radius_mm) ** 2 for g, s in zip(grids, spacing)) <= 1.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndi.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _axis_hulls(body: np.ndarray) -> np.ndarray:
    """Intersection of the 2D convex hulls of `body` taken slice by slice along each axis."""
    result = np.ones(body.shape, dtype=bool)
    for axis in range(3):
        hull = np.zeros(body.shape, dtype=bool)
        for index in range(body.shape[axis]):
            plane = tuple(index if a == axis else slice(None) for a in range(3))
            if body[plane].any():
                hull[plane] = convex_hull_image(body[plane])
        result &= hull
    return result


def _component_at(mask: np.ndarray, seed, spacing) -> np.ndarray | None:
    labels, count = ndi.label(mask)
    if count == 0:
        return None
    label = labels[seed]
    if label == 0:  # the seed fell on a removed voxel: take the nearest component
        nearest = ndi.distance_transform_edt(labels == 0, sampling=spacing, return_indices=True)[1]
        label = labels[tuple(nearest[(slice(None), *seed)])]
    return labels == label
