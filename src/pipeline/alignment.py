"""
src/pipeline/alignment.py

Does a reference mask sit on its scan the right way round? A mirrored mask still
gives metrics, just meaningless ones, so this compares the mask as given with the
same mask flipped along each axis, using only what the image shows:

    edge        gradient on the mask boundary / on surfaces moved in and out
    separation  share of the intensity variance around the boundary that
                inside / outside explains

A flip counts against the mask only when it is better on both, because neither
is enough alone: in a small fish almost any surface crosses an edge, and in the
Bluefin tuna extreme values swamp the variance. Flips of two or three axes are
not tried; on the tuna one of them scores as high as the correct frame.

The margins come from three fish, so this warns rather than refuses.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MAX_VOXELS = 4_000_000  # the check is coarse on purpose: it has to fit in memory next to the scan
CLIP_PERCENTILES = (0.1, 99.9)  # a few extreme voxels (metal, padding) must not decide the gradient
BETTER_BY = 1.1  # a flip must beat the mask as given by this factor on both measures
WEAK_EDGE = 1.1  # below this, the boundary is no stronger than the surfaces around it
AXES = ("z", "y", "x")

OK, MISMATCH, UNVERIFIED = "ok", "mismatch", "unverified"


def check_alignment(data: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """``status`` is ``ok``, ``mismatch`` (a flip fits the image better) or ``unverified``
    (the image cannot confirm the orientation); ``message`` says what to do about it."""
    data, mask = np.asarray(data), np.asarray(mask) != 0
    if data.shape != mask.shape:
        raise ValueError(f"scan {data.shape} and mask {mask.shape} are not on the same grid")

    step = _stride(data.shape, spacing)
    view = tuple(slice(None, None, k) for k in step)
    reduced = mask[view]
    if not reduced.any():
        return {"status": UNVERIFIED, "message": "the reference mask is empty, so its orientation cannot be checked"}

    image = np.asarray(data[view], dtype=np.float32)
    image = np.clip(image, *np.percentile(image, CLIP_PERCENTILES))
    gradient = ndi.gaussian_gradient_magnitude(image, sigma=1.0)
    voxel = tuple(s * k for s, k in zip(spacing, step))

    edge, separation = _measures(image, gradient, reduced, voxel)
    flips = {}
    for axis, name in enumerate(AXES):
        flipped_edge, flipped_separation = _measures(image, gradient, np.flip(reduced, axis), voxel)
        flips[name] = {"edge": flipped_edge, "separation": flipped_separation}

    better = [
        name
        for name, flip in flips.items()
        if flip["edge"] > BETTER_BY * edge and flip["separation"] > BETTER_BY * separation
    ]
    report = {"status": OK, "message": None, "edge": edge, "separation": separation, "flips": flips}
    if better:
        best = max(better, key=lambda name: flips[name]["edge"] * flips[name]["separation"])
        report["status"] = MISMATCH
        report["message"] = (
            f"the reference mask looks mirrored along {best} against this scan: flipped, its boundary fits the "
            f"image better (edge {flips[best]['edge']:.2f} against {edge:.2f}, separation "
            f"{flips[best]['separation']:.3f} against {separation:.3f}). Metrics are not valid until this is resolved."
        )
    elif edge < WEAK_EDGE:
        report["status"] = UNVERIFIED
        report["message"] = (
            f"the image shows no edge on the reference mask boundary (edge {edge:.2f}), so its orientation "
            "cannot be confirmed from the image."
        )
    return report


def _stride(shape: tuple[int, ...], spacing: tuple[float, ...]) -> tuple[int, int, int]:
    """Steps along each axis that bring the volume under MAX_VOXELS, finest axis first."""
    step = [1, 1, 1]
    while np.prod([n // k for n, k in zip(shape, step)]) > MAX_VOXELS:
        axis = int(np.argmin([s * k for s, k in zip(spacing, step)]))
        step[axis] += 1
    return step[0], step[1], step[2]


def _measures(image: np.ndarray, gradient: np.ndarray, mask: np.ndarray, voxel: tuple[float, ...]) -> tuple[float, float]:
    width = max(voxel)
    moved = 3 * width
    found = np.argwhere(mask)
    pad = [int(np.ceil((moved + 2 * width) / s)) + 1 for s in voxel]
    low, high = np.maximum(found.min(0) - pad, 0), np.minimum(found.max(0) + 1 + np.array(pad), mask.shape)
    window = tuple(slice(a, b) for a, b in zip(low, high))
    inside, near_gradient, near_image = mask[window], gradient[window], image[window]
    distance = ndi.distance_transform_edt(~inside, sampling=voxel) - ndi.distance_transform_edt(inside, sampling=voxel)

    boundary = np.abs(distance) <= width
    surfaces = [near_gradient[np.abs(distance - offset) <= width / 2] for offset in (-moved, moved)]
    surfaces = [values.mean() for values in surfaces if values.size]
    around = float(np.mean(surfaces)) if surfaces else 0.0
    edge = float(near_gradient[boundary].mean() / around) if boundary.any() and around > 0 else 0.0

    within, shell = near_image[inside], near_image[(distance > 0) & (distance <= moved)]
    both = np.concatenate([within, shell])
    variance = float(both.var())
    separation = 0.0
    if variance > 0 and shell.size:
        between = within.size * (within.mean() - both.mean()) ** 2 + shell.size * (shell.mean() - both.mean()) ** 2
        separation = float(between / (both.size * variance))
    return edge, separation
