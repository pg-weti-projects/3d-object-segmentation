"""
src/models/contrast_init.py

Lets the analytical methods start from the measured contrast (see
``src/models/contrast.py``) instead of from a small ball around the seed point.

A ball tells a method nothing about where the bladder ends, so Chan-Vese,
geodesic active contour and GrowCut each had to discover that on their own from
one point. With ``init_from: contrast`` they are handed:

* Chan-Vese - the intensity window of the bladder (``tmin``, ``tmax``), which is
  what it binarizes with; no change inside the method itself;
* geodesic active contour - the thresholded region as the initial level set;
* GrowCut - the inside of that region as object seeds and the tissue around it as
  background seeds.

The contrast is measured once, on the volume the method receives (in
semi-automatic mode: the contents of the box), and the regime it reports picks
the smoothing and the opening (``contrast.strategy_for``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from src.models import contrast as C
from src.models.grid import SeedPoint, WorkVolume
from src.models.roi import RoiResult

logger = logging.getLogger(__name__)

SEED_BALL = "seed_ball"
CONTRAST = "contrast"
INIT_CHOICES = (SEED_BALL, CONTRAST)

ESTIMATOR = "contrast"  # marks a RoiResult built here
SAME_STEP_TOL = 1e-6


def contrast_roi(
    volume: WorkVolume,
    seed: SeedPoint,
    *,
    sigma_mm: float | None = None,
    threshold_fraction: float | None = None,
    open_mm: float | None = None,
) -> RoiResult:
    """Measure the contrast and return it as a ``RoiResult``: ``roi_mask`` is the
    region the threshold gives, ``tmin``/``tmax`` the intensity window of the
    bladder. ``None`` for a setting means "take it from the regime"."""
    data = np.asarray(volume.data, dtype=np.float32)
    spacing = tuple(float(s) for s in volume.spacing)
    seed_voxel = tuple(int(v) for v in seed.voxel)
    # A work volume is always in the order of the scan (``to_work_volume`` undoes
    # the inversion that preprocessing may have applied, which is what
    # meta["invert_hu_undone"] records), so the bladder is the darker side here
    # even when the preprocessed volume was inverted.
    dark = True

    first = C.strategy_for(C.HIGH)
    estimate = C.estimate(
        data,
        spacing,
        seed_voxel,
        dark=dark,
        sigma_mm=first.sigma_mm if sigma_mm is None else float(sigma_mm),
        threshold_fraction=(
            first.threshold_fraction if threshold_fraction is None else float(threshold_fraction)
        ),
    )
    strategy = C.strategy_for(estimate.regime)
    sigma = strategy.sigma_mm if sigma_mm is None else float(sigma_mm)
    fraction = strategy.threshold_fraction if threshold_fraction is None else float(threshold_fraction)
    opening = strategy.open_mm if open_mm is None else float(open_mm)

    smooth = C.smooth_mm(data, spacing, sigma)
    if estimate.regime != C.HIGH:
        # A weak difference needs the harder smoothing before the levels mean
        # anything, so the measurement is repeated on the smoothed volume.
        estimate = C.estimate(
            data,
            spacing,
            seed_voxel,
            dark=dark,
            sigma_mm=sigma,
            threshold_fraction=fraction,
            smooth=smooth,
        )
    mask = C.region(smooth, spacing, estimate, open_mm=opening, use_hull=strategy.use_hull)

    low, high = float(data.min()), float(data.max())
    tmin, tmax = (low, estimate.threshold) if dark else (estimate.threshold, high)
    meta: dict[str, Any] = {
        "mode": CONTRAST,
        "contrast": estimate.as_dict(),
        "strategy": {
            "sigma_mm": sigma,
            "threshold_fraction": fraction,
            "open_mm": opening,
            "use_hull": strategy.use_hull,
        },
        "region_voxels": int(mask.sum()),
    }
    logger.info(
        "init_from=contrast: regime %s, window [%.4f, %.4f], region %d voxels",
        estimate.regime,
        tmin,
        tmax,
        int(mask.sum()),
    )
    return RoiResult(
        seed=seed,
        roi_mask=mask,
        tmin=float(tmin),
        tmax=float(tmax),
        estimator=ESTIMATOR,
        meta=meta,
    )


def candidates_for(volume: WorkVolume, roi: RoiResult) -> np.ndarray | None:
    """Everything inside the body that is on the bladder's side of the threshold,
    for a method that does its own search from a binary image (Chan-Vese). Built
    from the contrast ``RoiResult``, with the settings it recorded. ``None`` when
    that ``RoiResult`` was not built here."""
    if roi.estimator != ESTIMATOR:
        return None
    measured = roi.meta.get("contrast") or {}
    strategy = roi.meta.get("strategy") or {}
    data = np.asarray(volume.data, dtype=np.float32)
    spacing = tuple(float(s) for s in volume.spacing)
    smooth = C.smooth_mm(data, spacing, float(strategy.get("sigma_mm", 0.8)))
    estimate = C.Contrast(
        object_level=float(measured["object_level"]),
        tissue_level=float(measured["tissue_level"]),
        threshold=float(measured["threshold"]),
        gap=float(measured["gap"]),
        separability=float(measured["separability"]),
        cnr=float(measured["cnr"]),
        regime=str(measured["regime"]),
        bladder_is_dark=bool(measured["bladder_is_dark"]),
        seed_zyx=tuple(int(v) for v in measured["seed_zyx"]),  # type: ignore[arg-type]
        seed_radius_mm=float(measured["seed_radius_mm"]),
    )
    return C.candidates(smooth, estimate, use_hull=bool(strategy.get("use_hull", True)))


def region_on_grid(roi: RoiResult, volume_shape, domain, step, grid_shape) -> np.ndarray | None:
    """The region of a contrast ``RoiResult`` cut to the compute domain and put on
    the compute grid (nearest neighbour, mirroring ``domain.compute_grid``).
    ``None`` when this ``RoiResult`` carries no usable region."""
    if roi.estimator != ESTIMATOR:
        return None
    mask = np.asarray(roi.roi_mask, dtype=bool)
    if mask.shape != tuple(int(n) for n in volume_shape) or not mask.any():
        return None
    crop = mask[domain.slices]
    step = np.asarray(step, dtype=float)
    if np.all(np.abs(step - 1.0) < SAME_STEP_TOL):
        on_grid = crop
    else:
        on_grid = (
            ndi.affine_transform(
                crop.astype(np.uint8),
                step,
                offset=0.0,
                output_shape=tuple(int(n) for n in grid_shape),
                order=0,
                mode="nearest",
            )
            > 0
        )
    if not on_grid.any():
        logger.warning("init_from=contrast: the region is empty on the compute grid - falling back")
        return None
    return on_grid


def growcut_seeds(
    region: np.ndarray,
    image: np.ndarray,
    roi: RoiResult,
    *,
    margin_voxels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Object and background seeds for GrowCut from a contrast region: the inside
    of the region (eroded, so its boundary stays free) against the tissue outside
    it (kept a margin away, so the boundary is not pinned down either)."""
    structure = ndi.generate_binary_structure(region.ndim, 1)
    inside = ndi.binary_erosion(region, structure=structure, iterations=margin_voxels, border_value=0)
    if not inside.any():
        inside = region
    near = ndi.binary_dilation(region, structure=structure, iterations=margin_voxels, border_value=0)

    threshold = float(roi.tmax if roi.meta.get("contrast", {}).get("bladder_is_dark", True) else roi.tmin)
    dark = bool(roi.meta.get("contrast", {}).get("bladder_is_dark", True))
    tissue = image > threshold if dark else image < threshold
    outside = tissue & ~near
    return inside, outside
