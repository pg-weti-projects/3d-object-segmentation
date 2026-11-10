"""
src/models/contrast.py

How well the swim bladder stands out from the surrounding tissue, and the region
that follows from it. Shared by ``box_region`` and by the analytical methods
(Chan-Vese, geodesic active contour, GrowCut), so that they all start from the
same measurement instead of from a small ball around the seed point.

The measurement is made inside the volume it is given, which in semi-automatic
mode is the content of the box (see ``src/core/box.py``):

* the *object level* is the median in a small ball around the seed;
* the *tissue level* is the median of the tissue class (Otsu split);
* the *threshold* lies ``threshold_fraction`` of the way between them, which is
  also where partial-volume voxels at the edge of a gas pocket fall;
* *separability* is the gap between the two levels as a share of the dynamic
  range of the region, and *CNR* the same gap in units of the noise at the seed.

``separability`` decides the regime, because it is free of the unit and of the
offset of the scan (density in kg/m³, HU, or intensities rescaled to [0, 1]):

* ``high`` - gas, or any bladder that a threshold alone separates (Oreo);
* ``low``  - a bladder that differs from the tissue by little, where a threshold
  follows the noise and the shape has to carry more weight (Gadus);
* ``none`` - no usable difference; the caller should warn instead of pretending
  that the result means something (Bluefin tuna).

The bladder is expected to be darker than the tissue (gas, or fluid of a lower
density); pass ``dark=False`` for the opposite, e.g. after ``invert_hu``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

logger = logging.getLogger(__name__)

# Regime boundaries on `separability`, measured on the three scans of the
# project (see docs/contrast.md). They are deliberately on round numbers: with
# three species there is not enough data to place them more precisely.
HIGH_SEPARABILITY = 0.30
LOW_SEPARABILITY = 0.05

HIGH = "high"
LOW = "low"
NONE = "none"
REGIMES = (HIGH, LOW, NONE)

MAD_TO_SIGMA = 1.4826  # median absolute deviation -> standard deviation, normal case
SEED_RADIUS_FRACTION = 0.08  # of the smallest extent of the region, when no radius is given


@dataclass(frozen=True)
class Contrast:
    """The outcome of the measurement. Levels are in the units of the volume."""

    object_level: float
    tissue_level: float
    threshold: float
    gap: float
    separability: float
    cnr: float
    regime: str
    bladder_is_dark: bool
    seed_zyx: tuple[int, int, int]
    seed_radius_mm: float
    notes: tuple[str, ...] = ()

    @property
    def separated(self) -> bool:
        """True when the bladder and the tissue differ in the expected direction.
        A region can be found whenever this holds, even in the ``none`` regime."""
        return self.gap > 0

    @property
    def usable(self) -> bool:
        """True when the contrast is large enough for the result to be trusted."""
        return self.regime != NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_level": self.object_level,
            "tissue_level": self.tissue_level,
            "threshold": self.threshold,
            "gap": self.gap,
            "separability": self.separability,
            "cnr": self.cnr,
            "regime": self.regime,
            "bladder_is_dark": self.bladder_is_dark,
            "seed_zyx": list(self.seed_zyx),
            "seed_radius_mm": self.seed_radius_mm,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Strategy:
    """How to look for the bladder in a given regime. ``estimate`` measures the
    contrast with the ``high`` settings first; when the regime turns out to be
    lower, the measurement is repeated with the settings below, because a weak
    difference has to be smoothed harder before a threshold means anything."""

    sigma_mm: float
    threshold_fraction: float
    open_mm: float
    use_hull: bool


STRATEGIES: dict[str, Strategy] = {
    # Gas, or any bladder a threshold separates on its own.
    HIGH: Strategy(sigma_mm=0.8, threshold_fraction=0.4, open_mm=1.5, use_hull=True),
    # A few percent of the dynamic range: smooth harder, cut thin bridges with a
    # larger opening, and keep the threshold close to the bladder so that noise
    # on the tissue side is not swallowed.
    LOW: Strategy(sigma_mm=1.6, threshold_fraction=0.35, open_mm=2.0, use_hull=True),
    # Nothing to separate; the same settings as `low`, and the caller warns.
    NONE: Strategy(sigma_mm=1.6, threshold_fraction=0.35, open_mm=2.0, use_hull=True),
}


def strategy_for(regime: str) -> Strategy:
    if regime not in STRATEGIES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {list(REGIMES)}")
    return STRATEGIES[regime]


def smooth_mm(data: np.ndarray, spacing, sigma_mm: float) -> np.ndarray:
    """Gaussian smoothing with the same radius in mm along every axis."""
    data = np.asarray(data, dtype=np.float32)
    if sigma_mm <= 0:
        return data
    return ndi.gaussian_filter(data, [sigma_mm / float(s) for s in spacing])


def resolve_seed(seed_zyx, shape) -> tuple[int, int, int]:
    """The seed as integer indices; the centre of the region when none is given."""
    if seed_zyx is None:
        return tuple(int(n) // 2 for n in shape)  # type: ignore[return-value]
    seed = tuple(int(v) for v in seed_zyx)
    if len(seed) != 3 or any(not 0 <= v < n for v, n in zip(seed, shape)):
        raise ValueError(f"seed {seed} lies outside the volume {tuple(int(n) for n in shape)}")
    return seed  # type: ignore[return-value]


def estimate(
    data: np.ndarray,
    spacing,
    seed_zyx=None,
    *,
    dark: bool = True,
    sigma_mm: float = 0.8,
    seed_radius_mm: float | None = None,
    threshold_fraction: float = 0.4,
    smooth: np.ndarray | None = None,
) -> Contrast:
    """Measure the contrast between the bladder at the seed and the tissue around it.

    ``smooth`` can be passed in when the caller has already smoothed the data
    with ``smooth_mm`` and the same ``sigma_mm``, to avoid doing it twice.
    """
    data = np.asarray(data, dtype=np.float32)
    spacing = tuple(float(s) for s in spacing)
    if not np.isfinite(data).all():
        raise ValueError("Volume contains NaN or infinite values")
    if data.max() == data.min():
        raise ValueError("Volume is uniform - nothing to segment")
    if not 0 < threshold_fraction < 1:
        raise ValueError(f"threshold_fraction={threshold_fraction!r}: expected a number between 0 and 1")

    smoothed = smooth_mm(data, spacing, sigma_mm) if smooth is None else np.asarray(smooth, np.float32)
    seed = resolve_seed(seed_zyx, data.shape)
    extent = min(n * s for n, s in zip(data.shape, spacing))
    radius = float(seed_radius_mm) if seed_radius_mm else SEED_RADIUS_FRACTION * extent

    near_seed = within_mm(data.shape, seed, spacing, radius)
    object_level = float(np.median(smoothed[near_seed]))

    split = threshold_otsu(smoothed)
    tissue = smoothed > split if dark else smoothed < split
    tissue_level = float(np.median(smoothed[tissue])) if tissue.any() else float(np.median(smoothed))

    gap = (tissue_level - object_level) if dark else (object_level - tissue_level)
    low, high = (float(v) for v in np.percentile(smoothed, [1.0, 99.0]))
    span = high - low
    separability = float(np.clip(gap / span, 0.0, 1.0)) if span > 0 else 0.0

    raw_near_seed = data[near_seed]
    noise = MAD_TO_SIGMA * float(np.median(np.abs(raw_near_seed - np.median(raw_near_seed))))
    cnr = float(gap / noise) if noise > 0 else float("inf")

    notes: list[str] = []
    if gap <= 0:
        regime = NONE
        notes.append(
            f"the seed region ({object_level:.4g}) is not "
            f"{'darker' if dark else 'brighter'} than the tissue ({tissue_level:.4g})"
        )
    elif separability >= HIGH_SEPARABILITY:
        regime = HIGH
    elif separability >= LOW_SEPARABILITY:
        regime = LOW
        notes.append(
            f"low contrast ({100 * separability:.1f}% of the dynamic range, CNR {cnr:.1f}): "
            "the boundary is driven by the shape rather than by the threshold"
        )
    else:
        regime = NONE
        notes.append(
            f"contrast too low to segment ({100 * separability:.1f}% of the dynamic range, "
            f"CNR {cnr:.1f}): the bladder is nearly indistinguishable from the tissue"
        )

    threshold = object_level + threshold_fraction * (tissue_level - object_level)
    contrast = Contrast(
        object_level=object_level,
        tissue_level=tissue_level,
        threshold=threshold,
        gap=float(gap),
        separability=separability,
        cnr=cnr,
        regime=regime,
        bladder_is_dark=bool(dark),
        seed_zyx=seed,
        seed_radius_mm=radius,
        notes=tuple(notes),
    )
    logger.info(
        "contrast: object %.4g, tissue %.4g, threshold %.4g, separability %.3f, CNR %.1f, regime %s",
        object_level,
        tissue_level,
        threshold,
        separability,
        cnr,
        regime,
    )
    for note in notes:
        logger.warning("contrast: %s", note)
    return contrast


def candidates(
    smoothed: np.ndarray,
    contrast: Contrast,
    *,
    use_hull: bool = True,
) -> np.ndarray:
    """Everything below (or above) the threshold, limited to the body.

    The limit is the intersection of the per-axis convex hulls of the tissue, and
    it is what keeps gas that is open to the air outside the fish from being
    followed outwards. Methods that look for the bladder on their own (Chan-Vese
    binarizes per slice) use this instead of a plain threshold, which on a cropped
    scan also catches the air around the body."""
    smoothed = np.asarray(smoothed, dtype=np.float32)
    if not contrast.separated:
        return np.zeros(smoothed.shape, dtype=bool)
    dark = contrast.bladder_is_dark
    found = smoothed < contrast.threshold if dark else smoothed > contrast.threshold
    if use_hull:
        split = threshold_otsu(smoothed)
        body = largest_component(smoothed > split if dark else smoothed < split)
        if body.any():
            found &= axis_hulls(body)
    return found


def region(
    smoothed: np.ndarray,
    spacing,
    contrast: Contrast,
    *,
    open_mm: float = 1.5,
    use_hull: bool = True,
) -> np.ndarray:
    """The region that follows from the threshold: the candidates, opened, then the
    component at the seed with its holes filled. An empty mask means nothing was
    found."""
    smoothed = np.asarray(smoothed, dtype=np.float32)
    spacing = tuple(float(s) for s in spacing)
    if not contrast.separated:
        return np.zeros(smoothed.shape, dtype=bool)

    found = candidates(smoothed, contrast, use_hull=use_hull)
    if open_mm > 0:
        found = ndi.binary_opening(found, structure=ellipsoid(open_mm, spacing))

    component = component_at(found, contrast.seed_zyx, spacing)
    if component is None:
        logger.warning("contrast: no region found around the seed - empty mask")
        return np.zeros(smoothed.shape, dtype=bool)
    return ndi.binary_fill_holes(component)


def within_mm(shape, center, spacing, radius_mm: float) -> np.ndarray:
    """Ball of a given radius in mm around `center`; the centre voxel is always set."""
    grids = np.ogrid[tuple(slice(0, int(n)) for n in shape)]
    squared = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, center, spacing))
    inside = squared <= radius_mm**2
    inside[tuple(center)] = True
    return inside


def ellipsoid(radius_mm: float, spacing) -> np.ndarray:
    """Structuring element of a given radius in mm (odd shape, centre voxel set)."""
    half = [max(int(np.ceil(radius_mm / s)), 0) for s in spacing]
    grids = np.ogrid[tuple(slice(-n, n + 1) for n in half)]
    return sum((g * s / radius_mm) ** 2 for g, s in zip(grids, spacing)) <= 1.0


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndi.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes)))


def axis_hulls(body: np.ndarray) -> np.ndarray:
    """Intersection of the 2D convex hulls of `body` taken slice by slice along each
    axis. It keeps gas that is open to the air outside the body from being followed
    outwards, without closing the bladder itself."""
    result = np.ones(body.shape, dtype=bool)
    for axis in range(3):
        hull = np.zeros(body.shape, dtype=bool)
        for index in range(body.shape[axis]):
            plane = tuple(index if a == axis else slice(None) for a in range(3))
            if body[plane].any():
                hull[plane] = convex_hull_image(body[plane])
        result &= hull
    return result


def component_at(mask: np.ndarray, seed, spacing) -> np.ndarray | None:
    """The connected component of `mask` at `seed`, or the nearest one when the seed
    fell on a voxel that was removed. None when the mask is empty."""
    labels, count = ndi.label(mask)
    if count == 0:
        return None
    seed = tuple(int(v) for v in seed)
    label = int(labels[seed])
    if label == 0:
        nearest = ndi.distance_transform_edt(labels == 0, sampling=spacing, return_indices=True)[1]
        label = int(labels[tuple(nearest[(slice(None), *seed)])])
    return labels == label
