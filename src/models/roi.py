from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.models.grid import SeedPoint, WorkVolume, voxel_to_mm
from src.models.params import resolve_params

logger = logging.getLogger(__name__)

RangeEstimator = Literal["percentile", "minmax"]

STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


@dataclass
class RoiResult:
    seed: SeedPoint
    roi_mask: np.ndarray
    tmin: float
    tmax: float
    estimator: str
    meta: dict[str, Any] = field(default_factory=dict)


def estimate_range(values: np.ndarray, estimator: RangeEstimator = "percentile") -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("estimate_range: brak wartości (pusty ROI)")
    if estimator == "percentile":
        lo, hi = np.percentile(values, [1.0, 99.0])
    elif estimator == "minmax":
        lo, hi = values.min(), values.max()
    else:
        raise ValueError(f"Nieznany estymator zakresu: {estimator!r}")
    return float(lo), float(hi)


def ball_slices(
    shape: tuple[int, ...],
    center: tuple[int, int, int],
    radius_mm: float,
    spacing: tuple[float, float, float],
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    sl = []
    grids = []
    for c, n, s in zip(center, shape, spacing, strict=True):
        r_vox = int(np.floor(radius_mm / s))
        lo, hi = max(c - r_vox, 0), min(c + r_vox + 1, n)
        sl.append(slice(lo, hi))
        grids.append((np.arange(lo, hi) - c) * s)
    gx, gy, gz = np.meshgrid(*grids, indexing="ij")
    local = gx**2 + gy**2 + gz**2 <= radius_mm**2 + 1e-9
    return (sl[0], sl[1], sl[2]), local


def _check_voxel_inside(voxel: tuple[int, int, int], shape: tuple[int, int, int]) -> None:
    if any(not (0 <= v < n) for v, n in zip(voxel, shape, strict=True)):
        raise ValueError(f"Punkt startowy {voxel} leży poza wolumenem o kształcie {shape}")


def _range_meta(volume: WorkVolume, tmin: float, tmax: float) -> dict[str, Any]:
    meta = volume.meta if isinstance(volume.meta, Mapping) else {}
    hu_range = meta.get("hu_range")
    if hu_range is None or len(hu_range) != 2:
        return {}
    lo, hi = float(hu_range[0]), float(hu_range[1])
    return {"tmin_hu": lo + tmin * (hi - lo), "tmax_hu": lo + tmax * (hi - lo), "hu_range": [lo, hi]}


def roi_from_seed(volume: WorkVolume, seed: SeedPoint, cfg: Mapping[str, Any] | None = None) -> RoiResult:
    params = resolve_params(cfg)
    voxel = tuple(int(v) for v in seed.voxel)
    _check_voxel_inside(voxel, volume.shape)  # type: ignore[arg-type]
    radius = float(params["roi_radius_mm"])
    sl, local = ball_slices(volume.shape, voxel, radius, volume.spacing)  # type: ignore[arg-type]
    roi_mask = np.zeros(volume.shape, dtype=bool)
    roi_mask[sl] = local
    estimator = str(params["range_estimator"])
    tmin, tmax = estimate_range(volume.data[roi_mask], estimator)  # type: ignore[arg-type]
    seed = SeedPoint(source=seed.source, voxel=voxel, mm=voxel_to_mm(volume.affine, voxel))  # type: ignore[arg-type]
    meta = {
        "mode": "manual",
        "roi_radius_mm": radius,
        "roi_voxels": int(roi_mask.sum()),
        **_range_meta(volume, tmin, tmax),
    }
    logger.info(
        "ROI: punkt %s (%.2f, %.2f, %.2f mm), %d wokseli, Tmin=%.4f Tmax=%.4f (%s)",
        voxel,
        *seed.mm,
        meta["roi_voxels"],
        tmin,
        tmax,
        estimator,
    )
    return RoiResult(seed=seed, roi_mask=roi_mask, tmin=tmin, tmax=tmax, estimator=estimator, meta=meta)
