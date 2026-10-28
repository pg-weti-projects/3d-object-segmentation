from __future__ import annotations

import logging
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.volume import Volume
from src.models.contrast_init import contrast_roi
from src.models.factory import make_segmenter
from src.models.grid import SeedPoint, WorkVolume, voxel_to_mm
from src.models.params import CHAN_VESE, CONTRAST, contrast_init_kwargs, resolve_algorithm_params
from src.models.result import unprocessed_phi_value
from src.models.roi import RoiResult, roi_from_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticalSegmentation:
    mask: np.ndarray
    phi: np.ndarray
    algorithm: str
    status: str
    params_used: dict[str, Any]


def _algorithm_fields(algorithm: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(algorithm, Mapping):
        name, parameters = algorithm.get("name"), algorithm.get("parameters")
    else:
        name, parameters = getattr(algorithm, "name", None), getattr(algorithm, "parameters", None)
    if not name:
        raise ValueError("algorithm: brak nazwy metody (algorithm.name)")
    return str(name), dict(parameters or {})


def to_work_volume(
    volume: Volume, hu_range: Sequence[float] | None = None, invert_hu: bool = False
) -> WorkVolume:
    data = np.asarray(volume.data, dtype=np.float32).transpose(2, 1, 0)
    sz, sy, sx = (float(s) for s in volume.spacing_mm)
    meta: dict[str, Any] = {"source_format": volume.source_format, "source_hash": volume.source_hash}
    if hu_range is not None:
        lo, hi = float(hu_range[0]), float(hu_range[1])
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"hu_range={hu_range!r}: oczekiwano dwóch skończonych wartości rosnąco")
        meta["hu_range"] = [lo, hi]
    else:
        if invert_hu:
            raise ValueError("invert_hu wymaga hu_range (tak samo jak preprocessing projektu)")
        lo, hi = float(data.min()), float(data.max())
        if not hi > lo:
            raise ValueError("Wolumen ma stałą jasność — nie da się przeskalować do [0, 1] bez hu_range")
        logger.warning(
            "Brak preprocessing.hu_range: skalowanie [0, 1] według min/max danych (%.1f, %.1f) — skala "
            "zależy od zawartości skanu (Tmin/Tmax, mapa krawędzi GAC); zalecane podanie hu_range",
            lo,
            hi,
        )
    work = np.array(data, dtype=np.float32, order="C", copy=True)
    np.clip(work, lo, hi, out=work)
    work -= np.float32(lo)
    work *= np.float32(1.0 / (hi - lo))
    if invert_hu:
        np.subtract(np.float32(1.0), work, out=work)
    meta["invert_hu_undone"] = bool(invert_hu)
    affine = np.diag([sx, sy, sz, 1.0])
    return WorkVolume(
        data=work,
        affine=affine,
        intensity_unit=volume.units,
        source_path=f"{volume.source_format}:{volume.source_hash[:12]}",
        meta=meta,
    )


def seed_from_zyx(seed_zyx: Sequence[int], work: WorkVolume) -> SeedPoint:
    if len(seed_zyx) != 3:
        raise ValueError(f"seed_zyx={seed_zyx!r}: oczekiwano trzech indeksów (z, y, x)")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) for v in seed_zyx):
        raise ValueError(f"seed_zyx={seed_zyx!r}: indeksy muszą być liczbami całkowitymi")
    z, y, x = (int(v) for v in seed_zyx)
    voxel = (x, y, z)
    if any(not (0 <= v < n) for v, n in zip(voxel, work.shape, strict=True)):
        nx, ny, nz = work.shape
        raise ValueError(f"Punkt startowy (z, y, x)={tuple(seed_zyx)} leży poza wolumenem {(nz, ny, nx)}")
    return SeedPoint(source="manual", voxel=voxel, mm=voxel_to_mm(work.affine, voxel))


def segment_volume(
    volume: Volume,
    algorithm: Any,
    seed_zyx: Sequence[int],
    preprocessing: Any = None,
) -> AnalyticalSegmentation:
    name, parameters = _algorithm_fields(algorithm)
    segmenter = make_segmenter(name, parameters)
    hu_range = getattr(preprocessing, "hu_range", None) if preprocessing is not None else None
    invert_hu = bool(getattr(preprocessing, "invert_hu", False)) if preprocessing is not None else False
    work = to_work_volume(volume, hu_range, invert_hu)
    seed = seed_from_zyx(seed_zyx, work)
    resolved = resolve_algorithm_params(name, parameters)
    if resolved.get("init_from") == CONTRAST:
        # Measured once here, so that every method starts from the same region.
        roi = contrast_roi(work, seed, **contrast_init_kwargs(resolved))
        if not roi.roi_mask.any():
            return _nothing_to_segment(volume, work, name, seed_zyx, roi)
    elif name == CHAN_VESE:
        roi = roi_from_seed(work, seed, parameters)
    else:
        roi = RoiResult(seed=seed, roi_mask=np.zeros((1, 1, 1), bool), tmin=0.0, tmax=1.0, estimator="none")
    result = segmenter.segment(work, roi)
    mask = np.ascontiguousarray(result.mask.data.transpose(2, 1, 0))
    phi = np.ascontiguousarray(result.phi.transpose(2, 1, 0))
    params_used = {
        **result.params_used,
        "seed_zyx": [int(v) for v in seed_zyx],
        "intensity_scaling": {
            "hu_range": work.meta.get("hu_range"),
            "invert_hu_undone": work.meta["invert_hu_undone"],
        },
    }
    return AnalyticalSegmentation(
        mask=mask, phi=phi, algorithm=name, status=result.status, params_used=params_used
    )


def _nothing_to_segment(
    volume: Volume, work: WorkVolume, name: str, seed_zyx: Sequence[int], roi: RoiResult
) -> AnalyticalSegmentation:
    """An empty result when the measurement found nothing to separate: the bladder
    at the seed point is as dense as the tissue around it (the Bluefin tuna scan).
    Reported as an empty mask with the measurement, rather than letting a method
    fail on a region that does not exist or return a shape it cannot justify."""
    measured = roi.meta.get("contrast", {})
    notes = list(measured.get("notes") or ["the contrast is too low to segment"])
    logger.warning("Brak kontrastu do segmentacji: %s", "; ".join(notes))
    shape = tuple(int(n) for n in volume.shape)
    return AnalyticalSegmentation(
        mask=np.zeros(shape, dtype=np.uint8),
        phi=np.full(shape, unprocessed_phi_value(shape), dtype=np.float32),
        algorithm=name,
        status="empty",
        params_used={
            "algorithm": name,
            "init_source": roi.estimator,
            "contrast": measured,
            "roi_meta": {key: value for key, value in roi.meta.items() if key != "contrast"},
            "seed_zyx": [int(v) for v in seed_zyx],
            "warnings": notes,
            "n_warnings": len(notes),
            "intensity_scaling": {
                "hu_range": work.meta.get("hu_range"),
                "invert_hu_undone": work.meta["invert_hu_undone"],
            },
        },
    )
