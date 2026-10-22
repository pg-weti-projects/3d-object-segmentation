from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

import numpy as np

from src.models.chan_vese.binarize import binarize_slice, median_size_px
from src.models.chan_vese.evolution import chan_vese_2d
from src.models.chan_vese.init_contour import auto_init_contour
from src.models.chan_vese.object_region import ObjectRegion, select_object_region
from src.models.grid import Mask, WorkVolume
from src.models.params import resolve_params
from src.models.result import SegmentationResult, signed_distance, unprocessed_phi_value
from src.models.roi import RoiResult

logger = logging.getLogger(__name__)

_MAX_STORED_WARNINGS = 50


class _BinarizationParams(NamedTuple):
    tmin: float
    tmax: float
    levels: int | None
    median_size: tuple[int, int]
    gain: float


class _SliceRun(NamedTuple):
    mask: np.ndarray
    phi: np.ndarray
    per_slice: list[dict[str, Any]]
    warnings: list[str]
    n_jobs: int


def _segment_slice(
    task: tuple[int, np.ndarray, tuple[int, int], dict[str, Any]],
) -> tuple[int, np.ndarray, dict]:
    k, binary_slice, centroid, s = task
    phi0, init_info = auto_init_contour(
        binary_slice,
        centroid,
        init_size_px=s["init_size_px"],
        step_px=s["init_step_px"],
        anchor=s["init_anchor"],
        band_px=s["init_band_px"],
    )
    result = chan_vese_2d(
        binary_slice.astype(np.float64),
        phi0,
        mu=s["mu"],
        nu=s["nu"],
        lambda_obj=s["lambda_obj"],
        lambda_bg=s["lambda_bg"],
        eps=s["eps_px"],
        dt=s["dt"],
        tol=s["tol"],
        max_iterations=s["max_iterations"],
        data_term_sign=s["data_term_sign"],
        tol_region=s["tol_region"],
    )
    phi_sdf = signed_distance(result.phi > 0).astype(np.float32)
    info = {"init": init_info, "iterations": result.iterations, "converged": result.converged}
    return k, phi_sdf, info


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _slice_meta(obj: np.ndarray, pixel_mm2: float) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    slice_ids = np.flatnonzero(obj.any(axis=(0, 1)))
    slice_meta: dict[int, dict[str, Any]] = {}
    for k in slice_ids:
        xs, ys = np.nonzero(obj[:, :, k])
        centroid = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
        area_px = int(xs.size)
        slice_meta[int(k)] = {
            "k": int(k),
            "centroid": list(centroid),
            "area_px": area_px,
            "area_mm2": area_px * pixel_mm2,
        }
        logger.debug(
            "Przekrój %d: środek ciężkości %s, pole %d px = %.2f mm²",
            k,
            centroid,
            area_px,
            area_px * pixel_mm2,
        )
    return slice_ids, slice_meta


class ImprovedChanVese:
    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.params = resolve_params(cfg)

    def segment(self, volume: WorkVolume, roi: RoiResult) -> SegmentationResult:
        binarization = self._binarization_params(volume, roi)
        region = self._object_region(volume, roi, binarization)
        settings = self._cv_settings(volume)
        run = self._run_slices(volume, region, settings, self._n_jobs())
        status = "ok" if run.mask.any() else "empty"
        if run.warnings:
            logger.warning("Segmentacja: %d ostrzeżeń (pierwsze: %s)", len(run.warnings), run.warnings[0])
        logger.info(
            "Segmentacja: %d przekrojów, iteracje CV %s, maska %d wokseli, status %s",
            len(run.per_slice),
            _summary([r["iterations"] for r in run.per_slice]),
            int(run.mask.sum()),
            status,
        )
        params_used = self._params_used(volume, roi, binarization, region, settings, run)
        out_mask = Mask(
            data=run.mask,
            affine=np.asarray(volume.affine).copy(),
            source_path=volume.source_path,
            meta={"seed_source": roi.seed.source, "algorithm": "improved_chan_vese_2d"},
        )
        return SegmentationResult(mask=out_mask, phi=run.phi, params_used=params_used, status=status)

    def _binarization_params(self, volume: WorkVolume, roi: RoiResult) -> _BinarizationParams:
        p = self.params
        sx, sy, _ = volume.spacing
        tmin, tmax = float(roi.tmin), float(roi.tmax)
        levels = None if p["intensity_levels"] is None else int(p["intensity_levels"])
        if levels is not None:
            tmin = float(np.round(tmin * (levels - 1)) / (levels - 1))
            tmax = float(np.round(tmax * (levels - 1)) / (levels - 1))
        median_size = median_size_px(float(p["median_size_mm"]), (sx, sy))
        if median_size == (1, 1):
            logger.info(
                "Okno mediany %.3f mm przy spacingu %.4f mm = 1 px — brak filtracji medianowej",
                p["median_size_mm"],
                sx,
            )
        return _BinarizationParams(tmin, tmax, levels, median_size, float(p["sigmoid_gain"]))

    def _object_region(self, volume: WorkVolume, roi: RoiResult, b: _BinarizationParams) -> ObjectRegion:
        p = self.params
        data = volume.data
        binary3d = np.empty(volume.shape, dtype=bool)
        for k in range(volume.shape[2]):
            binary3d[:, :, k] = binarize_slice(data[:, :, k], b.tmin, b.tmax, b.median_size, b.gain, b.levels)
        return select_object_region(
            binary3d,
            roi.seed.voxel,
            bool(p["remove_outside_air"]),
            spacing=volume.spacing,
            search_radius_mm=float(p["roi_radius_mm"]),
        )

    def _cv_settings(self, volume: WorkVolume) -> dict[str, Any]:
        p = self.params
        sx, sy, _ = volume.spacing
        return {
            "init_size_px": int(p["init_size_px"]),
            "init_step_px": int(p["init_step_px"]),
            "init_anchor": str(p["init_anchor"]),
            "init_band_px": int(p["init_band_px"]),
            "mu": float(p["mu"]),
            "nu": float(p["nu"]),
            "lambda_obj": float(p["lambda_obj"]),
            "lambda_bg": float(p["lambda_bg"]),
            "eps_px": max(1.0, float(p["eps_mm"]) / float(np.mean([sx, sy]))),
            "dt": float(p["dt"]),
            "tol": float(p["tol"]),
            "max_iterations": int(p["max_iterations"]),
            "data_term_sign": str(p["data_term_sign"]),
            "tol_region": str(p["tol_region"]),
        }

    def _n_jobs(self) -> int:
        n_jobs = int(self.params["n_jobs"])
        return (os.cpu_count() or 1) if n_jobs < 0 else n_jobs

    def _run_slices(
        self, volume: WorkVolume, region: ObjectRegion, settings: dict[str, Any], n_jobs: int
    ) -> _SliceRun:
        shape = volume.shape
        sx, sy, _ = volume.spacing
        binary = region.binary
        slice_ids, slice_meta = _slice_meta(region.object_mask, sx * sy)

        def tasks() -> Iterator[tuple[int, np.ndarray, tuple[int, int], dict[str, Any]]]:
            for k in slice_ids:
                m = slice_meta[int(k)]
                yield int(k), binary[:, :, k].copy(), (m["centroid"][0], m["centroid"][1]), settings

        phi_vol = np.full(shape, unprocessed_phi_value(shape), dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        all_warnings: list[str] = list(region.warnings)

        def collect(results: Iterator[tuple[int, np.ndarray, dict]]) -> None:
            for k, phi_k, info in results:
                phi_vol[:, :, k] = phi_k
                mask[:, :, k] = phi_k > 0
                rec = slice_meta[k]
                rec.update(
                    {
                        "init_size_px": info["init"]["size_px"],
                        "init_steps": info["init"]["steps"],
                        "init_force": info["init"]["force"],
                        "init_p": info["init"]["p"],
                        "init_touched_edge": info["init"]["touched_edge"],
                        "iterations": info["iterations"],
                        "converged": info["converged"],
                        "mask_area_px": int(mask[:, :, k].sum()),
                    }
                )
                for w in info["init"]["warnings"]:
                    all_warnings.append(f"przekrój {k}: {w}")

        if n_jobs > 1 and len(slice_ids) > 1:
            chunksize = max(1, len(slice_ids) // (4 * n_jobs))
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                collect(pool.map(_segment_slice, tasks(), chunksize=chunksize))
        else:
            collect(map(_segment_slice, tasks()))

        per_slice = [slice_meta[int(k)] for k in slice_ids]
        return _SliceRun(mask, phi_vol, per_slice, all_warnings, n_jobs)

    def _params_used(
        self,
        volume: WorkVolume,
        roi: RoiResult,
        b: _BinarizationParams,
        region: ObjectRegion,
        settings: dict[str, Any],
        run: _SliceRun,
    ) -> dict[str, Any]:
        p = self.params
        per_slice = run.per_slice
        sizes = [r["init_size_px"] for r in per_slice]
        iters = [r["iterations"] for r in per_slice]
        return {
            "algorithm": "improved_chan_vese_2d",
            "seed": {
                "source": roi.seed.source,
                "voxel": [int(v) for v in roi.seed.voxel],
                "mm": [float(v) for v in roi.seed.mm],
                "voxel_used": list(region.seed_voxel),
            },
            "tmin": b.tmin,
            "tmax": b.tmax,
            "tmin_roi": float(roi.tmin),
            "tmax_roi": float(roi.tmax),
            "estimator": roi.estimator,
            "roi_meta": _jsonable(roi.meta),
            "spacing_mm": [float(s) for s in volume.spacing],
            "thresholding": {
                "outside_value": b.tmin - 1.0,
                "sigmoid_center": b.tmin - 0.5,
                "sigmoid_gain": b.gain,
                "binary_threshold": 0.5,
            },
            "intensity_levels": b.levels,
            "median_size_mm": float(p["median_size_mm"]),
            "median_size_px": list(b.median_size),
            "remove_outside_air": bool(p["remove_outside_air"]),
            "n_removed_regions": region.n_removed_regions,
            "init_size_px": settings["init_size_px"],
            "init_step_px": settings["init_step_px"],
            "init_anchor": settings["init_anchor"],
            "init_band_px": settings["init_band_px"],
            "data_term_sign": settings["data_term_sign"],
            "mu": settings["mu"],
            "nu": settings["nu"],
            "lambda_obj": settings["lambda_obj"],
            "lambda_bg": settings["lambda_bg"],
            "eps_mm": float(p["eps_mm"]),
            "eps_px": settings["eps_px"],
            "dt": settings["dt"],
            "tol": settings["tol"],
            "tol_region": settings["tol_region"],
            "max_iterations": settings["max_iterations"],
            "n_jobs": run.n_jobs,
            "n_slices_processed": len(per_slice),
            "slice_range": [per_slice[0]["k"], per_slice[-1]["k"]] if per_slice else None,
            "init_size_distribution": {
                "summary": _summary([float(v) for v in sizes]),
                "counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
            },
            "iterations": _summary([float(v) for v in iters]),
            "n_converged": int(sum(r["converged"] for r in per_slice)),
            "n_init_touched_edge": int(sum(r["init_touched_edge"] for r in per_slice)),
            "n_warnings": len(run.warnings),
            "warnings": run.warnings[:_MAX_STORED_WARNINGS],
            "object_area_mm2_total": float(sum(r["area_mm2"] for r in per_slice)),
            "mask_voxels": int(run.mask.sum()),
            "per_slice": per_slice,
            "status": "ok" if run.mask.any() else "empty",
        }
