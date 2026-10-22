from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

from src.models.domain import (
    check_compute_grid,
    compute_domain,
    compute_grid,
    compute_grid_spacing,
    init_ball,
    to_working_grid,
    touched_faces,
)
from src.models.grid import Mask, WorkVolume, voxel_to_mm
from src.models.params import GEODESIC_ACTIVE_CONTOUR, resolve_gac_params
from src.models.result import SegmentationResult
from src.models.roi import RoiResult

logger = logging.getLogger(__name__)

_AUTO_THRESHOLD_PERCENTILE = 40.0


def edge_map(image: np.ndarray, alpha: float, sigma_voxels: float) -> np.ndarray:
    return inverse_gaussian_gradient(image, alpha=alpha, sigma=sigma_voxels).astype(np.float32, copy=False)


def resolve_threshold(gimage: np.ndarray, threshold: str | float, balloon: float) -> tuple[float, bool]:
    if threshold != "auto":
        return float(threshold), False
    value = float(np.percentile(gimage, _AUTO_THRESHOLD_PERCENTILE))
    if balloon == 0 or np.any(gimage > value / abs(balloon)):
        return value, False
    gmax = gimage.max()
    below = gimage[gimage < gmax]
    return (float(below.max()) * abs(balloon) if below.size else 0.0), True


class _Converged(Exception):
    pass


@dataclass
class GacEvolution:
    mask: np.ndarray
    iterations: int
    converged: bool


def evolve(
    gimage: np.ndarray,
    init: np.ndarray,
    *,
    max_iterations: int,
    smoothing: int,
    threshold: float,
    balloon: float,
    stable_iterations: int,
) -> GacEvolution:
    state: dict[str, Any] = {"prev": None, "prev2": None, "stable": 0, "calls": 0}

    def callback(u: np.ndarray) -> None:
        state["calls"] += 1
        current = u.astype(bool, copy=True)
        repeated = any(m is not None and np.array_equal(current, m) for m in (state["prev"], state["prev2"]))
        state["stable"] = state["stable"] + 1 if repeated else 0
        state["prev2"], state["prev"] = state["prev"], current
        if state["stable"] >= stable_iterations:
            raise _Converged

    converged = False
    try:
        morphological_geodesic_active_contour(
            gimage,
            max_iterations,
            init_level_set=init,
            smoothing=smoothing,
            threshold=threshold,
            balloon=balloon,
            iter_callback=callback,
        )
    except _Converged:
        converged = True
    return GacEvolution(mask=state["prev"], iterations=int(state["calls"]) - 1, converged=converged)


class GeodesicActiveContour:
    name = GEODESIC_ACTIVE_CONTOUR

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.params = resolve_gac_params(cfg)

    def segment(self, volume: WorkVolume, roi: RoiResult) -> SegmentationResult:
        p = self.params
        t_start = time.perf_counter()
        shape = volume.shape
        spacing = volume.spacing
        seed = tuple(int(v) for v in roi.seed.voxel)

        domain = compute_domain(shape, spacing, seed, p["domain_size_mm"])
        cs = compute_grid_spacing(spacing, p["compute_spacing_mm"])
        image, step = compute_grid(volume.data, domain, spacing, cs)
        check_compute_grid(image.shape, GEODESIC_ACTIVE_CONTOUR, p["domain_size_mm"], cs)
        t0 = time.perf_counter()
        sigma_vox = float(p["gac_sigma_mm"]) / cs
        gimage = edge_map(image, float(p["gac_alpha"]), sigma_vox)
        del image
        threshold, adjusted = resolve_threshold(gimage, p["threshold"], float(p["balloon"]))
        t_edge = time.perf_counter() - t0
        center = [(seed[a] - domain.lo[a]) / step[a] for a in range(3)]
        radius_vox = float(p["init_radius_mm"]) / cs
        init = init_ball(gimage.shape, center, radius_vox)
        logger.info(
            "GAC: obszar obliczeń %s–%s wokseli (%s, przycięty: %s), siatka obliczeń %s co %.3f mm, "
            "próg balonu %.4f%s",
            list(domain.lo),
            list(domain.hi),
            list(domain.shape),
            domain.clipped,
            list(gimage.shape),
            cs,
            threshold,
            " (auto, poprawiony)" if adjusted else "",
        )
        t0 = time.perf_counter()
        evo = evolve(
            gimage,
            init,
            max_iterations=int(p["max_iterations"]),
            smoothing=int(p["smoothing"]),
            threshold=threshold,
            balloon=float(p["balloon"]),
            stable_iterations=int(p["stable_iterations"]),
        )
        t_evolve = time.perf_counter() - t0
        del gimage, init
        mask_c = evo.mask

        t0 = time.perf_counter()
        phi, mask = to_working_grid(mask_c, domain, step, cs, shape, spacing)
        t_back = time.perf_counter() - t0

        mask_voxels = int(mask.sum())
        voxel_mm3 = float(np.prod(spacing))
        status = "ok" if mask_voxels else "empty"
        inner, wall = touched_faces(mask_c, domain, shape)
        warnings: list[str] = []
        if inner:
            warnings.append(
                f"maska dotyka brzegu obszaru obliczeń ({', '.join(inner)}) — obszar "
                f"{p['domain_size_mm']} mm był prawdopodobnie za mały "
                "(zwiększ algorithm.parameters.domain_size_mm)"
            )
        if wall:
            warnings.append(
                f"maska dotyka ściany wolumenu ({', '.join(wall)}) — kontur wyszedł poza rybę lub poza skan"
            )
        if not evo.converged:
            warnings.append(
                f"brak zbieżności: maska zmieniała się jeszcze po {evo.iterations} iteracjach "
                "(limit max_iterations)"
            )
        for w in warnings:
            logger.warning("GAC: %s", w)
        logger.info(
            "GAC: %d iteracji (zbieżność: %s), maska %d wokseli = %.1f mm³, status %s, %.1f s",
            evo.iterations,
            evo.converged,
            mask_voxels,
            mask_voxels * voxel_mm3,
            status,
            time.perf_counter() - t_start,
        )

        last = [h - 1 for h in domain.hi]
        params_used: dict[str, Any] = {
            "algorithm": "geodesic_active_contour_3d",
            "method": "skimage.segmentation.morphological_geodesic_active_contour (Márquez-Neila i in. 2014)",
            "seed": {
                "source": roi.seed.source,
                "voxel": list(seed),
                "mm": [float(v) for v in roi.seed.mm],
            },
            "uses_roi_range": False,
            **p,
            "spacing_mm": [float(s) for s in spacing],
            "domain": {
                "lo_voxel": list(domain.lo),
                "hi_voxel_exclusive": list(domain.hi),
                "shape_voxels": list(domain.shape),
                "first_voxel_mm": list(voxel_to_mm(volume.affine, domain.lo)),
                "last_voxel_mm": list(voxel_to_mm(volume.affine, last)),
                "clipped_to_volume": domain.clipped,
            },
            "compute_spacing_mm_used": cs,
            "compute_shape": [int(n) for n in mask_c.shape],
            "gac_sigma_voxels": sigma_vox,
            "init_radius_voxels": radius_vox,
            "threshold_used": threshold,
            "threshold_auto_adjusted": adjusted,
            "iterations": evo.iterations,
            "converged": evo.converged,
            "touches_domain_boundary": inner,
            "touches_volume_wall": wall,
            "mask_voxels": mask_voxels,
            "volume_mm3": mask_voxels * voxel_mm3,
            "timings_s": {"edge_map": t_edge, "evolution": t_evolve, "to_working_grid": t_back},
            "n_warnings": len(warnings),
            "warnings": warnings,
            "status": status,
        }
        out_mask = Mask(
            data=mask,
            affine=np.asarray(volume.affine).copy(),
            source_path=volume.source_path,
            meta={"seed_source": roi.seed.source, "algorithm": GEODESIC_ACTIVE_CONTOUR},
        )
        return SegmentationResult(mask=out_mask, phi=phi, params_used=params_used, status=status)
