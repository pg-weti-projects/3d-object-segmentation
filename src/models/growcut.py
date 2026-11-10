from __future__ import annotations

import hashlib
import itertools
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from src.models.domain import (
    check_compute_grid,
    compute_domain,
    compute_grid,
    compute_grid_spacing,
    init_ball,
    to_working_grid,
    touched_faces,
)
from src.models.contrast_init import growcut_seeds, region_on_grid
from src.models.grid import Mask, WorkVolume, voxel_to_mm
from src.models.params import GROWCUT, resolve_growcut_params
from src.models.result import SegmentationResult
from src.models.roi import RoiResult

logger = logging.getLogger(__name__)

EMPTY, OBJECT, BACKGROUND = 0, 1, 2
_PAD = 2
_LABEL_CYCLE_ITERATIONS = 10


def neighborhood_offsets(ndim: int, kind: str) -> list[tuple[int, ...]]:
    if kind not in ("von_neumann", "moore"):
        raise ValueError(f"neighborhood={kind!r}; dozwolone: von_neumann, moore")
    offsets = [d for d in itertools.product((-1, 0, 1), repeat=ndim) if any(d)]
    if kind == "von_neumann":
        offsets = [d for d in offsets if sum(abs(v) for v in d) == 1]
    return offsets


def attack_coefficient(diff: np.ndarray | float, max_feature: float) -> np.ndarray:
    return _g(np.array(diff, dtype=np.float32), np.float32(1.0 / max_feature))


def _g(diff: np.ndarray, inv_max: np.float32) -> np.ndarray:
    np.abs(diff, out=diff)
    diff *= inv_max
    np.subtract(np.float32(1.0), diff, out=diff)
    np.maximum(diff, np.float32(0.0), out=diff)
    return diff


@dataclass
class GrowCutEvolution:
    labels: np.ndarray
    strengths: np.ndarray
    iterations: int
    converged: bool
    stop_reason: str = "stable"
    cycle_period: int = 0


def _shift(region: tuple[slice, ...], d: Sequence[int]) -> tuple[slice, ...]:
    return tuple(slice(s.start + di, s.stop + di) for s, di in zip(region, d, strict=True))


def _step(
    image: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    fixed: np.ndarray,
    lo: Sequence[int],
    hi: Sequence[int],
    offsets: Sequence[tuple[int, ...]],
    inv_max: np.float32,
    smooth_t1: int | None,
    smooth_t2: int | None,
) -> tuple[list[int], list[int]] | None:
    region = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    lp = labels[region]
    tp = strengths[region]
    ip = image[region]
    shape = lp.shape

    enemies: np.ndarray | None = None
    if smooth_t1 is not None or smooth_t2 is not None:
        ext = tuple(slice(s.start - 1, s.stop + 1) for s in region)
        le = labels[ext]
        count_obj = np.zeros(le.shape, dtype=np.int16)
        count_bg = np.zeros(le.shape, dtype=np.int16)
        for d in offsets:
            lq = labels[_shift(ext, d)]
            count_obj += lq == OBJECT
            count_bg += lq == BACKGROUND
        enemies = np.where(le == OBJECT, count_bg, np.where(le == BACKGROUND, count_obj, 0))
        del count_obj, count_bg, le
    inner = tuple(slice(1, 1 + n) for n in shape)

    best = np.zeros(shape, dtype=np.float32)
    best_label = np.zeros(shape, dtype=np.int8)
    for d in offsets:
        q = _shift(region, d)
        tq = strengths[q]
        if smooth_t1 is not None and enemies is not None:
            tq = np.where(enemies[_shift(inner, d)] >= smooth_t1, np.float32(0.0), tq)
        force = _g(ip - image[q], inv_max)
        force *= tq
        lq = labels[q]
        better = force > best
        better |= (force == best) & (lq < best_label) & (force > 0)
        np.copyto(best, force, where=better)
        np.copyto(best_label, lq, where=better)
    free = ~fixed[region]
    conquer = (best > tp) & free
    new_l = np.where(conquer, best_label, lp)
    new_t = np.where(conquer, best, tp)
    del best, best_label

    if smooth_t2 is not None and enemies is not None:
        forced = (enemies[inner] >= smooth_t2) & (lp != EMPTY) & free
        if forced.any():
            enemy_label = np.where(lp == OBJECT, BACKGROUND, OBJECT).astype(np.int8)
            weakest = np.full(shape, np.inf, dtype=np.float32)
            for d in offsets:
                q = _shift(region, d)
                force = _g(ip - image[q], inv_max)
                force *= strengths[q]
                cand = np.where(labels[q] == enemy_label, force, np.float32(np.inf))
                weakest = np.minimum(weakest, cand)
            new_l = np.where(forced, enemy_label, new_l)
            new_t = np.where(forced, weakest, new_t)

    changed = (new_l != lp) | (new_t != tp)
    if not changed.any():
        return None
    labels[region] = new_l
    strengths[region] = new_t
    c_lo: list[int] = []
    c_hi: list[int] = []
    for a in range(changed.ndim):
        other = tuple(i for i in range(changed.ndim) if i != a)
        idx = np.flatnonzero(changed.any(axis=other))
        c_lo.append(int(lo[a]) + int(idx[0]))
        c_hi.append(int(lo[a]) + int(idx[-1]) + 1)
    return c_lo, c_hi


def growcut_evolve(
    image: np.ndarray,
    labels: np.ndarray,
    strengths: np.ndarray,
    *,
    neighborhood: str = "von_neumann",
    max_iterations: int = 1000,
    smooth_t1: int | None = None,
    smooth_t2: int | None = None,
    max_feature: float | None = None,
    fixed: np.ndarray | None = None,
) -> GrowCutEvolution:
    img = np.asarray(image, dtype=np.float32)
    if labels.shape != img.shape or strengths.shape != img.shape:
        raise ValueError("growcut_evolve: image, labels i strengths muszą mieć ten sam kształt")
    offsets = neighborhood_offsets(img.ndim, neighborhood)
    if fixed is None:
        fixed = (labels != EMPTY) & (strengths >= 1.0)
    mf = float(img.max()) if max_feature is None and img.size else float(max_feature or 0.0)
    if not mf > 0.0:
        mf = 1.0
    inv_max = np.float32(1.0 / mf)

    img_p = np.pad(img, _PAD)
    lab = np.pad(np.asarray(labels, dtype=np.int8), _PAD)
    st = np.pad(np.asarray(strengths, dtype=np.float32), _PAD)
    fix = np.pad(np.asarray(fixed, dtype=bool), _PAD)
    inner_lo = [_PAD] * img.ndim
    inner_hi = [n + _PAD for n in img.shape]
    reach = 2 if smooth_t1 is not None else 1
    lo, hi = list(inner_lo), list(inner_hi)
    seen: dict[bytes, int] = {}
    label_history: list[np.ndarray] = []
    label_repeats = 0
    iterations = 0
    period = 0
    stop_reason = "max_iterations"
    while iterations < max_iterations:
        box = _step(img_p, lab, st, fix, lo, hi, offsets, inv_max, smooth_t1, smooth_t2)
        if box is None:
            stop_reason = "stable"
            break
        iterations += 1
        if smooth_t2 is not None:
            key = hashlib.blake2b(lab.tobytes() + st.tobytes(), digest_size=16).digest()
            if key in seen:
                stop_reason, period = "cycle", iterations - seen[key]
                break
            seen[key] = iterations
            oscillates = (
                len(label_history) == 2
                and np.array_equal(lab, label_history[0])
                and not np.array_equal(lab, label_history[1])
            )
            label_repeats = label_repeats + 1 if oscillates else 0
            if label_repeats >= _LABEL_CYCLE_ITERATIONS:
                stop_reason, period = "label_cycle", 2
                break
            label_history = [*label_history[-1:], lab.copy()]
        c_lo, c_hi = box
        lo = [max(a, c - reach) for a, c in zip(inner_lo, c_lo, strict=True)]
        hi = [min(b, c + reach) for b, c in zip(inner_hi, c_hi, strict=True)]
    core = tuple(slice(_PAD, n + _PAD) for n in img.shape)
    return GrowCutEvolution(
        labels=lab[core].copy(),
        strengths=st[core].copy(),
        iterations=iterations,
        converged=stop_reason != "max_iterations",
        stop_reason=stop_reason,
        cycle_period=period,
    )


def background_seeds(
    shape: Sequence[int],
    center: Sequence[float],
    bg_radius_voxels: float | None,
    neighborhood: str,
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.zeros(tuple(int(n) for n in shape), dtype=bool)
    for a in range(faces.ndim):
        idx: list[Any] = [slice(None)] * faces.ndim
        idx[a] = 0
        faces[tuple(idx)] = True
        idx[a] = -1
        faces[tuple(idx)] = True
    shell = np.zeros_like(faces)
    if bg_radius_voxels is not None:
        width = 1.0 if neighborhood == "von_neumann" else float(np.sqrt(faces.ndim))
        grids = np.ogrid[tuple(slice(0, int(n)) for n in shape)]
        dist = np.sqrt(sum((g - float(c)) ** 2 for g, c in zip(grids, center, strict=True)))
        shell = (dist >= bg_radius_voxels) & (dist < bg_radius_voxels + width)
    return faces, shell


class GrowCut:
    name = GROWCUT

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.params = resolve_growcut_params(cfg)

    def segment(self, volume: WorkVolume, roi: RoiResult) -> SegmentationResult:
        p = self.params
        t_start = time.perf_counter()
        shape = volume.shape
        spacing = volume.spacing
        seed = tuple(int(v) for v in roi.seed.voxel)

        domain = compute_domain(shape, spacing, seed, p["domain_size_mm"])
        cs = compute_grid_spacing(spacing, p["compute_spacing_mm"])
        image, step = compute_grid(volume.data, domain, spacing, cs)
        check_compute_grid(image.shape, GROWCUT, p["domain_size_mm"], cs)
        center = [(seed[a] - domain.lo[a]) / step[a] for a in range(3)]

        fg_radius_vox = float(p["fg_radius_mm"]) / cs
        bg_radius_vox = None if p["bg_radius_mm"] is None else float(p["bg_radius_mm"]) / cs
        faces, shell = background_seeds(image.shape, center, bg_radius_vox, p["neighborhood"])
        # With init_from="contrast" the object seeds are the inside of the
        # thresholded region and the tissue around it becomes background seeds,
        # so the labels start on both sides of the boundary rather than at a point.
        region = region_on_grid(roi, shape, domain, step, image.shape)
        if region is None:
            fg = init_ball(image.shape, center, fg_radius_vox).astype(bool)
            tissue = np.zeros(image.shape, dtype=bool)
        else:
            fg, tissue = growcut_seeds(region, image, roi)
        bg = (faces | shell | tissue) & ~fg
        labels = np.zeros(image.shape, dtype=np.int8)
        labels[fg] = OBJECT
        labels[bg] = BACKGROUND
        strengths = (labels != EMPTY).astype(np.float32)
        max_feature = float(image.max()) if image.size else 0.0
        max_feature_used = max_feature if max_feature > 0 else 1.0
        logger.info(
            "GrowCut: obszar obliczeń %s–%s wokseli (%s, przycięty: %s), siatka obliczeń %s co %.3f mm, "
            "sąsiedztwo %s, ziarna: obiekt %d, tło %d wokseli, max‖C‖ = %.4f",
            list(domain.lo),
            list(domain.hi),
            list(domain.shape),
            domain.clipped,
            list(image.shape),
            cs,
            p["neighborhood"],
            int(fg.sum()),
            int(bg.sum()),
            max_feature_used,
        )

        t0 = time.perf_counter()
        evo = growcut_evolve(
            image,
            labels,
            strengths,
            neighborhood=p["neighborhood"],
            max_iterations=int(p["max_iterations"]),
            smooth_t1=p["smooth_t1"],
            smooth_t2=p["smooth_t2"],
            max_feature=max_feature_used,
            fixed=labels != EMPTY,
        )
        t_evolve = time.perf_counter() - t0
        del image, strengths
        mask_c = evo.labels == OBJECT
        unreached = int((evo.labels == EMPTY).sum())
        n_bg = int((evo.labels == BACKGROUND).sum())

        t0 = time.perf_counter()
        phi, mask = to_working_grid(mask_c, domain, step, cs, shape, spacing)
        t_back = time.perf_counter() - t0

        mask_voxels = int(mask.sum())
        voxel_mm3 = float(np.prod(spacing))
        status = "ok" if mask_voxels else "empty"
        near = ndi.binary_dilation(mask_c, structure=np.ones((3, 3, 3), dtype=bool))
        inner, wall = touched_faces(near, domain, shape)
        touches_shell = bool((near & shell).any())
        del near
        warnings: list[str] = []
        if inner:
            warnings.append(
                f"obiekt dochodzi do ziaren tła na brzegu obszaru obliczeń ({', '.join(inner)}) — obszar "
                f"{p['domain_size_mm']} mm był prawdopodobnie za mały "
                "(zwiększ algorithm.parameters.domain_size_mm)"
            )
        if wall:
            warnings.append(
                f"obiekt dochodzi do ściany wolumenu ({', '.join(wall)}) — wyciek poza rybę lub poza skan"
            )
        if touches_shell:
            warnings.append(
                f"obiekt dochodzi do powłoki tła bg_radius_mm={p['bg_radius_mm']} — pęcherz może być większy"
            )
        if unreached:
            warnings.append(f"{unreached} wokseli nieosiągniętych (etykieta 0) — g = 0 na drodze od ziaren")
        if evo.stop_reason in ("cycle", "label_cycle"):
            warnings.append(
                f"automat w cyklu okresu {evo.cycle_period} ({evo.stop_reason}, wygładzanie T2) po "
                f"{evo.iterations} iteracjach — wynik to ostatni stan (doprecyzowanie 5)"
            )
        if not evo.converged:
            warnings.append(
                f"brak zbieżności: stan zmieniał się jeszcze po {evo.iterations} iteracjach "
                "(limit max_iterations)"
            )
        for w in warnings:
            logger.warning("GrowCut: %s", w)
        logger.info(
            "GrowCut: %d iteracji (zbieżność: %s, %s), maska %d wokseli = %.1f mm³, status %s, %.1f s",
            evo.iterations,
            evo.converged,
            evo.stop_reason,
            mask_voxels,
            mask_voxels * voxel_mm3,
            status,
            time.perf_counter() - t_start,
        )

        last = [h - 1 for h in domain.hi]
        params_used: dict[str, Any] = {
            "algorithm": "growcut_3d",
            "method": (
                "GrowCut (Vezhnevets & Konouchine, Graphicon 2005): automat komórkowy, ewolucja "
                "synchroniczna, wariant 'najsilniejszy atak'"
            ),
            "seed": {
                "source": roi.seed.source,
                "voxel": list(seed),
                "mm": [float(v) for v in roi.seed.mm],
            },
            "uses_roi_range": False,
            **p,
            "init_source": roi.estimator if region is not None else "seed_ball",
            "seeds_object_voxels": int(np.count_nonzero(fg)),
            "seeds_background_voxels": int(np.count_nonzero(bg)),
            "contrast": roi.meta.get("contrast") if region is not None else None,
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
            "fg_radius_voxels": fg_radius_vox,
            "bg_radius_voxels": bg_radius_vox,
            "seed_voxels": {"object": int(fg.sum()), "background": int(bg.sum()), "shell": int(shell.sum())},
            "max_feature": max_feature_used,
            "iterations": evo.iterations,
            "converged": evo.converged,
            "stop_reason": evo.stop_reason,
            "cycle_period": evo.cycle_period,
            "label_voxels": {"object": int(mask_c.sum()), "background": n_bg, "unreached": unreached},
            "touches_domain_boundary": inner,
            "touches_volume_wall": wall,
            "touches_background_shell": touches_shell,
            "mask_voxels": mask_voxels,
            "volume_mm3": mask_voxels * voxel_mm3,
            "timings_s": {"evolution": t_evolve, "to_working_grid": t_back},
            "n_warnings": len(warnings),
            "warnings": warnings,
            "status": status,
        }
        out_mask = Mask(
            data=mask,
            affine=np.asarray(volume.affine).copy(),
            source_path=volume.source_path,
            meta={"seed_source": roi.seed.source, "algorithm": GROWCUT},
        )
        return SegmentationResult(mask=out_mask, phi=phi, params_used=params_used, status=status)
