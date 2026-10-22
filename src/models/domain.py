from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from src.models.result import signed_distance, unprocessed_phi_value

SAME_STEP_TOL = 1e-6
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ComputeDomain:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]
    clipped: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(h - lo) for lo, h in zip(self.lo, self.hi, strict=True))  # type: ignore[return-value]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(lo, h) for lo, h in zip(self.lo, self.hi, strict=True))  # type: ignore[return-value]


def compute_domain(
    shape: Sequence[int],
    spacing: Sequence[float],
    seed_voxel: Sequence[int],
    domain_size_mm: float | None,
) -> ComputeDomain:
    if domain_size_mm is None:
        return ComputeDomain((0, 0, 0), tuple(int(n) for n in shape), clipped=False)  # type: ignore[arg-type]
    lo: list[int] = []
    hi: list[int] = []
    clipped = False
    for n, s, c in zip(shape, spacing, seed_voxel, strict=True):
        half = max(1, int(round(domain_size_mm / (2.0 * float(s)))))
        a, b = int(c) - half, int(c) + half + 1
        clipped = clipped or a < 0 or b > int(n)
        lo.append(max(0, a))
        hi.append(min(int(n), b))
    return ComputeDomain(tuple(lo), tuple(hi), clipped=clipped)  # type: ignore[arg-type]


def compute_grid_spacing(spacing: Sequence[float], compute_spacing_mm: float | None) -> float:
    finest = float(min(spacing))
    return finest if compute_spacing_mm is None else max(float(compute_spacing_mm), finest)


def compute_grid(
    data: np.ndarray, domain: ComputeDomain, spacing: Sequence[float], compute_spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    step = np.array([compute_spacing / float(s) for s in spacing], dtype=float)
    crop = data[domain.slices]
    if np.all(np.abs(step - 1.0) < SAME_STEP_TOL):
        return np.ascontiguousarray(crop, dtype=np.float32), np.ones(3)
    out_shape = tuple(
        int(math.floor((n - 1) / st + 1e-9)) + 1 for n, st in zip(crop.shape, step, strict=True)
    )
    image = ndi.affine_transform(
        np.asarray(crop, dtype=np.float32),
        step,
        offset=0.0,
        output_shape=out_shape,
        output=np.float32,
        order=1,
        mode="nearest",
    )
    return image, step


MIN_COMPUTE_VOXELS = 3


def check_compute_grid(shape: Sequence[int], algorithm: str, domain_size_mm: float | None, cs: float) -> None:
    if min(int(n) for n in shape) < MIN_COMPUTE_VOXELS:
        raise ValueError(
            f"{algorithm}: siatka obliczeń {list(shape)} ma mniej niż {MIN_COMPUTE_VOXELS} woksele w osi "
            f"(domain_size_mm={domain_size_mm}, krok {cs:g} mm) — zwiększ domain_size_mm albo zmniejsz "
            "compute_spacing_mm"
        )


def init_ball(shape: Sequence[int], center: Sequence[float], radius_voxels: float) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, int(n)) for n in shape)]
    dist2 = sum((g - float(c)) ** 2 for g, c in zip(grids, center, strict=True))
    ball = dist2 <= max(radius_voxels, 0.0) ** 2
    nearest = tuple(int(min(max(round(float(c)), 0), int(n) - 1)) for c, n in zip(center, shape, strict=True))
    ball[nearest] = True
    return ball.astype(np.int8)


def to_working_grid(
    mask_c: np.ndarray,
    domain: ComputeDomain,
    step: np.ndarray,
    compute_spacing: float,
    shape: Sequence[int],
    spacing: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    unit = compute_spacing / float(min(spacing))
    sdf_c = (signed_distance(mask_c) * unit).astype(np.float32)
    if np.all(np.abs(step - 1.0) < SAME_STEP_TOL):
        phi_dom = sdf_c
    else:
        phi_dom = ndi.affine_transform(
            sdf_c,
            1.0 / step,
            offset=0.0,
            output_shape=domain.shape,
            output=np.float32,
            order=1,
            mode="nearest",
        )
    del sdf_c
    phi = np.full(tuple(shape), unprocessed_phi_value(tuple(shape)), dtype=np.float32)
    phi[domain.slices] = phi_dom
    mask = np.zeros(tuple(shape), dtype=bool)
    mask[domain.slices] = phi_dom > 0
    return phi, mask


def touched_faces(mask: np.ndarray, domain: ComputeDomain, volume_shape: Sequence[int]) -> tuple[list, list]:
    inner: list[str] = []
    wall: list[str] = []
    for a in range(3):
        for side, index, at_wall in (
            ("-", 0, domain.lo[a] == 0),
            ("+", mask.shape[a] - 1, domain.hi[a] == int(volume_shape[a])),
        ):
            if np.take(mask, index, axis=a).any():
                (wall if at_wall else inner).append(f"{AXES[a]}{side}")
    return inner, wall
