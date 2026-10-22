from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from src.models.roi import STRUCTURE_26, ball_slices

logger = logging.getLogger(__name__)


class ObjectNotFoundError(RuntimeError):
    pass


@dataclass
class ObjectRegion:
    object_mask: np.ndarray
    binary: np.ndarray
    seed_voxel: tuple[int, int, int]
    n_removed_regions: int = 0
    warnings: list[str] = field(default_factory=list)


def _nearest_true_voxel(
    binary: np.ndarray, seed: tuple[int, int, int], radius_mm: float, spacing: tuple[float, float, float]
) -> tuple[int, int, int] | None:
    sl, ball = ball_slices(binary.shape, seed, radius_mm, spacing)
    candidates = ball & binary[sl]
    if not candidates.any():
        return None
    idx = np.argwhere(candidates)
    offsets = np.array([s.start for s in sl])
    pts = idx + offsets
    d2 = (((pts - np.asarray(seed)) * np.asarray(spacing)) ** 2).sum(axis=1)
    best = pts[int(np.argmin(d2))]
    return int(best[0]), int(best[1]), int(best[2])


def select_object_region(
    binary3d: np.ndarray,
    seed_voxel: tuple[int, int, int],
    remove_outside_air: bool = False,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    search_radius_mm: float = 3.0,
) -> ObjectRegion:
    binary = np.asarray(binary3d, dtype=bool)
    shape = binary.shape
    seed = tuple(int(v) for v in seed_voxel)
    if any(not (0 <= v < n) for v, n in zip(seed, shape, strict=True)):
        raise ObjectNotFoundError(f"Punkt startowy {seed} leży poza wolumenem {shape}")
    warnings: list[str] = []
    seed_in_original = bool(binary[seed])

    labels, n_labels = ndi.label(binary, structure=STRUCTURE_26)
    n_removed = 0
    if remove_outside_air:
        faces = np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
        touching = np.unique(faces)
        touching = touching[touching > 0]
        n_removed = int(touching.size)
        if n_removed:
            removed = np.zeros(n_labels + 1, dtype=bool)
            removed[touching] = True
            binary = binary & ~removed[labels]
            labels[removed[labels]] = 0
        logger.info("Usunięto %d obszarów dotykających ścian wolumenu", n_removed)

    if not seed_in_original:
        nearest = _nearest_true_voxel(binary, seed, search_radius_mm, spacing)  # type: ignore[arg-type]
        if nearest is None:
            after = (
                " (po usunięciu powietrza zewnętrznego, remove_outside_air=true)"
                if remove_outside_air
                else ""
            )
            raise ObjectNotFoundError(
                f"Punkt startowy {seed} leży w tle maski binarnej i w promieniu {search_radius_mm} mm "
                f"nie ma pikseli obiektu{after}"
            )
        msg = f"Punkt startowy {seed} leży w tle maski binarnej; użyto najbliższego piksela {nearest}"
        logger.warning(msg)
        warnings.append(msg)
        seed = nearest

    label = int(labels[seed])
    if label == 0:
        raise ObjectNotFoundError(
            f"Punkt startowy {seed} nie należy do żadnego obszaru maski binarnej"
            " po usunięciu powietrza zewnętrznego (remove_outside_air=true)"
        )
    object_mask = labels == label
    return ObjectRegion(
        object_mask=object_mask,
        binary=binary,
        seed_voxel=seed,  # type: ignore[arg-type]
        n_removed_regions=n_removed,
        warnings=warnings,
    )
