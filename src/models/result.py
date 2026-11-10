from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from src.models.grid import Mask


@dataclass
class SegmentationResult:
    mask: Mask
    phi: np.ndarray
    params_used: dict[str, Any]
    status: str


def signed_distance(inside: np.ndarray) -> np.ndarray:
    inside = np.asarray(inside, dtype=bool)
    if inside.all():
        return np.full(inside.shape, 0.5 + max(inside.shape), dtype=np.float64)
    if not inside.any():
        return np.full(inside.shape, -0.5 - max(inside.shape), dtype=np.float64)
    d_in = ndi.distance_transform_edt(inside)
    d_out = ndi.distance_transform_edt(~inside)
    return np.where(inside, d_in - 0.5, -(d_out - 0.5))


def unprocessed_phi_value(shape: tuple[int, ...]) -> float:
    return -0.5 - float(max(int(shape[0]), int(shape[1])))
