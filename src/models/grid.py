from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class WorkVolume:
    data: np.ndarray
    affine: np.ndarray
    intensity_unit: str
    source_path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"WorkVolume.data musi być 3D, jest {self.data.ndim}D")
        if self.affine.shape != (4, 4):
            raise ValueError("WorkVolume.affine musi mieć kształt (4, 4)")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(s) for s in np.linalg.norm(self.affine[:3, :3], axis=0))  # type: ignore[return-value]


@dataclass(frozen=True)
class Mask:
    data: np.ndarray
    affine: np.ndarray
    source_path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Mask.data musi być 3D, jest {self.data.ndim}D")
        if self.data.dtype != np.bool_:
            object.__setattr__(self, "data", self.data.astype(bool))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class SeedPoint:
    source: Literal["manual"]
    voxel: tuple[int, int, int]
    mm: tuple[float, float, float]


def voxel_to_mm(affine: np.ndarray, voxel) -> tuple[float, float, float]:
    v = np.asarray(affine) @ np.array([*voxel, 1.0], dtype=float)
    return (float(v[0]), float(v[1]), float(v[2]))
