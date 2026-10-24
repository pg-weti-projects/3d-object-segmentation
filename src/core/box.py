from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from .volume import Volume

_EPSILON = 1e-6  # absorbs floating point noise at voxel boundaries


class BoxError(ValueError):
    """Raised for an invalid box or a box that misses the volume."""


@dataclass(frozen=True)
class Box:
    """Axis-aligned box. Voxel ``i`` of an axis with spacing ``s`` covers
    ``[i * s, (i + 1) * s)`` mm; a box includes every voxel it overlaps."""

    min_mm: tuple[float, float, float]
    max_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        low = _three_numbers(self.min_mm, "min_mm")
        high = _three_numbers(self.max_mm, "max_mm")
        if any(a < 0 for a in low):
            raise BoxError("min_mm must not be negative")
        if any(b <= a for a, b in zip(low, high)):
            raise BoxError("max_mm must be greater than min_mm on every axis")
        object.__setattr__(self, "min_mm", low)
        object.__setattr__(self, "max_mm", high)

    @classmethod
    def from_mapping(cls, mapping) -> "Box":
        """From ``{"min_mm": [z, y, x], "max_mm": [z, y, x]}`` (the YAML / JSON form)."""
        if not isinstance(mapping, dict) or not {"min_mm", "max_mm"} <= set(mapping):
            raise BoxError("box needs 'min_mm' and 'max_mm' ([z, y, x] each)")
        unknown = set(mapping) - {"min_mm", "max_mm"}
        if unknown:
            raise BoxError(f"unknown box fields: {sorted(unknown)}")
        return cls(mapping["min_mm"], mapping["max_mm"])

    def to_mapping(self) -> dict:
        return {"min_mm": list(self.min_mm), "max_mm": list(self.max_mm)}

    @classmethod
    def from_voxels(
        cls,
        start: tuple[int, int, int],
        stop: tuple[int, int, int],
        spacing_mm: tuple[float, float, float],
    ) -> "Box":
        """Box covering voxels ``start`` .. ``stop - 1`` (like slice bounds) of a grid."""
        return cls(
            tuple(a * s for a, s in zip(start, spacing_mm)),
            tuple(b * s for b, s in zip(stop, spacing_mm)),
        )

    def slices(
        self, shape: tuple[int, int, int], spacing_mm: tuple[float, float, float]
    ) -> tuple[slice, slice, slice]:
        """Voxel slices of this box on a grid, clipped to the volume.

        Raises BoxError when the box does not overlap the volume at all."""
        result = []
        for axis, (low, high, size, step) in enumerate(
            zip(self.min_mm, self.max_mm, shape, spacing_mm)
        ):
            start = max(floor(low / step + _EPSILON), 0)
            stop = min(ceil(high / step - _EPSILON), size)
            if stop <= start:
                raise BoxError(
                    f"The box lies outside the volume on axis {'zyx'[axis]} "
                    f"({low:g}-{high:g} mm, the volume is {size * step:g} mm long)"
                )
            result.append(slice(start, stop))
        return tuple(result)


def crop_volume(volume: Volume, box: Box) -> tuple[Volume, tuple[slice, slice, slice]]:
    """The part of `volume` inside `box` and the voxel slices it came from
    (needed by `restore_mask`). Spacing, units and source hash are kept."""
    where = box.slices(volume.shape, volume.spacing_mm)
    cropped = Volume(
        np.ascontiguousarray(volume.data[where]),
        volume.spacing_mm,
        volume.units,
        volume.source_format,
        volume.source_hash,
    )
    return cropped, where


def restore_mask(
    mask: np.ndarray,
    full_shape: tuple[int, int, int],
    where: tuple[slice, slice, slice],
) -> np.ndarray:
    """Place a mask computed on the cropped volume back into a zero volume of
    the original shape, so metrics and the mesh work on the full grid."""
    mask = np.asarray(mask)
    expected = tuple(s.stop - s.start for s in where)
    if mask.shape != expected:
        raise BoxError(f"mask has shape {mask.shape}, the box has shape {expected}")
    restored = np.zeros(tuple(full_shape), dtype=np.uint8)
    restored[where] = mask != 0
    return restored


def _three_numbers(values, name: str) -> tuple[float, float, float]:
    try:
        numbers = tuple(float(v) for v in values)
    except (TypeError, ValueError) as error:
        raise BoxError(f"{name} must be three numbers [z, y, x]") from error
    if len(numbers) != 3 or not all(np.isfinite(numbers)):
        raise BoxError(f"{name} must be three finite numbers [z, y, x]")
    return numbers
