from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Volume:
    """A volume independent of its on-disk input format.

    The array is always ordered as ``(z, y, x)`` and spacing follows the same
    order.  ``source_hash`` identifies the original input bytes.
    """

    data: np.ndarray
    spacing_mm: tuple[float, float, float]
    units: str
    source_format: str
    source_hash: str

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3:
            raise ValueError(f"Volume data must be 3-dimensional, got {data.ndim}")
        if len(self.spacing_mm) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in self.spacing_mm
        ):
            raise ValueError("spacing_mm must contain three positive finite values")
        if not self.source_hash or len(self.source_hash) != 64:
            raise ValueError("source_hash must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "data", data)
        object.__setattr__(
            self, "spacing_mm", tuple(float(value) for value in self.spacing_mm)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
