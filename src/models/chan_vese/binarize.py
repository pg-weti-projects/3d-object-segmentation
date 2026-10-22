from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit


def quantize(image: np.ndarray, levels: int) -> np.ndarray:
    scale = float(levels - 1)
    return (np.round(np.asarray(image, dtype=np.float64) * scale) / scale).astype(np.float32)


def threshold_slice(image: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    out = image.copy()
    out[(image < tmin) | (image > tmax)] = np.float32(tmin - 1.0)
    return out


def median_size_px(median_size_mm: float, spacing_2d: tuple[float, float]) -> tuple[int, int]:
    sizes = []
    for s in spacing_2d:
        ratio = float(median_size_mm) / float(s)
        sizes.append(max(1, 2 * int(np.floor(ratio / 2.0)) + 1))
    return sizes[0], sizes[1]


def sigmoid(image: np.ndarray, center: float, gain: float = 10.0) -> np.ndarray:
    return expit(gain * (np.asarray(image, dtype=np.float64) - center))


def binarize_slice(
    image: np.ndarray,
    tmin: float,
    tmax: float,
    median_size: tuple[int, int] = (1, 1),
    gain: float = 10.0,
    intensity_levels: int | None = None,
) -> np.ndarray:
    if intensity_levels is not None:
        image = quantize(image, int(intensity_levels))
    thr = threshold_slice(image, tmin, tmax)
    in_range = thr >= tmin
    if not in_range.any():
        return np.zeros(thr.shape, dtype=bool)
    if median_size != (1, 1):
        thr = ndi.median_filter(thr, size=median_size, mode="nearest")
    s = sigmoid(thr, center=tmin - 0.5, gain=gain)
    return s > 0.5
