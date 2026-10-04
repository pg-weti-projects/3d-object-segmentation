from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .volume import Volume


@dataclass(frozen=True)
class PreprocessingConfig:
    target_spacing: tuple[float, float, float] | None = None
    hu_range: tuple[float, float] | None = None
    invert_hu: bool = False
    denoise: bool = False
    denoise_sigma: float = 1.0


def preprocess(volume: Volume, config: PreprocessingConfig) -> Volume:
    """Apply clipping, optional inversion, denoising, and isotropic resampling."""

    data = np.asarray(volume.data, dtype=np.float32)
    units = volume.units
    if config.hu_range is not None:
        low, high = config.hu_range
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("hu_range must contain finite values in ascending order")
        data = np.clip(data, low, high)
        units = "HU"
    if config.invert_hu:
        if config.hu_range is None:
            raise ValueError("invert_hu requires hu_range so the inversion is bounded")
        low, high = config.hu_range
        data = low + high - data
    if config.denoise:
        if config.denoise_sigma <= 0 or not np.isfinite(config.denoise_sigma):
            raise ValueError("denoise_sigma must be positive and finite")
        data = gaussian_filter(data, sigma=config.denoise_sigma)

    spacing = volume.spacing_mm
    if config.target_spacing is not None:
        target = tuple(float(value) for value in config.target_spacing)
        if len(target) != 3 or any(value <= 0 or not np.isfinite(value) for value in target):
            raise ValueError("target_spacing must contain three positive finite values")
        factors = tuple(old / new for old, new in zip(spacing, target))
        data = zoom(data, factors, order=1, mode="nearest", prefilter=False)
        spacing = target

    return Volume(data, spacing, units, volume.source_format, volume.source_hash)
