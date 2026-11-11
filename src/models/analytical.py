"""
src/models/analytical.py

Chan-Vese, the geodesic active contour and GrowCut under the contract the
experiment runner expects: ``segment(volume, parameters) -> uint8`` on the grid
of the volume the method was handed.

The methods themselves live behind ``segment_volume``, which the segmentation
page calls with the settings of its own form. This module is only the
translation between the two, so that one implementation answers to both the page
and the runner, and a number from the page and a number from an experiment mean
the same thing.

Two parameters belong to the translation rather than to a method:

``hu_range``   how the scan is scaled, given exactly as the preprocessing
               section of the experiment gives it. Left out, the volume is
               scaled by its own minimum and maximum, which makes an intensity
               depend on what the box happens to contain. Our mass-density
               scans (Oreo, Gadus) are not in HU and are left out on purpose.
``invert_hu``  whether that scaling inverted the scan, again as in preprocessing.

The seed point comes from ``data.seed_zyx``; the runner has already made it
relative to the cropped volume and so has already applied the box.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.core.volume import Volume
from src.models.params import CHAN_VESE, GEODESIC_ACTIVE_CONTOUR, GROWCUT
from src.models.segment import segment_volume


class _Scaling:
    """What ``segment_volume`` reads from the page's preprocessing settings."""

    def __init__(self, hu_range: Sequence[float] | None, invert_hu: bool) -> None:
        self.hu_range = hu_range
        self.invert_hu = invert_hu


class MaskWithReport(np.ndarray):
    """A mask that also says what the method did: the measured contrast, the
    parameters it used and any warnings. The runner writes it to
    ``method_report.json`` next to the mask."""

    report: dict[str, Any]


def _plain(value: Any) -> Any:
    """The report as JSON can hold it: numpy scalars and arrays become Python."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run(name: str, volume: Volume, parameters: Mapping[str, Any] | None) -> MaskWithReport:
    params = dict(parameters or {})
    seed = params.pop("seed_zyx", None)
    if seed is None:
        raise ValueError(f"'{name}' needs a point inside the bladder: give it as data.seed_zyx")
    scaling = _Scaling(params.pop("hu_range", None), bool(params.pop("invert_hu", False)))

    result = segment_volume(volume, {"name": name, "parameters": params}, seed, scaling)

    mask = np.ascontiguousarray(result.mask, dtype=np.uint8).view(MaskWithReport)
    mask.report = _plain({"algorithm": result.algorithm, "status": result.status, **result.params_used})
    return mask


def chan_vese(volume: Volume, parameters: Mapping[str, Any] | None = None) -> MaskWithReport:
    return _run(CHAN_VESE, volume, parameters)


def geodesic_active_contour(volume: Volume, parameters: Mapping[str, Any] | None = None) -> MaskWithReport:
    return _run(GEODESIC_ACTIVE_CONTOUR, volume, parameters)


def growcut(volume: Volume, parameters: Mapping[str, Any] | None = None) -> MaskWithReport:
    return _run(GROWCUT, volume, parameters)
