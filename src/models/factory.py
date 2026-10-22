from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from src.models.chan_vese import ImprovedChanVese
from src.models.geodesic import GeodesicActiveContour
from src.models.grid import WorkVolume
from src.models.growcut import GrowCut
from src.models.params import CHAN_VESE, GEODESIC_ACTIVE_CONTOUR, GROWCUT, algorithm_name
from src.models.result import SegmentationResult
from src.models.roi import RoiResult


class Segmenter(Protocol):
    def segment(self, volume: WorkVolume, roi: RoiResult) -> SegmentationResult: ...


def make_segmenter(name: str, parameters: Mapping[str, Any] | None = None) -> Segmenter:
    name = algorithm_name(name)
    if name == CHAN_VESE:
        return ImprovedChanVese(parameters)
    if name == GEODESIC_ACTIVE_CONTOUR:
        return GeodesicActiveContour(parameters)
    if name == GROWCUT:
        return GrowCut(parameters)
    raise ValueError(f"make_segmenter: brak implementacji algorytmu {name!r}")  # pragma: no cover
