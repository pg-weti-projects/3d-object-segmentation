from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import tifffile

from .volume import Volume, hash_bytes


class UnsupportedFormatError(ValueError):
    """Raised when an input suffix is not supported."""


class VolumeLoadError(ValueError):
    """Raised when a supported file cannot be interpreted as a volume."""


def load_volume(path: str | Path, format: str | None = None) -> Volume:
    """Load a DICOM, TIFF, or voxel JSON file.

    A directory is treated as a DICOM series.  JSON accepts either an object
    with a ``data`` or ``voxels`` member, or a raw nested array.
    """

    source = Path(path)
    if source.is_dir():
        return load_dicom_series(source)
    if not source.is_file():
        raise FileNotFoundError(f"Input volume does not exist: {source}")

    kind = (format or _format_from_suffix(source)).lower()
    if kind in {"dicom", "dcm"}:
        return load_dicom(source)
    if kind in {"tiff", "tif"}:
        return load_tiff(source)
    if kind in {"voxels-json", "json", "voxels"}:
        return load_json(source)
    raise UnsupportedFormatError(f"Unsupported volume format: {kind}")


def load_json(path: str | Path) -> Volume:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("data", payload.get("voxels"))
            if values is None:
                raise ValueError("JSON object must contain 'data' or 'voxels'")
            spacing = _spacing(payload.get("spacing_mm", payload.get("spacing")))
            units = str(payload.get("units", "HU"))
        else:
            values, spacing, units = payload, (1.0, 1.0, 1.0), "HU"
        data = _as_volume_array(values)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VolumeLoadError(f"Invalid voxel JSON: {source}") from exc
    return Volume(data, spacing, units, "voxels-json", hash_bytes(source.read_bytes()))


def load_tiff(path: str | Path) -> Volume:
    source = Path(path)
    try:
        data = _as_volume_array(tifffile.imread(source))
    except (OSError, ValueError) as exc:
        raise VolumeLoadError(f"Invalid TIFF volume: {source}") from exc
    return Volume(data, (1.0, 1.0, 1.0), "HU", "tiff", hash_bytes(source.read_bytes()))


def load_dicom(path: str | Path) -> Volume:
    source = Path(path)
    try:
        dataset = pydicom.dcmread(source)
        data = _as_volume_array(_dicom_pixels(dataset))
        spacing = _dicom_spacing(dataset)
    except (OSError, AttributeError, ValueError, TypeError) as exc:
        raise VolumeLoadError(f"Invalid DICOM file: {source}") from exc
    return Volume(data, spacing, "HU", "dicom", hash_bytes(source.read_bytes()))


def load_dicom_series(directory: str | Path) -> Volume:
    source = Path(directory)
    files = sorted(source.glob("*.dcm"))
    if not files:
        files = [item for item in source.iterdir() if item.is_file()]
    datasets = []
    try:
        for file in files:
            datasets.append((file, pydicom.dcmread(file)))
        datasets = [(file, ds) for file, ds in datasets if hasattr(ds, "PixelData")]
        if not datasets:
            raise ValueError("directory contains no DICOM pixel data")
        datasets.sort(key=lambda item: _slice_position(item[1]))
        arrays = [_dicom_pixels(ds) for _, ds in datasets]
        first = datasets[0][1]
        data = np.stack(arrays, axis=0)
        spacing = _dicom_spacing(first)
        digest = hash_bytes(b"".join(file.read_bytes() for file, _ in datasets))
    except (OSError, AttributeError, ValueError, TypeError) as exc:
        raise VolumeLoadError(f"Invalid DICOM series: {source}") from exc
    return Volume(data, spacing, "HU", "dicom", digest)


def _dicom_pixels(dataset: Any) -> np.ndarray:
    values = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    values = values * slope + intercept
    if values.ndim != 2:
        raise ValueError("a single DICOM slice must contain a 2D pixel array")
    return values


def _dicom_spacing(dataset: Any) -> tuple[float, float, float]:
    pixel_spacing = tuple(float(value) for value in getattr(dataset, "PixelSpacing", (1, 1)))
    thickness = float(getattr(dataset, "SpacingBetweenSlices", getattr(dataset, "SliceThickness", 1)))
    return (thickness, pixel_spacing[0], pixel_spacing[1])


def _slice_position(dataset: Any) -> float:
    position = getattr(dataset, "ImagePositionPatient", None)
    return float(position[2]) if position is not None else float(getattr(dataset, "InstanceNumber", 0))


def _as_volume_array(values: Any) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("voxel data must be a numeric 2D or 3D array")
    return np.asarray(data, dtype=np.float32)


def _spacing(value: Any) -> tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError("spacing must contain exactly three values")
    return result


def _format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "voxels-json"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix == ".dcm":
        return "dicom"
    raise UnsupportedFormatError(f"Unsupported input suffix: {suffix or '<none>'}")
