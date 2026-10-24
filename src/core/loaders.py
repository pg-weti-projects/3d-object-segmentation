from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import tifffile

from .volume import Volume, hash_bytes

logger = logging.getLogger(__name__)


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
        values, spacing, units = _json_volume_fields(payload)
        data = _as_volume_array(values)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VolumeLoadError(f"Invalid voxel JSON: {source}") from exc
    return Volume(data, spacing, units, "voxels-json", hash_bytes(source.read_bytes()))


def _json_volume_fields(payload: Any) -> tuple[Any, tuple[float, float, float], str]:
    if not isinstance(payload, dict):
        return payload, (1.0, 1.0, 1.0), "HU"

    values = payload.get("data", payload.get("voxels"))
    spacing = payload.get("spacing_mm", payload.get("spacing"))
    units = payload.get("units", "HU")
    if values is not None:
        return values, _spacing(spacing), str(units)

    specimens = payload.get("specimens")
    if not isinstance(specimens, list) or len(specimens) != 1:
        raise ValueError("JSON object must contain 'data', 'voxels', or one specimen")
    shapes = specimens[0].get("shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise ValueError("JSON specimen must contain exactly one shape")
    shape = shapes[0]
    values = shape.get("mass_density", shape.get("sound_speed_compressional"))
    if values is None:
        raise ValueError("JSON shape must contain voxel data")
    spacing = _spacing_from_units(
        shape.get("voxel_size"),
        shape.get("voxel_size_units", "mm"),
    )
    units = str(
        shape.get(
            "mass_density_units",
            shape.get("sound_speed_compressional_units", "HU"),
        )
    )
    return values, spacing, units


def load_tiff(path: str | Path) -> Volume:
    source = Path(path)
    try:
        data = _as_volume_array(tifffile.imread(source))
    except (OSError, ValueError) as exc:
        raise VolumeLoadError(f"Invalid TIFF volume: {source}") from exc
    return Volume(data, (1.0, 1.0, 1.0), "HU", "tiff", hash_bytes(source.read_bytes()))


def load_tiff_series(paths: list[str | Path]) -> Volume:
    """Load multiple TIFF slices into one ``(Z, Y, X)`` volume."""
    files = sorted((Path(path) for path in paths), key=_natural_path_key)
    if not files:
        raise VolumeLoadError("TIFF series contains no files")
    try:
        arrays = [_as_volume_array(tifffile.imread(path)) for path in files]
        if any(array.shape[0] != 1 for array in arrays):
            raise ValueError("TIFF series files must contain 2D slices")
        data = np.concatenate(arrays, axis=0)
        digest = hash_bytes(b"".join(path.read_bytes() for path in files))
    except (OSError, ValueError) as exc:
        raise VolumeLoadError("Invalid TIFF series") from exc
    return Volume(data, (1.0, 1.0, 1.0), "HU", "tiff", digest)


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
    """A folder of DICOM slices as one volume, ordered by slice position.

    The slices are read one at a time into the output array: a long series (the
    tuna body at 1 mm is 918 slices of 512x512) does not fit in memory three
    times over, which is what keeping every parsed slice, every converted slice
    and the stack of them all at once amounted to."""
    source = Path(directory)
    files = sorted(source.glob("*.dcm"))
    if not files:
        files = [item for item in source.iterdir() if item.is_file()]
    try:
        headers = []
        for file in files:
            header = pydicom.dcmread(file, stop_before_pixels=True)
            if hasattr(header, "Rows") and hasattr(header, "Columns"):
                headers.append((file, header))
        if not headers:
            raise ValueError("directory contains no DICOM pixel data")
        headers.sort(key=lambda item: _slice_position(item[1]))
        spacing = _series_spacing([header for _, header in headers])
        first = headers[0][1]
        data = np.empty((len(headers), int(first.Rows), int(first.Columns)), dtype=np.float32)
        digest = hashlib.sha256()
        for index, (file, _header) in enumerate(headers):
            content = file.read_bytes()
            digest.update(content)  # the same bytes, in the same order, as before
            data[index] = _dicom_pixels(pydicom.dcmread(io.BytesIO(content)))
    except (OSError, AttributeError, ValueError, TypeError) as exc:
        raise VolumeLoadError(f"Invalid DICOM series: {source}") from exc
    return Volume(data, spacing, "HU", "dicom", digest.hexdigest())


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


def _series_spacing(datasets: list[Any]) -> tuple[float, float, float]:
    """Spacing of a series, with the distance between slices taken from where the
    slices actually are (``ImagePositionPatient``) rather than from how thick they
    are. The two differ whenever slices overlap: the Bluefin tuna body series is
    reconstructed every 3.5 mm from 7 mm slices, and reading the thickness made
    the volume twice as long as its own reference mask.

    ``SpacingBetweenSlices``, when the scanner wrote it, is the same distance; it
    stays as the fallback for a series whose slice positions are missing."""
    first = datasets[0]
    pixel_spacing = tuple(float(value) for value in getattr(first, "PixelSpacing", (1, 1)))
    positions = []
    for dataset in datasets:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is None or len(position) != 3:
            positions = []
            break
        positions.append(float(position[2]))
    if len(positions) > 1:
        gaps = np.diff(np.sort(np.asarray(positions, dtype=float)))
        step = float(np.median(gaps))
        if np.isfinite(step) and step > 0:
            if not np.allclose(gaps, step, rtol=0.01, atol=1e-6):
                logger.warning(
                    "DICOM series: the slices are not evenly spaced (%.4g to %.4g mm apart, using the "
                    "median %.4g mm); some slices may be missing, and distances along z are then only "
                    "approximate",
                    float(gaps.min()),
                    float(gaps.max()),
                    step,
                )
            return (step, pixel_spacing[0], pixel_spacing[1])
    return _dicom_spacing(first)


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


def _spacing_from_units(value: Any, units: str) -> tuple[float, float, float]:
    spacing = _spacing(value)
    normalized_units = units.strip().lower()
    if normalized_units in {"m", "meter", "meters"}:
        return tuple(item * 1000 for item in spacing)
    if normalized_units in {"mm", "millimeter", "millimeters"}:
        return spacing
    raise ValueError(f"Unsupported voxel size units: {units}")


def _format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "voxels-json"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix == ".dcm":
        return "dicom"
    raise UnsupportedFormatError(f"Unsupported input suffix: {suffix or '<none>'}")


def _natural_path_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(character for character in stem if character.isdigit())
    return (int(digits) if digits else -1, path.name)
