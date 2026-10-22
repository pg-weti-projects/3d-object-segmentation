from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from werkzeug.datastructures import FileStorage

from src.core import PreprocessingConfig, Volume, preprocess
from src.models.params import (
    ALGORITHM_DEFAULTS,
    ALGORITHMS,
    CHAN_VESE,
    DEFAULT_PARAMS,
    GEODESIC_ACTIVE_CONTOUR,
    GROWCUT,
)
from src.models.segment import segment_volume
from src.pipeline.experiment import AlgorithmConfig

from .processing import ALLOWED_EXTENSIONS, _load_uploaded_volume, preprocessing_config, slices_for_preview

_DATA_FILE = "volume.npy"
_META_FILE = "volume.json"
_STORE_LOCK = threading.Lock()
_MISSING = "No loaded volume on the server. Load the scan again."

PREVIEW_MAX_SIDE = 256
PREVIEW_MAX_VOXELS = 8_000_000

ALGORITHM_LABELS: dict[str, str] = {
    CHAN_VESE: "Improved Chan–Vese (2D slice by slice, Siri & Latte 2020)",
    GEODESIC_ACTIVE_CONTOUR: "Geodesic active contour (3D, MorphGAC)",
    GROWCUT: "GrowCut (3D cellular automaton, Vezhnevets & Konouchine 2005)",
}

ALGORITHM_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    CHAN_VESE: [
        ("data_term_sign", "Data term sign", "choice"),
        ("init_anchor", "Initial rectangle anchor", "choice"),
        ("remove_outside_air", "Remove outside air", "bool"),
        ("roi_radius_mm", "ROI radius [mm]", "number"),
        ("max_iterations", "Max iterations", "int"),
    ],
    GEODESIC_ACTIVE_CONTOUR: [
        ("domain_size_mm", "Compute domain size [mm] (none = whole volume)", "number"),
        ("compute_spacing_mm", "Compute grid spacing [mm] (none = finest)", "number"),
        ("balloon", "Balloon force", "number"),
        ("max_iterations", "Max iterations", "int"),
    ],
    GROWCUT: [
        ("domain_size_mm", "Compute domain size [mm] (none = whole volume)", "number"),
        ("compute_spacing_mm", "Compute grid spacing [mm] (none = finest)", "number"),
        ("bg_radius_mm", "Background shell radius [mm] (none = no shell)", "number"),
        ("fg_radius_mm", "Object seed radius [mm]", "number"),
    ],
}

_GRID_HELP = "Resolution of the computation: larger is faster but gives a coarser mask boundary."

FIELD_HELP: dict[str, dict[str, str]] = {
    CHAN_VESE: {
        "data_term_sign": (
            "'paper' (as in the publication) lets the contour only shrink, so the mask is often too small; "
            "'chan_vese' lets it grow or shrink to fit the bladder."
        ),
        "init_anchor": (
            "Where the starting contour sits in each slice: 'corner' grows from the object centre to one "
            "side (as in the publication), 'center' grows evenly around it."
        ),
        "remove_outside_air": (
            "Ignores air touching the volume edges, so the mask does not leak into the air around the fish."
        ),
        "roi_radius_mm": (
            "Size of the sample around the seed used to learn the bladder brightness; keep it inside the "
            "bladder, or the mask will leak into tissue."
        ),
        "max_iterations": (
            "How long the contour may move in each slice: with 'paper' more iterations shrink the mask, "
            "with 'chan_vese' too few may stop it before the bladder edge."
        ),
    },
    GEODESIC_ACTIVE_CONTOUR: {
        "domain_size_mm": (
            "Size of the region around the seed that is segmented; it must fit the whole bladder, bigger is "
            "slower."
        ),
        "compute_spacing_mm": _GRID_HELP,
        "balloon": (
            "How easily the contour inflates past weak edges: higher fills the bladder better but may leak "
            "out of it, 0 keeps it near the starting sphere and negative values shrink it."
        ),
        "max_iterations": (
            "How far the contour can grow: too few iterations leave the mask smaller than the bladder."
        ),
    },
    GROWCUT: {
        "domain_size_mm": (
            "Size of the region around the seed that is segmented; its walls count as background, so it "
            "must fit the whole bladder and its size shifts the mask boundary."
        ),
        "compute_spacing_mm": _GRID_HELP,
        "bg_radius_mm": (
            "Distance from the seed where background is forced; set it just outside the bladder to stop the "
            "mask leaking into tissue (none = off)."
        ),
        "fg_radius_mm": (
            "Radius around the seed that is always counted as bladder; keep it inside the bladder."
        ),
    },
}

NULLABLE_FIELDS: frozenset[str] = frozenset({"domain_size_mm", "compute_spacing_mm", "bg_radius_mm"})

FIELD_CHOICES: dict[str, tuple[str, ...]] = {
    "data_term_sign": ("paper", "chan_vese"),
    "init_anchor": ("corner", "center"),
}


def algorithm_defaults(name: str) -> dict[str, Any]:
    return dict(DEFAULT_PARAMS) if name == CHAN_VESE else dict(ALGORITHM_DEFAULTS[name])


def form_context() -> dict[str, Any]:
    return {
        "algorithms": [(name, ALGORITHM_LABELS[name]) for name in ALGORITHMS],
        "fields": ALGORITHM_FIELDS,
        "help": FIELD_HELP,
        "defaults": {name: algorithm_defaults(name) for name in ALGORITHMS},
        "choices": FIELD_CHOICES,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
    }


def parse_seed(text: str) -> tuple[int, int, int]:
    parts = [p for p in re.split(r"[,;\s]+", text or "") if p]
    if len(parts) != 3:
        raise ValueError("Seed point must be three voxel indices: z, y, x.")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("Seed point must contain numbers: z, y, x.") from exc
    if any(not v.is_integer() for v in values):
        raise ValueError("Seed point indices must be whole numbers.")
    return int(values[0]), int(values[1]), int(values[2])


def _parse_value(raw: str, kind: str, key: str, nullable: bool) -> Any:
    text = raw.strip()
    if text.lower() in {"none", "null"}:
        if not nullable:
            raise ValueError(f"{key} cannot be none.")
        return None
    if kind == "int":
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{key} must be a whole number.") from exc
    if kind == "number":
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc
    if kind == "choice" and text not in FIELD_CHOICES[key]:
        raise ValueError(f"{key} must be one of: {', '.join(FIELD_CHOICES[key])}.")
    return text


def algorithm_config(form: Mapping[str, str]) -> AlgorithmConfig:
    name = form.get("algorithm", "")
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown segmentation method: {name!r}.")
    parameters: dict[str, Any] = {}
    for key, _label, kind in ALGORITHM_FIELDS[name]:
        field = f"{name}__{key}"
        if kind == "bool":
            parameters[key] = form.get(field) == "on"
            continue
        raw = form.get(field, "")
        if raw.strip():
            parameters[key] = _parse_value(raw, kind, key, nullable=key in NULLABLE_FIELDS)
    advanced = form.get("advanced_parameters", "").strip()
    if advanced:
        try:
            extra = json.loads(advanced)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Advanced parameters are not valid JSON: {exc.msg}.") from exc
        if not isinstance(extra, dict):
            raise ValueError("Advanced parameters must be a JSON object.")
        parameters.update(extra)
    return AlgorithmConfig(name=name, parameters=parameters)


def default_store_dir() -> Path:
    return Path(tempfile.gettempdir()) / "3d-seg-segmentation"


def save_last_volume(store_dir: Path, volume: Volume, meta: Mapping[str, Any]) -> str:
    store_dir.mkdir(parents=True, exist_ok=True)
    volume_id = uuid.uuid4().hex
    record = {
        **meta,
        "volume_id": volume_id,
        "spacing_mm": list(volume.spacing_mm),
        "units": volume.units,
        "source_format": volume.source_format,
        "source_hash": volume.source_hash,
    }
    data_tmp = store_dir / f"{_DATA_FILE}.{volume_id}.tmp"
    meta_tmp = store_dir / f"{_META_FILE}.{volume_id}.tmp"
    with _STORE_LOCK:
        try:
            (store_dir / _META_FILE).unlink(missing_ok=True)
            with data_tmp.open("wb") as handle:
                np.save(handle, np.asarray(volume.data, dtype=np.float32))
            os.replace(data_tmp, store_dir / _DATA_FILE)
            meta_tmp.write_text(json.dumps(record), encoding="utf-8")
            os.replace(meta_tmp, store_dir / _META_FILE)
        finally:
            data_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
    return volume_id


def load_last_volume(store_dir: Path, volume_id: str) -> tuple[Volume, dict[str, Any]]:
    meta_path, data_path = store_dir / _META_FILE, store_dir / _DATA_FILE
    with _STORE_LOCK:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("volume_id") != volume_id:
                raise ValueError(
                    "This volume was replaced by a newer one (only the last volume is kept). "
                    "Load the scan again."
                )
            data = np.load(data_path)
            volume = Volume(
                data, tuple(meta["spacing_mm"]), meta["units"], meta["source_format"], meta["source_hash"]
            )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(_MISSING) from exc
    return volume, meta


def preview_steps(shape: tuple[int, ...]) -> tuple[int, int, int]:
    steps = [max(1, math.ceil(n / PREVIEW_MAX_SIDE)) for n in shape]
    while True:
        sizes = [math.ceil(n / step) for n, step in zip(shape, steps)]
        if math.prod(sizes) <= PREVIEW_MAX_VOXELS:
            return steps[0], steps[1], steps[2]
        steps[int(np.argmax(sizes))] += 1


def mask_preview(mask: np.ndarray, steps: tuple[int, int, int]) -> list:
    reduced = np.asarray(mask, dtype=bool)
    for axis, step in enumerate(steps):
        if step > 1:
            starts = np.arange(0, reduced.shape[axis], step)
            reduced = np.logical_or.reduceat(reduced, starts, axis=axis)
    return reduced.astype(np.uint8).tolist()


def _view(volume: Volume, meta: Mapping[str, Any]) -> dict[str, Any]:
    steps = preview_steps(volume.shape)
    preview = volume.data[:: steps[0], :: steps[1], :: steps[2]]
    return {
        "volume_id": meta["volume_id"],
        "input_shape": meta["input_shape"],
        "output_shape": list(volume.shape),
        "input_spacing": meta["input_spacing"],
        "output_spacing": list(volume.spacing_mm),
        "units": volume.units,
        "source_format": volume.source_format,
        "preprocessing": meta["preprocessing"],
        "preview_step": list(steps),
        "preview_shape": list(preview.shape),
        "preview_spacing": [s * k for s, k in zip(volume.spacing_mm, steps)],
        "slices": slices_for_preview(preview),
        "seed": None,
        "result": None,
    }


def load_uploads(uploads: list[FileStorage], form: Mapping[str, str], store_dir: Path) -> dict[str, Any]:
    suffixes = {Path(file.filename).suffix.lower() for file in uploads}
    if not suffixes.issubset(ALLOWED_EXTENSIONS):
        raise ValueError("Supported files are .dcm, .json, .tif, and .tiff.")
    config = preprocessing_config(form)
    with tempfile.TemporaryDirectory() as directory:
        volume = _load_uploaded_volume(uploads, suffixes, Path(directory))
        processed = preprocess(volume, config)
    meta: dict[str, Any] = {
        "input_shape": list(volume.shape),
        "input_spacing": list(volume.spacing_mm),
        "preprocessing": {
            "target_spacing": list(config.target_spacing) if config.target_spacing else None,
            "hu_range": list(config.hu_range) if config.hu_range else None,
            "invert_hu": config.invert_hu,
            "denoise": config.denoise,
        },
    }
    meta["volume_id"] = save_last_volume(store_dir, processed, meta)
    return _view(processed, meta)


def run_segmentation(form: Mapping[str, str], store_dir: Path) -> dict[str, Any]:
    volume, meta = load_last_volume(store_dir, form.get("volume_id", ""))
    algorithm = algorithm_config(form)
    seed = parse_seed(form.get("seed_zyx", ""))
    pre = meta["preprocessing"]
    config = PreprocessingConfig(
        hu_range=tuple(pre["hu_range"]) if pre["hu_range"] else None, invert_hu=bool(pre["invert_hu"])
    )
    start = time.perf_counter()
    result = segment_volume(volume, algorithm, seed, config)
    elapsed = time.perf_counter() - start

    voxel_mm3 = float(np.prod(volume.spacing_mm))
    mask_voxels = int(result.mask.sum())
    warnings = list(result.params_used.get("warnings", []))
    view = _view(volume, meta)
    view["seed"] = list(seed)
    view["result"] = {
        "algorithm": algorithm.name,
        "algorithm_label": ALGORITHM_LABELS[algorithm.name],
        "parameters": algorithm.parameters,
        "status": result.status,
        "mask_voxels": mask_voxels,
        "mask_volume_mm3": mask_voxels * voxel_mm3,
        "elapsed_s": elapsed,
        "n_warnings": int(result.params_used.get("n_warnings", len(warnings))),
        "warnings": warnings[:10],
        "mask": mask_preview(result.mask, tuple(view["preview_step"])),
    }
    return view


def reload_view(form: Mapping[str, str], store_dir: Path) -> dict[str, Any]:
    volume, meta = load_last_volume(store_dir, form.get("volume_id", ""))
    view = _view(volume, meta)
    try:
        view["seed"] = list(parse_seed(form.get("seed_zyx", "")))
    except ValueError:
        view["seed"] = None
    return view
