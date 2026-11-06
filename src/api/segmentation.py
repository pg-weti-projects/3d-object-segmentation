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
from src.core.box import Box, BoxError, crop_volume, restore_mask
from src.models import box_region
from src.models.foundation import sam_med3d
from src.models.foundation.sam_med3d import assets as sam_med3d_assets
from src.models.params import (
    ALGORITHM_DEFAULTS,
    ALGORITHMS,
    CHAN_VESE,
    DEFAULT_PARAMS,
    GEODESIC_ACTIVE_CONTOUR,
    GROWCUT,
    INIT_CHOICES,
)
from src.models.segment import segment_volume
from src.pipeline.experiment import AlgorithmConfig

from .processing import ALLOWED_EXTENSIONS, _load_uploaded_volume, preprocessing_config, slices_for_preview

_DATA_FILE = "volume.npy"
_META_FILE = "volume.json"
_STORE_LOCK = threading.Lock()
_MISSING = "No loaded volume on the server. Load the scan again."

BOX_REGION = "box_region"
SAM_MED3D = "sam_med3d"
METHODS: tuple[str, ...] = (*ALGORITHMS, BOX_REGION, SAM_MED3D)

PREVIEW_MAX_SIDE = 256
PREVIEW_MAX_VOXELS = 8_000_000

ALGORITHM_LABELS: dict[str, str] = {
    CHAN_VESE: "Improved Chan–Vese (2D slice by slice, Siri & Latte 2020)",
    GEODESIC_ACTIVE_CONTOUR: "Geodesic active contour (3D, MorphGAC)",
    GROWCUT: "GrowCut (3D cellular automaton, Vezhnevets & Konouchine 2005)",
    BOX_REGION: "Box region (threshold between bladder and tissue; needs a box)",
    SAM_MED3D: "SAM-Med3D (foundation model; one click in a box with a slice range)",
}

_START_FIELD = ("init_from", "Start from (contrast is recommended)", "choice")
_CONTRAST_FRACTION_FIELD = ("contrast_fraction", "Threshold position (0-1, none = automatic)", "number")

ALGORITHM_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    CHAN_VESE: [
        _START_FIELD,
        _CONTRAST_FRACTION_FIELD,
        ("data_term_sign", "Data term sign", "choice"),
        ("init_anchor", "Initial rectangle anchor", "choice"),
        ("remove_outside_air", "Remove outside air", "bool"),
        ("roi_radius_mm", "ROI radius [mm]", "number"),
        ("max_iterations", "Max iterations", "int"),
    ],
    GEODESIC_ACTIVE_CONTOUR: [
        _START_FIELD,
        _CONTRAST_FRACTION_FIELD,
        ("domain_size_mm", "Compute domain size [mm] (none = whole volume)", "number"),
        ("compute_spacing_mm", "Compute grid spacing [mm] (none = finest)", "number"),
        ("balloon", "Balloon force", "number"),
        ("max_iterations", "Max iterations", "int"),
    ],
    GROWCUT: [
        _START_FIELD,
        _CONTRAST_FRACTION_FIELD,
        ("domain_size_mm", "Compute domain size [mm] (none = whole volume)", "number"),
        ("compute_spacing_mm", "Compute grid spacing [mm] (none = finest)", "number"),
        ("bg_radius_mm", "Background shell radius [mm] (none = no shell)", "number"),
        ("fg_radius_mm", "Object seed radius [mm]", "number"),
    ],
    BOX_REGION: [
        ("threshold_fraction", "Threshold position (0-1)", "number"),
        ("open_mm", "Opening radius [mm]", "number"),
        ("sigma_mm", "Smoothing sigma [mm]", "number"),
        ("bladder_is_dark", "Bladder darker than tissue", "bool"),
        ("use_hull", "Limit to convex hull of the tissue", "bool"),
    ],
    SAM_MED3D: [
        ("threshold", "Mask threshold (0-1)", "number"),
        ("padding", "Padding of the cube", "choice"),
        ("intensity_shift", "Intensity shift (a number or auto)", "text"),
        ("context", "Cube beyond the box", "choice"),
        ("spacing", "Cube voxel size", "choice"),
    ],
}

METHOD_NOTES: dict[str, str] = {
    SAM_MED3D: (
        "Needs a box with a slice range: its cube is scaled to the box, so a box over all slices of a long "
        "scan makes it far too coarse. Load the scan with target spacing and HU range set to none and "
        "denoising off to get the same mask as the CLI. The first run loads the weights, which takes "
        "a while; later runs are faster."
    ),
}

_GRID_HELP = "Resolution of the computation: larger is faster but gives a coarser mask boundary."
_START_HELP = (
    "'contrast' measures how dark the bladder is against the tissue in the box and starts the method "
    "from the region that follows, which is much more accurate; 'seed_ball' starts from a small sphere "
    "at the seed point."
)
_CONTRAST_FRACTION_HELP = (
    "Only for the 'contrast' start: where the threshold lies between the brightness of the bladder "
    "(seed) and of the tissue. Leave empty to let the measured contrast choose it; lower gives a "
    "smaller region, higher a larger one."
)

FIELD_HELP: dict[str, dict[str, str]] = {
    CHAN_VESE: {
        "init_from": _START_HELP,
        "contrast_fraction": _CONTRAST_FRACTION_HELP,
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
        "init_from": _START_HELP + " With 'contrast' the balloon force is off unless you set it yourself.",
        "contrast_fraction": _CONTRAST_FRACTION_HELP,
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
        "init_from": _START_HELP + " With 'contrast' the tissue around the region seeds the background.",
        "contrast_fraction": _CONTRAST_FRACTION_HELP,
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
    BOX_REGION: {
        "threshold_fraction": (
            "Where the threshold lies between the brightness of the bladder (seed) and of the tissue: "
            "0.5 is halfway, lower gives a smaller mask, higher a larger one."
        ),
        "open_mm": (
            "Cuts thin leaks out of the mask; use at least the voxel size. Too large removes small bladders."
        ),
        "sigma_mm": "Smoothing before thresholding; helps with noisy, low-contrast scans.",
        "bladder_is_dark": "Untick if the bladder is brighter than the tissue around it.",
        "use_hull": (
            "Stops the mask at the outline of the body, so gas that is open to the air outside is not "
            "followed out of the fish."
        ),
    },
    SAM_MED3D: {
        "threshold": (
            "Cut-off on the model's probability: lower gives a larger mask, higher a smaller one. 0.5 is "
            "what the model was trained with."
        ),
        "padding": (
            "What fills the cube where it reaches past the scan (and past the box with 'Cube beyond the "
            "box' = box): '0' as the model was trained, 'minimum' the lowest scan value in the cube."
        ),
        "intensity_shift": (
            "Added to every value before the model's normalisation, which only uses values above 0. 'auto' "
            "(default) finds air and tissue in the box (Otsu) and puts 0 where HU has it, but only when "
            "almost nothing in the box is below 0 (e.g. mass density, Gadus); HU scans are left as they are. "
            "A number sets the shift yourself; 0 keeps the scan as it is."
        ),
        "context": (
            "What fills the cube where it reaches past the box: 'scan' (default) shows the scan around the "
            "box, as the model saw whole scans in training; 'box' pads it with the padding value. The mask "
            "is kept inside the box either way."
        ),
        "spacing": (
            "'box' scales the cube so that the box's longest side fills it; 'trained' keeps the 1.5 mm the "
            "model was trained at, and scales only a box longer than 192 mm."
        ),
    },
}

NULLABLE_FIELDS: frozenset[str] = frozenset(
    {"domain_size_mm", "compute_spacing_mm", "bg_radius_mm", "contrast_fraction"}
)

FIELD_CHOICES: dict[str, tuple[str, ...]] = {
    "data_term_sign": ("paper", "chan_vese"),
    "init_anchor": ("corner", "center"),
    "init_from": INIT_CHOICES,
    "padding": ("0", "minimum"),
    "context": sam_med3d.pipeline.CONTEXTS,
    "spacing": sam_med3d.pipeline.SPACINGS,
}


def algorithm_defaults(name: str) -> dict[str, Any]:
    if name == SAM_MED3D:
        defaults = {key: sam_med3d.pipeline.DEFAULT_PARAMETERS[key] for key, _label, _kind in ALGORITHM_FIELDS[SAM_MED3D]}
        return {**defaults, "padding": "0" if defaults["padding"] == 0 else str(defaults["padding"])}
    if name == BOX_REGION:
        return {key: box_region.DEFAULT_PARAMETERS[key] for key, _label, _kind in ALGORITHM_FIELDS[BOX_REGION]}
    return dict(DEFAULT_PARAMS) if name == CHAN_VESE else dict(ALGORITHM_DEFAULTS[name])


def measured_contrast(params_used: dict[str, Any]) -> dict[str, Any] | None:
    """The contrast measurement of a run, when the method started from one. The
    three methods report it in slightly different places: Chan-Vese through the
    ROI it was given, the other two directly."""
    found = params_used.get("contrast")
    if not found:
        found = (params_used.get("roi_meta") or {}).get("contrast")
    if not isinstance(found, dict) or "regime" not in found:
        return None
    return {
        "regime": str(found["regime"]),
        "separability": float(found.get("separability", 0.0)),
        "cnr": float(found.get("cnr", 0.0)),
        "notes": [str(note) for note in found.get("notes", [])],
    }


def algorithm_label(name: str) -> str:
    """The label in the method list; SAM-Med3D says so when it cannot run here."""
    if name == SAM_MED3D and sam_med3d_assets.missing():
        return f"{ALGORITHM_LABELS[name]} - not installed (run {sam_med3d_assets.INSTALL_COMMAND})"
    return ALGORITHM_LABELS[name]


def form_context() -> dict[str, Any]:
    return {
        "algorithms": [(name, algorithm_label(name)) for name in METHODS],
        "notes": METHOD_NOTES,
        "fields": ALGORITHM_FIELDS,
        "help": FIELD_HELP,
        "defaults": {name: algorithm_defaults(name) for name in METHODS},
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


_BOX_FIELDS = ("box_x_min", "box_x_max", "box_y_min", "box_y_max")
_SLICE_FIELDS = ("box_z_from", "box_z_to")


def parse_box(form: Mapping[str, str], volume: Volume) -> Box | None:
    """The box drawn on the axial image (voxel indices of the preprocessed volume, max exclusive)
    as a `Box`, or None when no box was drawn. It covers every slice unless the optional slice range
    (first and last slice, both included) is given."""
    raw = [str(form.get(name, "")).strip() for name in _BOX_FIELDS]
    if not any(raw):
        return None
    try:
        x0, x1, y0, y1 = (int(value) for value in raw)
    except ValueError as exc:
        raise ValueError("The box must be four whole numbers; draw it again.") from exc
    depth, rows, columns = volume.shape
    if not (0 <= x0 < x1 <= columns and 0 <= y0 < y1 <= rows):
        raise ValueError(
            f"The box x {x0}-{x1}, y {y0}-{y1} does not fit in the volume (x < {columns}, y < {rows})."
        )
    z0, z1 = 0, depth
    slices = [str(form.get(name, "")).strip() for name in _SLICE_FIELDS]
    if any(slices):
        try:
            first, last = (int(value) for value in slices)
        except ValueError as exc:
            raise ValueError("The slice range needs both its first and last slice, as whole numbers.") from exc
        first, last = min(first, last), max(first, last)
        if not (0 <= first and last < depth):
            raise ValueError(f"The slice range {first}-{last} does not fit in the volume (slices 0-{depth - 1}).")
        z0, z1 = first, last + 1
    try:
        return Box.from_voxels((z0, y0, x0), (z1, y1, x1), volume.spacing_mm)
    except BoxError as exc:
        raise ValueError(str(exc)) from exc


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
    if name not in METHODS:
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
    if name == SAM_MED3D and parameters.get("padding") == "0":
        parameters["padding"] = 0.0  # the choice is text; the method takes a number or "minimum"
    shift = parameters.get("intensity_shift") if name == SAM_MED3D else None
    if isinstance(shift, str) and shift.strip().lower() != sam_med3d.pipeline.AUTO:
        try:
            parameters["intensity_shift"] = float(shift)
        except ValueError as exc:
            raise ValueError("intensity_shift must be a number or auto.") from exc
    elif isinstance(shift, str):
        parameters["intensity_shift"] = sam_med3d.pipeline.AUTO
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
    box = parse_box(form, volume)
    if algorithm.name in (BOX_REGION, SAM_MED3D) and box is None:
        raise ValueError("Draw a box around the bladder on the axial image first (drag with the mouse).")
    target, where, local_seed = volume, None, seed
    if box is not None:
        try:
            target, where = crop_volume(volume, box)
        except BoxError as exc:
            raise ValueError(str(exc)) from exc
        local_seed = tuple(s - w.start for s, w in zip(seed, where))
        if any(not 0 <= s < n for s, n in zip(local_seed, target.shape)):
            raise ValueError(f"The seed point {seed} lies outside the box; click inside it or draw the box again.")
    if algorithm.name == SAM_MED3D and where[0].stop - where[0].start == volume.shape[0] > 1:
        raise ValueError(
            "SAM-Med3D needs the box limited in z: set its first and last slice (the bladder's), "
            f"or its cube is scaled to all {volume.shape[0]} slices and becomes too coarse."
        )

    start = time.perf_counter()
    measured = None
    if algorithm.name == BOX_REGION:
        local_mask = box_region.segment(target, {**algorithm.parameters, "seed_zyx": local_seed})
        status, warnings, n_warnings = ("ok" if local_mask.any() else "empty"), [], 0
    elif algorithm.name == SAM_MED3D:
        # the same function and parameters as the CLI (runner): the box's contents and the click in them,
        # or with context "scan" the whole volume, the box and the click in it
        if sam_med3d.pipeline.takes_scan(algorithm.parameters):
            box_zyx = [[w.start, w.stop] for w in where]
            local_mask = sam_med3d.segment(volume, {**algorithm.parameters, "seed_zyx": list(seed), "box_zyx": box_zyx})
        else:
            local_mask = sam_med3d.segment(target, {**algorithm.parameters, "seed_zyx": list(local_seed)})
        report = local_mask.report
        status = "ok" if np.asarray(local_mask).any() else "empty"
        warnings = [
            f"Cube voxel {report['spacing_mm']:.3g} mm ({report['spacing_vs_trained']:.2g} x the 1.5 mm "
            "the model was trained at)."
        ]
        if any([pre["target_spacing"], pre["hu_range"], pre["denoise"]]):
            warnings.append(
                "This volume was resampled, clipped or denoised when it was loaded, so the mask can differ "
                "from the CLI's. Load it with target spacing and HU range none and denoising off."
            )
        rule = report.get("intensity_shift_rule")
        if rule and rule["applied"]:
            warnings.append(
                f"Intensity shift auto: {rule['shift']:.4g} (air {rule['air']:.4g}, tissue {rule['tissue']:.4g} in the box), "
                "so that air is below 0 as in HU."
            )
        warnings += list(report["warnings"])
        n_warnings = len(warnings)
    else:
        result = segment_volume(target, algorithm, local_seed, config)
        local_mask, status = result.mask, result.status
        warnings = list(result.params_used.get("warnings", []))
        n_warnings = int(result.params_used.get("n_warnings", len(warnings)))
        measured = measured_contrast(result.params_used)
    elapsed = time.perf_counter() - start
    whole = np.asarray(local_mask).shape == tuple(volume.shape)  # SAM-Med3D with context "scan"
    mask = restore_mask(local_mask, volume.shape, where) if where is not None and not whole else np.asarray(local_mask)

    voxel_mm3 = float(np.prod(volume.spacing_mm))
    mask_voxels = int(mask.sum())
    view = _view(volume, meta)
    view["seed"] = list(seed)
    view["result"] = {
        "algorithm": algorithm.name,
        "algorithm_label": algorithm_label(algorithm.name),
        "parameters": algorithm.parameters,
        "status": status,
        "box": None if where is None else {
            "x": [where[2].start, where[2].stop], "y": [where[1].start, where[1].stop], "z": [where[0].start, where[0].stop],
            "all_slices": where[0].stop - where[0].start == volume.shape[0],
        },
        "mask_voxels": mask_voxels,
        "mask_volume_mm3": mask_voxels * voxel_mm3,
        "elapsed_s": elapsed,
        "n_warnings": n_warnings,
        "warnings": warnings[:10],
        "contrast": measured,
        "mask": mask_preview(mask, tuple(view["preview_step"])),
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
